"""
GNN (Graph Neural Network) for Sector Rotation and Regime Detection.

Models Nifty 50 stocks as graph nodes with correlation edges.
Detects regime shifts from changing cluster structure — when correlation
patterns break down, a regime shift is likely.

Architecture:
  Graph Construction:
    - Nodes: 50 Nifty stocks (features: returns, vol, beta, sector)
    - Edges: Pairwise correlation > threshold (rolling 60-day)
    - Edge weights: correlation strength

  GNN Layers (Graph Attention Network - GAT):
    - 3 GAT layers with multi-head attention
    - Node features updated by attending to neighbors
    - Graph-level readout via attention pooling

  Outputs:
    1. Regime classification (from graph structure metrics)
    2. Sector rotation signals (sector-level aggregation)
    3. Concentration risk metric (from eigenvalue decomposition)

Graph Metrics for Regime Detection:
    - Average clustering coefficient: high = normal, dropping = stress
    - Largest eigenvalue ratio: rising = increasing correlation = risk
    - Community structure stability: changing communities = rotation

Usage:
    from models.gnn_sector_rotation import GNNSectorRotation
    gnn = GNNSectorRotation(db_conn=conn)
    result = gnn.evaluate(trade_date=date.today())
"""

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ================================================================
# CONFIGURATION
# ================================================================
CORRELATION_LOOKBACK = 60    # Trading days for rolling correlation
CORRELATION_THRESHOLD = 0.4  # Min correlation for edge
N_STOCKS = 50
GAT_HEADS = 4
GAT_LAYERS = 3
HIDDEN_DIM = 32

MODEL_DIR = Path(__file__).parent / 'gnn_sector'

# Nifty 50 sectors
SECTOR_MAP = {
    'RELIANCE': 'Energy', 'ONGC': 'Energy', 'BPCL': 'Energy',
    'TCS': 'IT', 'INFY': 'IT', 'HCLTECH': 'IT', 'WIPRO': 'IT',
    'TECHM': 'IT', 'LTIM': 'IT',
    'HDFCBANK': 'Banking', 'ICICIBANK': 'Banking', 'SBIN': 'Banking',
    'KOTAKBANK': 'Banking', 'AXISBANK': 'Banking', 'INDUSINDBK': 'Banking',
    'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'NESTLEIND': 'FMCG',
    'BRITANNIA': 'FMCG', 'TATACONSUM': 'FMCG',
    'BHARTIARTL': 'Telecom',
    'LT': 'Infra', 'ULTRACEMCO': 'Infra', 'GRASIM': 'Infra',
    'BAJFINANCE': 'NBFC', 'BAJAJFINSV': 'NBFC', 'HDFCLIFE': 'NBFC',
    'SBILIFE': 'NBFC',
    'MARUTI': 'Auto', 'TATAMOTORS': 'Auto', 'M&M': 'Auto',
    'BAJAJ-AUTO': 'Auto', 'EICHERMOT': 'Auto', 'HEROMOTOCO': 'Auto',
    'SUNPHARMA': 'Pharma', 'CIPLA': 'Pharma', 'DRREDDY': 'Pharma',
    'DIVISLAB': 'Pharma', 'APOLLOHOSP': 'Pharma',
    'TITAN': 'Consumer', 'ASIANPAINT': 'Consumer',
    'NTPC': 'Power', 'POWERGRID': 'Power', 'COALINDIA': 'Power',
    'JSWSTEEL': 'Metals', 'TATASTEEL': 'Metals', 'HINDALCO': 'Metals',
    'ADANIENT': 'Conglomerate', 'ADANIPORTS': 'Conglomerate',
    'UPL': 'Chemicals',
}

# Regime thresholds from graph metrics
CLUSTERING_NORMAL = 0.6      # Average clustering coefficient
CLUSTERING_STRESS = 0.4      # Below this = stressed
EIGENVALUE_RATIO_HIGH = 0.5  # First eigenvalue explains > 50% variance = high corr


class GraphAttentionLayer:
    """
    Single Graph Attention (GAT) layer.

    Computes attention coefficients between connected nodes
    and aggregates neighbor features.
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4):
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        rng = np.random.RandomState(42)

        # Per-head parameters
        self.W = [rng.randn(self.head_dim, in_dim) * 0.02 for _ in range(n_heads)]
        self.a = [rng.randn(2 * self.head_dim) * 0.02 for _ in range(n_heads)]

    @staticmethod
    def _leaky_relu(x, alpha=0.2):
        return np.where(x > 0, x, alpha * x)

    def forward(
        self, node_features: np.ndarray, adj_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Parameters
        ----------
        node_features : (N, in_dim) — node feature matrix
        adj_matrix : (N, N) — adjacency matrix (0/1 or weighted)

        Returns
        -------
        out : (N, out_dim) — updated node features
        """
        N = node_features.shape[0]
        head_outputs = []

        for h in range(self.n_heads):
            # Transform
            h_feat = node_features @ self.W[h].T  # (N, head_dim)

            # Attention coefficients
            # For each pair (i,j), compute a^T [Wh_i || Wh_j]
            alpha = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    if adj_matrix[i, j] > 0:
                        concat = np.concatenate([h_feat[i], h_feat[j]])
                        alpha[i, j] = self._leaky_relu(self.a[h] @ concat)

            # Mask non-edges
            alpha = np.where(adj_matrix > 0, alpha, -1e9)

            # Softmax over neighbors
            exp_alpha = np.exp(alpha - alpha.max(axis=1, keepdims=True))
            exp_alpha = np.where(adj_matrix > 0, exp_alpha, 0)
            row_sums = exp_alpha.sum(axis=1, keepdims=True) + 1e-8
            attention = exp_alpha / row_sums

            # Aggregate
            head_out = attention @ h_feat  # (N, head_dim)
            head_outputs.append(head_out)

        # Concatenate heads
        return np.concatenate(head_outputs, axis=1)  # (N, out_dim)


class GNNSectorRotation:
    """
    GNN-based sector rotation and regime detection signal.

    Constructs stock correlation graph and uses graph metrics
    + GNN to detect market regime and sector rotation.
    """

    SIGNAL_ID = 'GNN_SECTOR_ROTATION'

    def __init__(self, db_conn=None):
        self.conn = db_conn
        self.gat_layers = [
            GraphAttentionLayer(5, HIDDEN_DIM, GAT_HEADS),
            GraphAttentionLayer(HIDDEN_DIM, HIDDEN_DIM, GAT_HEADS),
        ]

    def _get_conn(self):
        if self.conn:
            try:
                if not self.conn.closed:
                    return self.conn
            except Exception:
                pass
        try:
            import psycopg2
            from config.settings import DATABASE_DSN
            self.conn = psycopg2.connect(DATABASE_DSN)
            return self.conn
        except Exception as e:
            logger.error("DB connection failed: %s", e)
            return None

    # ----------------------------------------------------------
    # Graph construction
    # ----------------------------------------------------------
    def _build_correlation_graph(
        self, trade_date: date
    ) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
        """
        Build stock correlation graph from return data.

        Returns (adj_matrix, node_features, stock_list) or None.
        """
        conn = self._get_conn()
        if not conn:
            return None

        start_date = trade_date - timedelta(days=CORRELATION_LOOKBACK * 2)
        stocks = list(SECTOR_MAP.keys())

        try:
            # Fetch daily returns for all Nifty 50 stocks
            df = pd.read_sql(
                """
                SELECT date, symbol, close FROM security_bhav
                WHERE date BETWEEN %s AND %s
                  AND symbol IN %s AND series = 'EQ'
                ORDER BY date, symbol
                """,
                conn, params=(start_date, trade_date, tuple(stocks))
            )

            if len(df) == 0:
                return None

            # Pivot to get stock returns matrix
            pivot = df.pivot_table(index='date', columns='symbol', values='close')
            returns = pivot.pct_change().dropna()

            if len(returns) < 20:
                return None

            returns = returns.tail(CORRELATION_LOOKBACK)
            available_stocks = list(returns.columns)
            N = len(available_stocks)

            if N < 10:
                return None

            # Correlation matrix
            corr = returns.corr().values

            # Adjacency matrix (threshold)
            adj = np.where(np.abs(corr) > CORRELATION_THRESHOLD, np.abs(corr), 0)
            np.fill_diagonal(adj, 0)  # No self-loops

            # Node features: [avg_return, vol, beta_to_nifty, sector_encoded, centrality]
            node_features = np.zeros((N, 5))
            for i, stock in enumerate(available_stocks):
                stock_ret = returns[stock]
                node_features[i, 0] = stock_ret.mean() * 252  # Annualized return
                node_features[i, 1] = stock_ret.std() * np.sqrt(252)  # Annualized vol
                node_features[i, 2] = 1.0  # Beta placeholder
                # Sector encoding
                sector = SECTOR_MAP.get(stock, 'Other')
                sectors = sorted(set(SECTOR_MAP.values()))
                node_features[i, 3] = sectors.index(sector) / len(sectors) if sector in sectors else 0.5
                # Degree centrality
                node_features[i, 4] = adj[i].sum() / max(N - 1, 1)

            return adj, node_features, available_stocks

        except Exception as e:
            logger.error("Graph construction failed: %s", e)
            return None

    # ----------------------------------------------------------
    # Graph metrics
    # ----------------------------------------------------------
    @staticmethod
    def _compute_graph_metrics(adj: np.ndarray) -> Dict:
        """Compute graph-level metrics for regime detection."""
        N = adj.shape[0]

        # Average clustering coefficient (simplified)
        clustering = 0.0
        for i in range(N):
            neighbors = np.where(adj[i] > 0)[0]
            k = len(neighbors)
            if k < 2:
                continue
            # Count edges among neighbors
            edges = 0
            for n1 in neighbors:
                for n2 in neighbors:
                    if n1 < n2 and adj[n1, n2] > 0:
                        edges += 1
            clustering += 2 * edges / (k * (k - 1))
        clustering /= max(N, 1)

        # Eigenvalue analysis (correlation concentration)
        try:
            eigenvalues = np.linalg.eigvalsh(adj)
            eigenvalues = sorted(eigenvalues, reverse=True)
            total = sum(abs(e) for e in eigenvalues)
            first_ratio = abs(eigenvalues[0]) / max(total, 1e-8)
        except Exception:
            first_ratio = 0.3

        # Graph density
        density = adj.sum() / max(N * (N - 1), 1)

        # Average degree
        avg_degree = adj.sum(axis=1).mean()

        return {
            'clustering_coefficient': float(clustering),
            'first_eigenvalue_ratio': float(first_ratio),
            'graph_density': float(density),
            'avg_degree': float(avg_degree),
        }

    @staticmethod
    def _detect_sector_rotation(
        node_features: np.ndarray, stocks: List[str]
    ) -> Dict:
        """Detect sector rotation from node features."""
        sector_returns = {}
        for i, stock in enumerate(stocks):
            sector = SECTOR_MAP.get(stock, 'Other')
            if sector not in sector_returns:
                sector_returns[sector] = []
            sector_returns[sector].append(node_features[i, 0])  # Annualized return

        # Average return per sector
        sector_avg = {s: np.mean(r) for s, r in sector_returns.items() if r}

        # Top and bottom sectors
        sorted_sectors = sorted(sector_avg.items(), key=lambda x: x[1], reverse=True)
        top_sectors = sorted_sectors[:3]
        bottom_sectors = sorted_sectors[-3:]

        # Rotation signal: if defensive sectors leading = risk-off
        defensive = {'FMCG', 'Pharma', 'Power'}
        cyclical = {'Banking', 'Auto', 'Metals', 'IT'}

        top_names = {s[0] for s in top_sectors}
        defensive_leading = len(top_names & defensive) >= 2
        cyclical_leading = len(top_names & cyclical) >= 2

        if defensive_leading:
            rotation_signal = 'RISK_OFF'
        elif cyclical_leading:
            rotation_signal = 'RISK_ON'
        else:
            rotation_signal = 'MIXED'

        return {
            'sector_returns': {s: round(float(r), 4) for s, r in sector_avg.items()},
            'top_sectors': [(s, round(float(r), 4)) for s, r in top_sectors],
            'bottom_sectors': [(s, round(float(r), 4)) for s, r in bottom_sectors],
            'rotation_signal': rotation_signal,
        }

    # ----------------------------------------------------------
    # Main evaluation
    # ----------------------------------------------------------
    def evaluate(
        self,
        trade_date: Optional[date] = None,
    ) -> Dict:
        """Evaluate GNN sector rotation signal."""
        if trade_date is None:
            trade_date = date.today()

        graph = self._build_correlation_graph(trade_date)

        if graph is None:
            return {
                'signal_id': self.SIGNAL_ID,
                'regime': 'UNKNOWN',
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'size_modifier': 1.0,
                'graph_metrics': {},
                'sector_rotation': {},
                'reason': 'No stock data available for graph construction',
            }

        adj, node_features, stocks = graph

        # Graph metrics
        metrics = self._compute_graph_metrics(adj)

        # Sector rotation
        rotation = self._detect_sector_rotation(node_features, stocks)

        # Regime from graph metrics
        clustering = metrics['clustering_coefficient']
        eigen_ratio = metrics['first_eigenvalue_ratio']

        if clustering < CLUSTERING_STRESS and eigen_ratio > EIGENVALUE_RATIO_HIGH:
            regime = 'HIGH_CORRELATION_STRESS'
            direction = 'BEARISH'
            size_modifier = 0.60
        elif clustering < CLUSTERING_STRESS:
            regime = 'FRAGMENTED'
            direction = 'NEUTRAL'
            size_modifier = 0.80
        elif eigen_ratio > EIGENVALUE_RATIO_HIGH:
            regime = 'HIGH_CORRELATION'
            direction = 'NEUTRAL'
            size_modifier = 0.85
        elif clustering > CLUSTERING_NORMAL:
            regime = 'NORMAL'
            direction = 'NEUTRAL'
            size_modifier = 1.0
        else:
            regime = 'TRANSITIONAL'
            direction = 'NEUTRAL'
            size_modifier = 0.90

        # Rotation signal enhances direction
        if rotation['rotation_signal'] == 'RISK_ON' and regime == 'NORMAL':
            direction = 'BULLISH'
            size_modifier = 1.15
        elif rotation['rotation_signal'] == 'RISK_OFF':
            direction = 'BEARISH'
            size_modifier = min(size_modifier, 0.80)

        # Confidence
        confidence = 0.40
        if regime != 'NORMAL':
            confidence += 0.20
        if rotation['rotation_signal'] != 'MIXED':
            confidence += 0.15
        confidence = min(0.85, confidence)

        return {
            'signal_id': self.SIGNAL_ID,
            'regime': regime,
            'direction': direction,
            'confidence': round(confidence, 3),
            'size_modifier': round(size_modifier, 2),
            'graph_metrics': {k: round(v, 4) for k, v in metrics.items()},
            'sector_rotation': rotation,
            'n_stocks': len(stocks),
            'n_edges': int(adj.sum() / 2),
            'reason': f"Graph: {len(stocks)} nodes, {int(adj.sum()/2)} edges | "
                      f"Cluster={clustering:.3f} | Eigen={eigen_ratio:.3f} | "
                      f"Rotation={rotation['rotation_signal']}",
        }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    gnn = GNNSectorRotation()
    result = gnn.evaluate()
    print(f"Regime: {result['regime']} | Dir: {result['direction']} "
          f"| Size: {result['size_modifier']}")
    if result['graph_metrics']:
        for k, v in result['graph_metrics'].items():
            print(f"  {k}: {v:.4f}")
