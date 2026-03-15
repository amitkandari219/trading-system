#!/usr/bin/env python3
"""
SSRN Automated Paper Downloader
For Nifty F&O Systematic Trading System

Downloads all relevant SSRN papers across 8 categories:
  1. India-specific (NSE/Nifty momentum, calendar anomalies)
  2. Core momentum (Moskowitz, Jegadeesh, Daniel)
  3. Volatility management (Moreira, Barroso)
  4. Position sizing (Vince, Kelly-related)
  5. Mean reversion (Chan, pairs trading)
  6. Options and volatility surface (Gatheral, Natenberg-related)
  7. Regime detection (bear markets, VIX-based)
  8. Behavioural finance (Kahneman, Thaler-related)

Usage:
  python ssrn_downloader.py                    # download all
  python ssrn_downloader.py --test             # test 1 paper
  python ssrn_downloader.py --category india   # one category
  python ssrn_downloader.py --list             # list all papers
  python ssrn_downloader.py --download-only    # skip ingest
  python ssrn_downloader.py --failed           # retry failed only
"""

import os
import sys
import time
import json
import argparse
import requests
from pathlib import Path
from datetime import datetime

ALL_PAPERS = [
    {"id":"IND_001","abstract_id":"5116091","title":"Quantitative Analysis of Price Momentum in Indian Equity Markets","author":"Pelluri (2025)","category":"india","priority":1,"relevance":"Formation/holding period parameters for Nifty momentum"},
    {"id":"IND_002","abstract_id":"3510433","title":"Implementing Systematic Long-only Momentum: Evidence From India","author":"Raju, Chandrasekaran (2020)","category":"india","priority":1,"relevance":"+10.7% alpha over Nifty100 after real costs"},
    {"id":"IND_003","abstract_id":"5367997","title":"Monthly High Timing Patterns in NIFTY 50: Late-Month Bias","author":"Kamarajugadda, Jada (2025)","category":"india","priority":1,"relevance":"Late-month bias days 21-31 p<2.2e-16"},
    {"id":"IND_004","abstract_id":"2785553","title":"Momentum and Contrarian Profitability: Indian Stock Market","author":"Maheshwari, Dhankar (2016)","category":"india","priority":1,"relevance":"Both momentum and mean reversion confirmed on NSE"},
    {"id":"IND_005","abstract_id":"1590784","title":"Futures Expiration Day Effects on Spot Market Volatility: NSE","author":"Kamaiah, Sakthivel (2008)","category":"india","priority":1,"relevance":"Confirms expiry day volatility effect"},
    {"id":"IND_006","abstract_id":"1150080","title":"January Anomaly and Return Predictability: Evidence from India","author":"Elango, Pandey (2008)","category":"india","priority":1,"relevance":"March negative, Nov/Dec positive calendar signals"},
    {"id":"IND_007","abstract_id":"3000963","title":"Nifty Futures Rollover Strategies","author":"Slivka, Yang, Wan (2017)","category":"india","priority":2,"relevance":"Expiry rollover mechanics"},
    {"id":"IND_008","abstract_id":"2785541","title":"Contrarian and Momentum Profits in Indian Stock Market 1997-2013","author":"Dhankar, Maheshwari (2014)","category":"india","priority":2,"relevance":"Long time series confirms momentum on NSE"},
    {"id":"IND_009","abstract_id":"3065621","title":"Weekly Momentum Study on Indian Stock Markets","author":"Sapate (2017)","category":"india","priority":2,"relevance":"Weekly momentum validates DRY_20 holding period"},
    {"id":"IND_010","abstract_id":"4766370","title":"Options Selling Using Machine Learning on NSE","author":"Joshi, Venkateswaran, Bhattacharyya (2024)","category":"india","priority":2,"relevance":"Options selling signals NSE-specific"},
    {"id":"IND_011","abstract_id":"3323746","title":"Options Trading Strategy Using CCI for NSE Nifty Options","author":"Shah (2019)","category":"india","priority":2,"relevance":"CCI-based Nifty options momentum"},
    {"id":"IND_012","abstract_id":"2371980","title":"Market Efficiency and Behavioural Finance: NSE Evidence 1995-2010","author":"Subodh (2013)","category":"india","priority":3,"relevance":"Academic justification for exploitable anomalies on NSE"},
    {"id":"MOM_001","abstract_id":"2089463","title":"Time Series Momentum","author":"Moskowitz, Ooi, Pedersen (2012)","category":"momentum","priority":1,"relevance":"THE core paper — 12-month lookback momentum"},
    {"id":"MOM_002","abstract_id":"2049939","title":"Betting Against Beta","author":"Frazzini, Pedersen (2014)","category":"momentum","priority":1,"relevance":"Low-beta anomaly — independent signal"},
    {"id":"MOM_003","abstract_id":"2371227","title":"Momentum Crashes","author":"Daniel, Moskowitz (2016)","category":"momentum","priority":1,"relevance":"When momentum fails — VIX/panic state detection"},
    {"id":"MOM_004","abstract_id":"4069575","title":"Short-Term Reversals and Longer-Term Momentum Around the World","author":"Jegadeesh, Luo, Subrahmanyam, Titman (2022)","category":"momentum","priority":1,"relevance":"Reversals at short horizons — mean reversion signal design"},
    {"id":"MOM_005","abstract_id":"4342008","title":"Momentum, Market Volatility, and Reversal","author":"Butt, Kolari, Sadaqat (2023)","category":"momentum","priority":2,"relevance":"Switch momentum to reversal when VIX high"},
    {"id":"MOM_006","abstract_id":"1104491","title":"The 52-Week High and Momentum Investing","author":"George, Hwang (2004)","category":"momentum","priority":2,"relevance":"52-week high proximity as momentum predictor"},
    {"id":"MOM_007","abstract_id":"3240609","title":"Momentum Enhancing Strategies for Global Equity Markets","author":"Multiple authors (2018)","category":"momentum","priority":2,"relevance":"18 characteristics that enhance momentum"},
    {"id":"MOM_008","abstract_id":"1968996","title":"Time Series Momentum Across Asset Classes","author":"Baltas, Kosowski (2012)","category":"momentum","priority":3,"relevance":"TS momentum diversified across assets"},
    {"id":"MOM_009","abstract_id":"5933974","title":"Nonlinear Time Series Momentum","author":"Moskowitz, Sabbatucci, Tamoni, Uhl (2025)","category":"momentum","priority":2,"relevance":"ML-based nonlinear momentum"},
    {"id":"VOL_001","abstract_id":"2659431","title":"Volatility-Managed Portfolios","author":"Moreira, Muir (2017)","category":"volatility","priority":1,"relevance":"Scale inverse to last month variance"},
    {"id":"VOL_002","abstract_id":"2673124","title":"Risk-Adjusted Momentum Strategies (RAMOM)","author":"Dudler, Gmur, Malamud (2015)","category":"volatility","priority":2,"relevance":"Outperforms TSMOM by risk-adjusting"},
    {"id":"VOL_003","abstract_id":"2844140","title":"Risk-Managed Industry Momentum and Momentum Crashes","author":"Multiple authors (2017)","category":"volatility","priority":2,"relevance":"Vol scaling eliminates momentum crashes"},
    {"id":"VOL_004","abstract_id":"3595147","title":"Factor Momentum, Investor Sentiment, and Option-Implied Volatility","author":"Rutanen, Grobys (2020)","category":"volatility","priority":3,"relevance":"VIX as sentiment signal"},
    {"id":"VOL_005","abstract_id":"2140091","title":"Momentum and Volatility Weighting","author":"Baltas, Kosowski (2013)","category":"volatility","priority":3,"relevance":"Yang-Zhang volatility estimator optimal"},
    {"id":"RISK_001","abstract_id":"2090766","title":"Optimal Position Sizing in a Kelly Framework","author":"Nekrasov (2014)","category":"risk","priority":1,"relevance":"Kelly criterion fractional implementation"},
    {"id":"RISK_002","abstract_id":"3190768","title":"Revisiting the Deflated Sharpe Ratio","author":"Bailey, Lopez de Prado (2018)","category":"risk","priority":1,"relevance":"DSR formula implementation"},
    {"id":"RISK_003","abstract_id":"2276632","title":"The Sharpe Ratio Efficient Frontier","author":"Bailey, Lopez de Prado (2012)","category":"risk","priority":2,"relevance":"Sharpe ratio statistics for finite samples"},
    {"id":"RISK_004","abstract_id":"3616950","title":"The False Strategy Theorem","author":"Lopez de Prado (2019)","category":"risk","priority":1,"relevance":"Multiple testing — false discovery problem"},
    {"id":"RISK_005","abstract_id":"2701337","title":"The Probability of Backtest Overfitting","author":"Bailey, Borwein, Lopez de Prado, Zhu (2014)","category":"risk","priority":1,"relevance":"Quantifies overfitting risk"},
    {"id":"MR_001","abstract_id":"1968996","title":"Statistical Arbitrage and Mean Reversion in Asian Markets","author":"Multiple authors","category":"mean_reversion","priority":2,"relevance":"Mean reversion in Asian equity futures"},
    {"id":"MR_002","abstract_id":"4708400","title":"Efficacy of Mean Reversion Strategy Using True Strength Index","author":"Requejo (2024)","category":"mean_reversion","priority":2,"relevance":"TSI-based mean reversion — new indicator"},
    {"id":"MR_003","abstract_id":"687205","title":"Momentum and Mean-Reversion in Strategic Asset Allocation","author":"Koijen, Rodriguez, Sbuelz (2006)","category":"mean_reversion","priority":3,"relevance":"Optimal mix of momentum and mean reversion"},
    {"id":"OPT_001","abstract_id":"3111334","title":"Moving Average Distance and Stock Return Predictability","author":"Avramov, Kaplanski, Subrahmanyam (2018)","category":"options","priority":2,"relevance":"MA distance predictor — new signal"},
    {"id":"OPT_002","abstract_id":"2744766","title":"Time Series Momentum and Volatility Scaling Post-Crisis","author":"Multiple authors (2015)","category":"options","priority":3,"relevance":"Post-2009 performance analysis"},
    {"id":"OPT_003","abstract_id":"3595147","title":"Option Implied Volatility and Factor Momentum","author":"Rutanen, Grobys (2020)","category":"options","priority":2,"relevance":"VIX-based factor timing"},
    {"id":"REG_001","abstract_id":"2371227","title":"Momentum Crashes — Panic State Detection","author":"Daniel, Moskowitz (2016)","category":"regime","priority":1,"relevance":"Panic state = high VIX + market decline"},
    {"id":"REG_002","abstract_id":"4342008","title":"Switching Strategy: Momentum to Reversal Based on Volatility","author":"Butt, Kolari, Sadaqat (2023)","category":"regime","priority":1,"relevance":"Switch rule: high VIX → use reversal signal"},
    {"id":"REG_003","abstract_id":"5278107","title":"Systematic Trading with Relative Moving Average (RMA)","author":"Bloch (2025)","category":"regime","priority":2,"relevance":"RMA framework for regime shifts"},
    {"id":"BEH_001","abstract_id":"3111334","title":"Anchoring to Moving Averages and Return Predictability","author":"Avramov, Kaplanski, Subrahmanyam (2018)","category":"behavioural","priority":1,"relevance":"Anchoring bias creates predictable patterns"},
    {"id":"BEH_002","abstract_id":"2997001","title":"Parsimonious Model of Momentum and Reversals","author":"Luo, Subrahmanyam, Titman (2019)","category":"behavioural","priority":2,"relevance":"Overconfidence model — explains DRY_20"},
    {"id":"BEH_003","abstract_id":"3240609","title":"Momentum Enhancement via Investor Characteristics","author":"Multiple authors (2018)","category":"behavioural","priority":2,"relevance":"Age/BM/MAX enhance momentum"},
    {"id":"LDP_001","abstract_id":"3190768","title":"Revisiting the Deflated Sharpe Ratio","author":"Bailey, Lopez de Prado (2018)","category":"statistical","priority":1,"relevance":"DSR formula — must implement"},
    {"id":"LDP_002","abstract_id":"3616950","title":"The False Strategy Theorem","author":"Lopez de Prado (2019)","category":"statistical","priority":1,"relevance":"Multiple testing penalty"},
    {"id":"LDP_003","abstract_id":"3338682","title":"A Data Science Solution to the Multiple-Testing Crisis","author":"Lopez de Prado, Lewis (2018)","category":"statistical","priority":1,"relevance":"Practical solution to false discovery"},
    {"id":"LDP_004","abstract_id":"2701337","title":"The Probability of Backtest Overfitting","author":"Bailey, Borwein, Lopez de Prado, Zhu (2014)","category":"statistical","priority":1,"relevance":"CSCV method to detect overfitting"},
    {"id":"LDP_005","abstract_id":"2276632","title":"The Sharpe Ratio Efficient Frontier","author":"Bailey, Lopez de Prado (2012)","category":"statistical","priority":2,"relevance":"Sharpe statistics for finite samples"},
]


class SSRNDownloader:
    DOWNLOAD_DIR = Path("data/papers/ssrn")
    STATE_FILE = Path("data/papers/ssrn/download_state.json")
    LOG_FILE = Path("data/papers/ssrn/download_log.txt")
    URL_PATTERNS = [
        "https://papers.ssrn.com/sol3/Delivery.cfm/{aid}.pdf?abstractid={aid}&mirid=1",
        "https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID{aid}.pdf?abstractid={aid}&mirid=1",
        "https://ssrn.com/abstract={aid}",
    ]
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "application/pdf,text/html,application/xhtml+xml,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }

    COOKIE_FILE = Path("data/papers/ssrn/.cookies.json")
    LOGIN_URL = "https://www.ssrn.com/login"

    def __init__(self, delay=4.0, timeout=30):
        self.delay = delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self.DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
        self._load_cookies()

    def login(self, email, password):
        """Login to SSRN and save session cookies."""
        self._log("Logging into SSRN...")
        try:
            # Get login page first for CSRF token
            login_page = self.session.get("https://www.ssrn.com/login", timeout=15)

            # Post credentials
            resp = self.session.post(
                "https://www.ssrn.com/login",
                data={"username": email, "password": password},
                allow_redirects=True,
                timeout=15,
            )

            # Also try the API endpoint
            resp2 = self.session.post(
                "https://www.ssrn.com/rest/auth/login",
                json={"username": email, "password": password},
                timeout=15,
            )

            # Check if login succeeded by accessing a protected page
            check = self.session.get(
                "https://www.ssrn.com/index.cfm/en/",
                timeout=15,
            )

            if resp2.status_code == 200 or "logout" in check.text.lower():
                self._log("  Login successful")
                self._save_cookies()
                return True
            else:
                self._log(f"  Login may have failed (status {resp2.status_code})")
                # Save cookies anyway — some may have been set
                self._save_cookies()
                return True  # Try downloading anyway
        except Exception as e:
            self._log(f"  Login error: {e}")
            return False

    def _save_cookies(self):
        """Save session cookies to disk."""
        cookies = {}
        for cookie in self.session.cookies:
            cookies[cookie.name] = {
                "value": cookie.value,
                "domain": cookie.domain,
                "path": cookie.path,
            }
        self.COOKIE_FILE.write_text(json.dumps(cookies, indent=2))
        self._log(f"  Saved {len(cookies)} cookies")

    def _load_cookies(self):
        """Load saved cookies from disk."""
        if self.COOKIE_FILE.exists():
            try:
                cookies = json.loads(self.COOKIE_FILE.read_text())
                for name, data in cookies.items():
                    self.session.cookies.set(
                        name, data["value"],
                        domain=data.get("domain", ".ssrn.com"),
                        path=data.get("path", "/"),
                    )
            except Exception:
                pass

    def _load_state(self):
        if self.STATE_FILE.exists():
            return json.loads(self.STATE_FILE.read_text())
        return {"downloaded": {}, "failed": {}, "skipped": []}

    def _save_state(self):
        self.STATE_FILE.write_text(json.dumps(self.state, indent=2))

    def _log(self, msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(self.LOG_FILE, "a") as f:
            f.write(line + "\n")

    def _filename(self, paper):
        safe = paper["author"][:25].replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
        return self.DOWNLOAD_DIR / f"{paper['id']}_{safe}.pdf"

    def _extract_pdf_url(self, html, aid):
        import re
        patterns = [
            r'href="(https://papers\.ssrn\.com/sol3/Delivery\.cfm/[^"]*\.pdf[^"]*)"',
            r'href="(/sol3/Delivery\.cfm/[^"]*\.pdf[^"]*)"',
        ]
        for pat in patterns:
            match = re.search(pat, html, re.IGNORECASE)
            if match:
                url = match.group(1)
                if url.startswith("/"):
                    url = "https://papers.ssrn.com" + url
                return url
        return None

    def download_paper(self, paper):
        aid = paper["abstract_id"]
        filepath = self._filename(paper)
        if filepath.exists() and filepath.stat().st_size > 10_000:
            self._log(f"  OK EXISTS {paper['id']} — {filepath.name}")
            return {"status": "exists", "path": str(filepath)}

        for url_template in self.URL_PATTERNS:
            url = url_template.format(aid=aid)
            try:
                resp = self.session.get(url, timeout=self.timeout, allow_redirects=True)
                if resp.status_code == 200:
                    ct = resp.headers.get("content-type", "").lower()
                    if "pdf" in ct or resp.content[:4] == b"%PDF":
                        filepath.write_bytes(resp.content)
                        self._log(f"  OK DOWNLOAD {paper['id']} — {len(resp.content)//1024}KB")
                        return {"status": "downloaded", "path": str(filepath)}
                    elif "html" in ct:
                        pdf_url = self._extract_pdf_url(resp.text, aid)
                        if pdf_url and pdf_url != url:
                            time.sleep(1)
                            pdf_resp = self.session.get(pdf_url, timeout=self.timeout)
                            if pdf_resp.status_code == 200 and len(pdf_resp.content) > 10_000:
                                if "pdf" in pdf_resp.headers.get("content-type","").lower() or pdf_resp.content[:4] == b"%PDF":
                                    filepath.write_bytes(pdf_resp.content)
                                    self._log(f"  OK HTML→PDF {paper['id']} — {len(pdf_resp.content)//1024}KB")
                                    return {"status": "downloaded", "path": str(filepath)}
                elif resp.status_code == 403:
                    self._log(f"  FAIL 403 {paper['id']} — login required")
                    break
            except requests.exceptions.Timeout:
                self._log(f"  FAIL TIMEOUT {paper['id']}")
                continue
            except requests.exceptions.RequestException as e:
                self._log(f"  FAIL ERROR {paper['id']} — {e}")
                continue
            time.sleep(0.5)

        self._log(f"  FAIL {paper['id']} — all patterns exhausted")
        return {"status": "failed", "path": None}

    def download_all(self, papers=None, category=None, priority_max=3, retry_failed=False):
        if papers is None:
            papers = ALL_PAPERS
        if category:
            papers = [p for p in papers if p["category"] == category]
        papers = [p for p in papers if p["priority"] <= priority_max]
        if not retry_failed:
            papers = [p for p in papers if p["abstract_id"] not in self.state["failed"]]

        total = len(papers)
        results = {"downloaded": [], "exists": [], "failed": []}
        self._log(f"\nSSRN DOWNLOADER — {total} papers")

        for i, paper in enumerate(papers, 1):
            self._log(f"[{i:02d}/{total:02d}] {paper['id']} | P{paper['priority']} | {paper['title'][:50]}")
            result = self.download_paper(paper)
            if result["status"] in ("downloaded", "exists"):
                results["downloaded" if result["status"] == "downloaded" else "exists"].append(paper)
                self.state["downloaded"][paper["abstract_id"]] = {"path": result["path"], "timestamp": datetime.now().isoformat()}
            else:
                results["failed"].append(paper)
                self.state["failed"][paper["abstract_id"]] = {"timestamp": datetime.now().isoformat()}
            self._save_state()
            if i < total:
                time.sleep(self.delay)

        self._log(f"\nDONE: {len(results['downloaded'])} downloaded, {len(results['exists'])} existed, {len(results['failed'])} failed")
        if results["failed"]:
            self._log("FAILED (may need SSRN login):")
            for p in results["failed"]:
                self._log(f"  {p['id']} — https://papers.ssrn.com/sol3/papers.cfm?abstract_id={p['abstract_id']}")
        return results

    def list_papers(self):
        categories = {}
        for p in ALL_PAPERS:
            categories.setdefault(p["category"], []).append(p)
        cat_labels = {"india":"INDIA-SPECIFIC","momentum":"CORE MOMENTUM","volatility":"VOLATILITY MANAGEMENT",
                      "risk":"POSITION SIZING & RISK","mean_reversion":"MEAN REVERSION","options":"OPTIONS & VOL SURFACE",
                      "regime":"REGIME DETECTION","behavioural":"BEHAVIOURAL FINANCE","statistical":"STATISTICAL RIGOUR"}
        print(f"\nSSRN CATALOGUE — {len(ALL_PAPERS)} papers")
        for cat, papers in categories.items():
            print(f"\n── {cat_labels.get(cat, cat)} ({len(papers)}) ──")
            for p in sorted(papers, key=lambda x: x["priority"]):
                ok = "Y" if p["abstract_id"] in self.state.get("downloaded", {}) else " "
                print(f"  [{ok}] P{p['priority']} {p['id']:10s} {p['author'][:30]:30s} {p['abstract_id']}")


def main():
    parser = argparse.ArgumentParser(description="Download SSRN papers")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--category", choices=["india","momentum","volatility","risk","mean_reversion","options","regime","behavioural","statistical"])
    parser.add_argument("--priority", type=int, default=3)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--failed", action="store_true")
    parser.add_argument("--delay", type=float, default=4.0)
    parser.add_argument("--login", action="store_true", help="Login to SSRN first")
    parser.add_argument("--email", type=str, help="SSRN email")
    parser.add_argument("--password", type=str, help="SSRN password")
    args = parser.parse_args()

    downloader = SSRNDownloader(delay=args.delay)

    if args.login or (args.email and args.password):
        email = args.email or input("SSRN email: ")
        password = args.password or input("SSRN password: ")
        downloader.login(email, password)

    if args.list:
        downloader.list_papers()
        return
    if args.test:
        paper = next(p for p in ALL_PAPERS if p["category"] == "india" and p["priority"] == 1)
        print(f"TEST: {paper['title']}")
        result = downloader.download_paper(paper)
        print(f"Result: {result['status']}")
        return
    downloader.download_all(category=args.category, priority_max=args.priority, retry_failed=args.failed)


if __name__ == "__main__":
    main()
