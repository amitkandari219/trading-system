"""
Bayesian Online Changepoint Detection (Adams & MacKay 2007).

Computes P(changepoint at time t) for each day.
Separates temporary regime shifts from structural decay.
Pure NumPy implementation — no external library needed.
"""

import numpy as np
from scipy.stats import norm
from typing import List, Tuple


class BOCDDetector:
    """
    Bayesian Online Changepoint Detection.

    Maintains a run-length distribution: P(r_t | x_{1:t})
    where r_t is the run length (time since last changepoint).
    """

    def __init__(self, hazard_rate=1/100, mu_prior=0.0, sigma_prior=1.0):
        """
        Args:
            hazard_rate: prior probability of changepoint at each step (1/expected_run_length)
            mu_prior: prior mean of normal observation model
            sigma_prior: prior std of observation model
        """
        self.hazard = hazard_rate
        self.mu0 = mu_prior
        self.sigma0 = sigma_prior

        # Sufficient statistics for Bayesian updating
        self.run_length_probs = np.array([1.0])  # P(r_t = 0) = 1 at start
        self.mu_params = np.array([mu_prior])
        self.sigma_params = np.array([sigma_prior])
        self.n_obs = np.array([0.0])
        self.sum_x = np.array([0.0])
        self.sum_x2 = np.array([0.0])

        self.changepoint_prob_history = []
        self.t = 0

    def update(self, x) -> float:
        """
        Process one observation, return P(changepoint today).

        Args:
            x: today's observation (P&L)

        Returns:
            changepoint_probability: P(changepoint at time t)
        """
        self.t += 1

        # Predictive probability of x under each run length
        pred_probs = self._predictive_prob(x)

        # Growth probabilities (no changepoint)
        growth = self.run_length_probs * pred_probs * (1 - self.hazard)

        # Changepoint probability (sum of all run lengths collapsing to 0)
        cp_prob = np.sum(self.run_length_probs * pred_probs * self.hazard)

        # New run length distribution
        new_probs = np.zeros(len(growth) + 1)
        new_probs[0] = cp_prob
        new_probs[1:] = growth

        # Normalize
        total = new_probs.sum()
        if total > 0:
            new_probs /= total

        self.run_length_probs = new_probs

        # Update sufficient statistics
        new_n = np.zeros(len(self.n_obs) + 1)
        new_sum = np.zeros(len(self.sum_x) + 1)
        new_sum2 = np.zeros(len(self.sum_x2) + 1)

        new_n[0] = 0
        new_n[1:] = self.n_obs + 1
        new_sum[0] = 0
        new_sum[1:] = self.sum_x + x
        new_sum2[0] = 0
        new_sum2[1:] = self.sum_x2 + x**2

        self.n_obs = new_n
        self.sum_x = new_sum
        self.sum_x2 = new_sum2

        # Trim very long run lengths (memory management)
        max_len = 500
        if len(self.run_length_probs) > max_len:
            self.run_length_probs = self.run_length_probs[:max_len]
            self.n_obs = self.n_obs[:max_len]
            self.sum_x = self.sum_x[:max_len]
            self.sum_x2 = self.sum_x2[:max_len]
            # Renormalize
            total = self.run_length_probs.sum()
            if total > 0:
                self.run_length_probs /= total

        self.changepoint_prob_history.append(cp_prob)
        return cp_prob

    def _predictive_prob(self, x):
        """Compute predictive probability under each run length."""
        probs = np.zeros(len(self.n_obs))

        for i in range(len(self.n_obs)):
            n = self.n_obs[i]
            if n < 2:
                # Use prior
                probs[i] = norm.pdf(x, self.mu0, self.sigma0)
            else:
                # Posterior predictive (normal-normal conjugate)
                mean = self.sum_x[i] / n
                var = (self.sum_x2[i] / n - mean**2) * (n + 1) / n
                std = max(np.sqrt(abs(var)), 1e-6)
                probs[i] = norm.pdf(x, mean, std)

        return np.maximum(probs, 1e-300)  # prevent underflow

    def get_most_likely_run_length(self) -> int:
        """Return most probable current run length."""
        if len(self.run_length_probs) == 0:
            return 0
        return int(np.argmax(self.run_length_probs))

    def get_changepoint_probability(self) -> float:
        """Return probability that a changepoint occurred at the most recent step."""
        if not self.changepoint_prob_history:
            return 0.0
        return self.changepoint_prob_history[-1]

    def get_recent_max_cp_prob(self, window=10) -> float:
        """Max changepoint probability in recent window."""
        if not self.changepoint_prob_history:
            return 0.0
        recent = self.changepoint_prob_history[-window:]
        return max(recent)

    def reset(self):
        """Reset detector state."""
        self.__init__(self.hazard, self.mu0, self.sigma0)
