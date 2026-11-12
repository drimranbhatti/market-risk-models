"""Gaussian hidden Markov model implemented from scratch.

* scaled forward-backward recursions (Rabiner, 1989) for numerical stability
* Viterbi decoding in log space
* Baum-Welch (EM) estimation with multiple random restarts
States are sorted by mean so that state 0 is the most bearish.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class GaussianHMM:
    pi: np.ndarray      # (K,) initial distribution
    A: np.ndarray       # (K, K) transitions, rows sum to 1
    mu: np.ndarray      # (K,)
    sigma: np.ndarray   # (K,)

    @property
    def K(self):
        return len(self.pi)

    def emission(self, x):
        return stats.norm.pdf(np.asarray(x)[:, None], self.mu[None, :], self.sigma[None, :])

    # ---- inference -------------------------------------------------
    def forward_backward(self, x):
        """Returns (alpha_hat, beta_hat, scale c, log-likelihood). Posterior = alpha_hat * beta_hat."""
        B = self.emission(x)
        T, K = B.shape
        alpha, beta, c = np.zeros((T, K)), np.zeros((T, K)), np.zeros(T)
        alpha[0] = self.pi * B[0]; c[0] = alpha[0].sum(); alpha[0] /= c[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * B[t]
            c[t] = alpha[t].sum(); alpha[t] /= c[t]
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        return alpha, beta, c, float(np.log(c).sum())

    def log_likelihood(self, x):
        return self.forward_backward(x)[3]

    def posterior(self, x):
        a, b, _, _ = self.forward_backward(x)
        g = a * b
        return g / g.sum(axis=1, keepdims=True)

    def viterbi(self, x):
        logB = np.log(self.emission(x) + 1e-300)
        logA = np.log(self.A + 1e-300)
        T, K = logB.shape
        delta = np.zeros((T, K)); psi = np.zeros((T, K), dtype=int)
        delta[0] = np.log(self.pi + 1e-300) + logB[0]
        for t in range(1, T):
            cand = delta[t - 1][:, None] + logA
            psi[t] = cand.argmax(axis=0)
            delta[t] = cand.max(axis=0) + logB[t]
        path = np.zeros(T, dtype=int); path[-1] = delta[-1].argmax()
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def stationary(self):
        w, v = np.linalg.eig(self.A.T)
        s = np.real(v[:, np.argmin(np.abs(w - 1))])
        return s / s.sum()

    def expected_duration(self):
        return 1 / (1 - np.diag(self.A))

    def sample(self, T, seed=0):
        rng = np.random.default_rng(seed)
        s = np.zeros(T, dtype=int); s[0] = rng.choice(self.K, p=self.pi)
        for t in range(1, T):
            s[t] = rng.choice(self.K, p=self.A[s[t - 1]])
        return rng.normal(self.mu[s], self.sigma[s]), s


def _sorted(m: GaussianHMM) -> GaussianHMM:
    o = np.argsort(m.mu)
    return GaussianHMM(m.pi[o], m.A[np.ix_(o, o)], m.mu[o], m.sigma[o])


def baum_welch(x, K=2, n_iter=300, tol=1e-6, n_restarts=8, seed=0, min_sigma=1e-4) -> tuple[GaussianHMM, list]:
    """EM estimation. Returns (best model, log-likelihood history of the best restart)."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    best, best_ll, best_hist = None, -np.inf, []
    for _ in range(n_restarts):
        qs = np.sort(rng.choice(x, K, replace=False))
        m = GaussianHMM(np.full(K, 1 / K), np.full((K, K), 0.1 / max(K - 1, 1)) + np.eye(K) * (0.9 - 0.1 / max(K - 1, 1)),
                        qs, np.full(K, x.std()) * rng.uniform(0.5, 1.5, K))
        hist = []
        for _ in range(n_iter):
            B = m.emission(x)
            a, b, c, ll = m.forward_backward(x)
            gamma = a * b; gamma /= gamma.sum(axis=1, keepdims=True)
            xi = (a[:-1, :, None] * m.A[None] * (B[1:] * b[1:])[:, None, :]) / c[1:, None, None]
            m = GaussianHMM(gamma[0], xi.sum(0) / xi.sum(0).sum(1, keepdims=True),
                            (gamma * x[:, None]).sum(0) / gamma.sum(0), None)
            m.sigma = np.maximum(np.sqrt((gamma * (x[:, None] - m.mu) ** 2).sum(0) / gamma.sum(0)), min_sigma)
            hist.append(ll)
            if len(hist) > 1 and abs(hist[-1] - hist[-2]) < tol:
                break
        ll = m.log_likelihood(x)
        if ll > best_ll:
            best, best_ll, best_hist = m, ll, hist
    return _sorted(best), best_hist


def bic(model: GaussianHMM, x) -> float:
    K = model.K
    n_params = (K - 1) + K * (K - 1) + 2 * K
    return -2 * model.log_likelihood(x) + n_params * np.log(len(x))
