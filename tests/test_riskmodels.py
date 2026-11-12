import numpy as np
import pandas as pd
import pytest

from riskmodels import backtest, hmm, var

rng = np.random.default_rng(0)
r = pd.Series(rng.normal(0, 0.01, 3000), index=pd.bdate_range("2010-01-01", periods=3000))


def test_parametric_var_matches_theory_and_no_lookahead():
    v = var.parametric_normal(r, 250, 0.99)
    assert np.isnan(v["VaR"].iloc[249]) and not np.isnan(v["VaR"].iloc[250])   # uses only the 250 prior days
    assert v["VaR"].dropna().mean() == pytest.approx(0.01 * 2.326, rel=0.05)
    assert (v["ES"].dropna() > v["VaR"].dropna()).all()


def test_backtests_on_correct_model():
    h = backtest.exceptions(r, var.parametric_normal(r)["VaR"])
    k = backtest.kupiec_pof(h)
    assert k["p_pof"] > 0.01
    assert backtest.basel_zone(3) == "green" and backtest.basel_zone(7) == "yellow" and backtest.basel_zone(12) == "red"


def test_kupiec_rejects_bad_model():
    bad = pd.Series(0.005, index=r.index)                  # VaR far too small
    assert backtest.kupiec_pof(backtest.exceptions(r, bad))["p_pof"] < 1e-6


def test_hmm_forward_matches_brute_force():
    m = hmm.GaussianHMM(np.array([0.6, 0.4]), np.array([[0.9, 0.1], [0.2, 0.8]]), np.array([-1.0, 1.0]), np.array([1.0, 0.5]))
    x = np.array([0.3, -1.2, 0.8])
    import itertools
    from scipy import stats
    brute = 0.0
    for s in itertools.product([0, 1], repeat=3):
        p = m.pi[s[0]] * stats.norm.pdf(x[0], m.mu[s[0]], m.sigma[s[0]])
        for t in (1, 2):
            p *= m.A[s[t - 1], s[t]] * stats.norm.pdf(x[t], m.mu[s[t]], m.sigma[s[t]])
        brute += p
    assert m.log_likelihood(x) == pytest.approx(np.log(brute))
    assert np.allclose(m.posterior(x).sum(1), 1)


def test_baum_welch_recovers_parameters_and_monotone_ll():
    true = hmm.GaussianHMM(np.array([0.5, 0.5]), np.array([[0.97, 0.03], [0.05, 0.95]]), np.array([-0.01, 0.005]), np.array([0.02, 0.007]))
    x, s = true.sample(4000, seed=1)
    est, hist = hmm.baum_welch(x, K=2, seed=0)
    assert np.all(np.diff(hist) > -1e-6)                    # EM never decreases the likelihood
    assert est.mu == pytest.approx(true.mu, abs=0.003)
    assert est.sigma == pytest.approx(true.sigma, rel=0.1)
    assert (est.viterbi(x) == s).mean() > 0.9


def test_tercile_bins_use_training_data_only():
    from riskmodels import bayesnet
    s = pd.Series(np.r_[np.arange(30.0), np.full(30, 100.0)], index=pd.date_range("2000-01-31", periods=60, freq="ME"))
    st = bayesnet.tercile_states(s, fit_end=s.index[29])
    assert (st.iloc[30:] == "high").all()                 # later values never move the cut-offs
    assert st.iloc[:30].value_counts().min() >= 9
