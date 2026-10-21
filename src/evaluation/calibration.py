"""Post-hoc temperature scaling fitted on validation patients only."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax


def fit_temperature(logp: np.ndarray, y: np.ndarray, bounds=(0.05, 20.0)) -> float:
    """Scalar T minimising NLL of softmax(logp / T). `logp` may be logits or log-probabilities."""
    ok = y >= 0
    logp, y = logp[ok], y[ok]

    def nll(log_t):
        return -log_softmax(logp / np.exp(log_t), axis=1)[np.arange(len(y)), y].mean()

    r = minimize_scalar(nll, bounds=np.log(bounds), method="bounded")
    return float(np.exp(r.x))
