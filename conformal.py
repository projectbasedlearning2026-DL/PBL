"""conformal.py - split conformal calibration of the transformer's P10-P90 interval (CQR, Romano et al. 2019).

Fitted on the VALIDATION block only, applied unchanged to the test block (test labels are never used).
Scores are divided by a per-series scale so that large and small series are on a common footing.
Guarantee caveat: it is exact only for exchangeable data; with time series the validation block precedes the
test block, so coverage is approximate - that is why the achieved test coverage is always reported.
"""
import numpy as np


def cqr_offset(y_val, lo, hi, scale, alpha=0.2):
    """Return the normalised offset Q (>= or < 0): intervals become [lo - Q*scale, hi + Q*scale]."""
    s = (np.maximum(lo - y_val, y_val - hi) / scale[:, None]).ravel()
    n = s.size
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, level, method="higher"))


def apply_offset(lo, mid, hi, scale, Q):
    """Widen (Q>0) or shrink (Q<0) the interval; keep 0 <= lo <= mid <= hi."""
    lo2 = np.clip(lo - Q * scale[:, None], 0, None)
    hi2 = hi + Q * scale[:, None]
    return np.minimum(lo2, mid), mid, np.maximum(hi2, mid)
