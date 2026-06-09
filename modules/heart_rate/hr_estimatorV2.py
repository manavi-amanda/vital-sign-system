"""
modules/heart_rate/hr_estimator.py
===================================
Hybrid Heart Rate estimator.

  HR_hybrid = (1 - x) * HR_rgb  +  x * HR_resp

  x = 0.0  →  pure rPPG  (VitalLens camera signal)
  x = 1.0  →  pure respiratory-derived estimate
  0 < x < 1 →  blended  (tune with hybrid_hr_tuner.py)
"""

import numpy as np

HR_RR_RATIO: float = 4.5          # clinical HR:RR multiplier (range 4–5)
DEFAULT_X:   float = 0.0          # default mixing weight — update after tuning


def estimate_heart_rate(rr_bpm: float) -> dict:
    """
    Respiratory-only HR estimate (kept for backward compatibility).
    Called when no rPPG signal is available.
    """
    if rr_bpm is None or rr_bpm <= 0:
        return {
            "error": "Invalid RR value",
            "hr_min": None, "hr_max": None, "hr_estimated": None,
        }

    hr_min = rr_bpm * 4.0
    hr_max = rr_bpm * 5.0
    hr_mid = rr_bpm * HR_RR_RATIO
    hr_estimated = float(np.mean([hr_min, hr_mid, hr_max]))

    return {
        "rr_bpm":       rr_bpm,
        "hr_min":       round(hr_min,       2),
        "hr_max":       round(hr_max,       2),
        "hr_estimated": round(hr_estimated, 2),
        "source":       "respiratory_only",
    }


def estimate_hybrid_heart_rate(
    hr_rgb:  float,
    rr_bpm:  float,
    x:       float = DEFAULT_X,
) -> dict:
    """
    Blend rPPG HR (from VitalLens) with respiratory-derived HR.

    Parameters
    ----------
    hr_rgb  : heart rate from VitalLens rPPG (bpm)
    rr_bpm  : respiratory rate from VitalLens (rpm)
    x       : mixing weight  0=pure rPPG  1=pure respiratory

    Returns
    -------
    dict with hr_estimated, hr_rgb, hr_resp, hybrid_x, source
    """
    if not (0.0 <= x <= 1.0):
        raise ValueError(f"x must be in [0.0, 1.0], got {x}")

    hr_resp   = rr_bpm * HR_RR_RATIO
    hr_hybrid = (1.0 - x) * hr_rgb + x * hr_resp

    return {
        "hr_estimated": round(hr_hybrid, 2),
        "hr_rgb":       round(hr_rgb,    2),
        "hr_resp":      round(hr_resp,   2),
        "hybrid_x":     x,
        "source":       "hybrid" if x > 0.0 else "rppg_only",
    }