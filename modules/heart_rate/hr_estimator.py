import numpy as np


def estimate_heart_rate(rr_bpm: float):
    # """
    # Estimate Heart Rate (HR) from Respiratory Rate (RR)
    # using clinical HR:RR ratio approximation.

    # Clinical rule:
    #     HR / RR ≈ 4 to 5 (healthy adult at rest)
    #     more precise range: 4.5 – 5.5

    # Therefore:
    #     HR ≈ RR × 4  (lower bound)
    #     HR ≈ RR × 5  (upper bound)
    #     HR ≈ RR × 4.5–5.5 (typical physiological range)

    # Returns:
    #     dict containing HR estimate range + midpoint + ratio
    # """

    if rr_bpm is None or rr_bpm <= 0:
        return {
            "error": "Invalid RR value",
            "hr_min": None,
            "hr_max": None,
            "hr_estimated": None,
            "ratio_range": None
        }

    # ---- Clinical multipliers ----
    low_ratio = 4.0
    high_ratio = 5.0
    mid_ratio = 4.5

    hr_min = rr_bpm * low_ratio
    hr_max = rr_bpm * high_ratio
    hr_mid = rr_bpm * mid_ratio

    # Optional: more stable estimate (median-like)
    hr_estimated = float(np.mean([hr_min, hr_mid, hr_max]))

    ratio_range = (4.0, 5.0)
    print(f"[INFO] Heart rate successfully estimated. Results: {hr_estimated:.2f} ")
    return {
        "rr_bpm": rr_bpm,
        "hr_min": round(hr_min, 2),
        "hr_max": round(hr_max, 2),
        "hr_estimated": round(hr_estimated, 2),
        "ratio_range": ratio_range
    }