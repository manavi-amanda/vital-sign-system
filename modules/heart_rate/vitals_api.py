"""
modules/heart_rate/vitals_api.py
"""
import os
import time

_HERE     = os.path.dirname(os.path.abspath(__file__))
_CLIP_DIR = os.path.join(_HERE, "..", "..", "tmp_clips")


def process_vitallens(file_bytes: bytes, api_key: str, suffix: str = ".mp4") -> dict:
    from vitallens import VitalLens

    clip_dir = os.path.normpath(os.path.abspath(_CLIP_DIR))
    os.makedirs(clip_dir, exist_ok=True)

    ext      = suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
    filename = f"vl_{int(time.time() * 1000)}{ext}"
    tmp_path = os.path.normpath(os.path.join(clip_dir, filename))

    try:
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)
            f.flush()
            os.fsync(f.fileno())

        time.sleep(0.1)

        vl      = VitalLens(method="vitallens", api_key=api_key)
        results = vl(tmp_path)

        if not results or len(results) == 0:
            raise RuntimeError("VitalLens returned no result — no face detected.")

        vitals_raw = results[0].get("vitals", {})
        hr  = vitals_raw.get("heart_rate",       {})
        rr  = vitals_raw.get("respiratory_rate", {})
        hrv = vitals_raw.get("hrv_sdnn",         {})

        return {
            "heart_rate":       hr.get("value"),
            "hr_confidence":    hr.get("confidence"),
            "respiratory_rate": rr.get("value"),
            "rr_confidence":    rr.get("confidence"),
            "hrv_sdnn":         hrv.get("value"),
            "raw":              results[0],
        }

    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass