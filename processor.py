import json
from datetime import datetime

from modules.respiratory.video_processor import process_video
from modules.heart_rate.hr_estimatorV2 import estimate_heart_rate
from modules.blood_pressure.predictor import predict_blood_pressure
from modules.heart_rate.vitals_api import process_vitallens
from modules.heart_rate.hr_estimatorV2 import estimate_hybrid_heart_rate
from modules.age_detector.age_detector import detect_age

import state

LOG_FILE = "output_logs.jsonl"


def log_run(data: dict):
    data["timestamp"] = datetime.now().isoformat()

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")


def run_pipeline(video_path):
    """
    Process a recorded video and update the latest results.
    Returns the final output dictionary.
    """

    print("\n========== PIPELINE START ==========")

    with open(video_path, "rb") as f:
        file_bytes = f.read()

    # Step 1 - Respiratory + Temperature
    result = process_video(file_bytes)

    print("\n[RESPIRATORY RESULT]")
    print(result)

    if not result or "final_bpm" not in result:
        raise Exception("RR analysis failed")

    # Step 2 - Heart Rate
    hr_result = estimate_heart_rate(result["final_bpm"])

    print("\n[HEART RATE RESULT]")
    print(hr_result)

    # Step 3 - Blood Pressure
    bp_result = predict_blood_pressure(
        body_temp=float(result["body_temperature"]["temp_c_estimate"]),
        heart_rate=float(hr_result["hr_estimated"]),
        age=None
    )

    print("\n[BLOOD PRESSURE RESULT]")
    print(bp_result)

    final_output = {
        "bpm": result["final_bpm"],
        "body_temperature": result["body_temperature"]["temp_c_estimate"],
        "heart_rate": hr_result["hr_estimated"],
        "blood_pressure": bp_result["bp_category"]
    }

    print("\n========== FINAL OUTPUT ==========")
    print(final_output)
    print("=================================\n")

    # Save for FastAPI
    state.latest_result.clear()
    state.latest_result.update(final_output)

    # Log to file
    log_run(final_output.copy())

    return final_output

def run_rgb_pipeline(video_path, api_key, hybrid_x=0.0):

    print("\n========== RGB PIPELINE START ==========")

    if not (0.0 <= hybrid_x <= 1.0):
        raise Exception("hybrid_x must be between 0 and 1")

    # Read video
    with open(video_path, "rb") as f:
        file_bytes = f.read()

    suffix = "." + video_path.split(".")[-1]

    # Step 1 - Age Detection
    detected_age = 24

    # print("\n[AGE RESULT]")
    # print({"age": detected_age})

    # Step 2 - VitalLens Analysis
    vl_result = process_vitallens(
        file_bytes,
        api_key=api_key,
        suffix=suffix
    )

    print("\n[VITALLENS RESULT]")
    print(vl_result)

    hr_rgb = vl_result.get("heart_rate")
    rr_bpm = vl_result.get("respiratory_rate")

    if hr_rgb is None or rr_bpm is None:
        raise Exception(
            "VitalLens could not detect HR or RR. Check lighting and face visibility."
        )

    # Step 3 - Hybrid HR
    hr_result = estimate_hybrid_heart_rate(
        hr_rgb=hr_rgb,
        rr_bpm=rr_bpm,
        x=hybrid_x
    )

    print("\n[HYBRID HEART RATE RESULT]")
    print(hr_result)

    # Step 4 - Blood Pressure
    bp_result = predict_blood_pressure(
        body_temp=37.0,
        heart_rate=hr_result["hr_estimated"],
        age=detected_age
    )

    # print("\n[BLOOD PRESSURE RESULT]")
    # print(bp_result)

    final_output = {
        "heart_rate": hr_result["hr_estimated"],
        "bpm": round(rr_bpm, 2),
        "body_temperature": "Not found",
        # "hrv_sdnn_ms": vl_result.get("hrv_sdnn"),
        # "age": detected_age,

        # "hybrid_detail": {
        #     "hr_rgb": hr_result["hr_rgb"],
        #     "hr_resp": hr_result["hr_resp"],
        #     "hybrid_x": hr_result["hybrid_x"],
        #     "source": hr_result["source"]
        # },
        #
        # "confidence": {
        #     "hr": vl_result.get("hr_confidence"),
        #     "rr": vl_result.get("rr_confidence")
        # },

        "blood_pressure": bp_result["bp_category"]
    }

    print("\n========== RGB FINAL OUTPUT ==========")
    print(final_output)
    print("======================================\n")

    state.latest_result.clear()
    state.latest_result.update(final_output)

    log_run(final_output.copy())

    return final_output