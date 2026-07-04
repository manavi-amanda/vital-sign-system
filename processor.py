import json
from datetime import datetime

from modules.respiratory.video_processor import process_video
from modules.heart_rate.hr_estimatorV2 import estimate_heart_rate
from modules.blood_pressure.predictor import predict_blood_pressure

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