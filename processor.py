import json
from datetime import datetime
import cv2
import os

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

def determine_human_status(heart_rate, respiration_rate, body_temperature):
    """
    Determine human status based on estimated vital signs.

    NOTE:
    This function DOES NOT determine death.
    It only reports whether vital signs are detected.
    """

    reasons = []
    confidence = 0.0

    if heart_rate is not None:
        try:
            heart_rate = float(heart_rate)

            if heart_rate >= 20:
                confidence += 0.6
                reasons.append(f"Heart rate detected ({heart_rate:.1f} bpm)")
            else:
                reasons.append("Heart rate extremely low or not detected")

        except:
            heart_rate = None

    else:
        reasons.append("Heart rate unavailable")

    if respiration_rate is not None:
        try:
            respiration_rate = float(respiration_rate)

            if respiration_rate >= 4:
                confidence += 0.3
                reasons.append(f"Respiration detected ({respiration_rate:.1f} bpm)")
            else:
                reasons.append("Respiration extremely low or not detected")

        except:
            respiration_rate = None

    else:
        reasons.append("Respiration unavailable")

    if body_temperature is not None:
        try:
            body_temperature = float(body_temperature)

            if body_temperature >= 30:
                confidence += 0.1
                reasons.append(f"Body temperature {body_temperature:.1f}°C")

        except:
            body_temperature = None

    if confidence >= 0.7:
        status = "Alive"

    elif confidence >= 0.3:
        status = "Likely Alive"

    elif confidence > 0:
        status = "No Detectable Vital Signs"

    else:
        status = "Unable to Determine"

    return {
        "status": status,
        "confidence": round(confidence, 2),
        "reason": reasons
    }




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

    human_status = determine_human_status(
        heart_rate=hr_result["hr_estimated"],
        respiration_rate=result["final_bpm"],
        body_temperature=result["body_temperature"]["temp_c_estimate"]
    )

    print("\n[BLOOD PRESSURE RESULT]")
    print(bp_result)

    final_output = {
        "bpm": result["final_bpm"],
        "body_temperature": result["body_temperature"]["temp_c_estimate"],
        "heart_rate": hr_result["hr_estimated"],
        "blood_pressure": bp_result["bp_category"],
        "human_status": human_status
    }

    print("\n========== FINAL OUTPUT ==========")
    print(final_output)
    print("=================================\n")

    # Save for FastAPI
    if state.mode == 1:

        state.latest_result.clear()
        state.latest_result.update(final_output)

    else:

        state.thermal_result = final_output.copy()

        state.latest_result.clear()

        state.latest_result.update({

            "thermal_completed": True

        })

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

    detected_age = 24  # Default fallback age
    temp_frame_path = "temp_age_frame.jpg"

    print("\n[EXTRACTING FRAME FOR AGE DETECTION]")
    try:
        # Open the video and read the first frame
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if ret:
            # Save the single frame temporarily
            cv2.imwrite(temp_frame_path, frame)

            # Call your age detector using the single image frame
            # (Assuming detect_age accepts a file path. Adjust if it expects bytes)
            detected_age = detect_age(temp_frame_path)

            print(f"Age successfully detected: {detected_age}")
        else:
            print("Warning: Could not extract frame from video. Using default age.")

    except Exception as e:
        print(f"Age detection failed (API limit reached?): {e}. Using default age {detected_age}.")

    finally:
        # Always clean up the temporary image file
        if os.path.exists(temp_frame_path):
            os.remove(temp_frame_path)

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

    thermal_temp = 37.0
    thermal_resp = rr_bpm

    if state.mode == 2 and state.thermal_result is not None:
        thermal_temp = float(state.thermal_result["body_temperature"])
        thermal_resp = float(state.thermal_result["bpm"])

    # Step 4 - Blood Pressure
    bp_result = predict_blood_pressure(
        body_temp=thermal_temp,
        heart_rate=hr_result["hr_estimated"],
        age=detected_age
    )

    human_status = determine_human_status(
        heart_rate=hr_result["hr_estimated"],
        respiration_rate=thermal_resp,
        body_temperature=thermal_temp
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

        "blood_pressure": "Not found",
        "human_status": human_status
    }

    print("\n========== RGB FINAL OUTPUT ==========")
    print(final_output)
    print("======================================\n")

    if state.mode == 1:

        state.latest_result.clear()
        state.latest_result.update(final_output)

    else:

        combined_output = {
            "body_temperature": thermal_temp,
            "bpm": thermal_resp,
            "heart_rate": hr_result["hr_estimated"],
            "blood_pressure": bp_result["bp_category"],
            "human_status": human_status
        }

        state.latest_result.clear()

        state.latest_result.update(combined_output)


    log_run(final_output.copy())

    return final_output