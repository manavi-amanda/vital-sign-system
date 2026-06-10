import cv2
import time
import tempfile
import os

from modules.age_detector.age_detector import detect_age
from modules.respiratory.video_processor import process_video
from modules.heart_rate.hr_estimator import estimate_heart_rate
from modules.blood_pressure.predictor import predict_blood_pressure

RTSP_URL = "rtsp://username:password@camera-ip:554/stream"

WINDOW_SECONDS = 30


def start_rtsp_processing():

    print(f"Connecting to RTSP: {RTSP_URL}")

    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print("Failed to connect to RTSP stream")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Connected. FPS={fps}, Resolution={width}x{height}")

    while True:

        try:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as temp_file:

                temp_video_path = temp_file.name

            writer = cv2.VideoWriter(
                temp_video_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )

            print(f"\nRecording {WINDOW_SECONDS}s clip...")

            start_time = time.time()

            while time.time() - start_time < WINDOW_SECONDS:

                ret, frame = cap.read()

                if not ret:
                    print("Frame read failed")
                    continue

                writer.write(frame)

            writer.release()

            print("Processing clip...")

            # -----------------------------
            # Age Detection
            # -----------------------------
            age = detect_age(temp_video_path)

            # -----------------------------
            # RR Analysis
            # -----------------------------
            with open(temp_video_path, "rb") as f:
                video_bytes = f.read()

            rr_result = process_video(video_bytes)

            rr = rr_result["final_bpm"]

            body_temp = float(
                rr_result["body_temperature"]["temp_c_estimate"]
            )

            # -----------------------------
            # HR
            # -----------------------------
            hr_result = estimate_heart_rate(rr)

            hr = hr_result["hr_estimated"]

            # -----------------------------
            # BP
            # -----------------------------
            bp_result = predict_blood_pressure(
                body_temp=body_temp,
                heart_rate=hr
            )

            # -----------------------------
            # Console Output
            # -----------------------------
            print("\n========================")
            print("Vital Analysis Result")
            print("========================")
            print(f"Age              : {age}")
            print(f"Respiratory Rate : {rr}")
            print(f"Heart Rate       : {hr}")
            print(f"Body Temp        : {body_temp}")
            print(f"Blood Pressure   : {bp_result['bp_category']}")
            print("========================\n")

        except Exception as e:
            print(f"Processing Error: {e}")

        finally:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)