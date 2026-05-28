import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
import os

from .respiratoryRate import analyse_respiratory_rate
from modules.temperature.temp_estimator import TemperatureEstimator


# ---------------------------
# PATH SETUP
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

PERSON_MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n.pt")
FACE_MODEL_PATH   = os.path.join(MODEL_DIR, "yolov8n-face.pt")


# ---------------------------
# YOLO DETECTOR
# ---------------------------
class YOLODetector:
    def __init__(self, conf=0.35):
        self.person_model = YOLO(PERSON_MODEL_PATH)

        try:
            self.face_model = YOLO(FACE_MODEL_PATH)
            self.face_available = True
            print("[INFO] Face model loaded successfully")
        except Exception as e:
            self.face_model = None
            self.face_available = False
            print("[WARN] Face model failed to load:", str(e))

        self.conf = conf

    def detect_faces(self, frame):
        faces = []

        if self.face_available:
            results = self.face_model(frame, conf=self.conf, verbose=False)[0].boxes
            for b in results:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                faces.append((x1, y1, x2, y2))
        else:
            # fallback ROI
            h, w = frame.shape[:2]
            faces.append((w//3, h//4, w*2//3, h*3//4))

        return faces


# ---------------------------
# ROI INTENSITY (RR SIGNAL)
# ---------------------------
def extract_roi_intensity(frame, x1, y1, x2, y2):
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


# ---------------------------
# MAIN VIDEO PROCESSOR
# ---------------------------
def process_video(file_bytes, fps=25):

    # Save temp file
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp.write(file_bytes)
    temp.close()

    cap = cv2.VideoCapture(temp.name)

    detector = YOLODetector()
    temp_estimator = TemperatureEstimator()

    intensity_log = []

    print("[INFO] Starting video processing...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect_faces(frame)

        if len(faces) > 0:
            x1, y1, x2, y2 = faces[0]

            # ---------------------------
            # RR signal extraction
            # ---------------------------
            intensity = extract_roi_intensity(frame, x1, y1, x2, y2)

            if intensity is not None:
                intensity_log.append(intensity)

            # ---------------------------
            # TEMPERATURE estimation
            # ---------------------------
            temp_estimator.update(frame, (x1, y1, x2, y2))

    cap.release()

    # ---------------------------
    # VALIDATION
    # ---------------------------
    if len(intensity_log) < fps * 3:
        print("[ERROR] Video too short for RR analysis. Frames:", len(intensity_log))
        return {
            "error": "video too short for RR analysis",
            "final_bpm": None
        }

    # ---------------------------
    # RESPIRATORY RATE
    # ---------------------------
    rr_result = analyse_respiratory_rate(intensity_log, fps)

    if rr_result is None:
        return {
            "error": "RR analysis failed",
            "final_bpm": None
        }

    # ---------------------------
    # TEMPERATURE RESULT
    # ---------------------------
    temp_result = temp_estimator.get_results()

    print("[INFO] RR analysis successful:", rr_result["final_bpm"])
    print("[INFO] Temperature analysis successful:", temp_result["temp_c_estimate"])

    # ---------------------------
    # FINAL OUTPUT
    # ---------------------------
    return {
        **rr_result,
        "body_temperature": temp_result
    }