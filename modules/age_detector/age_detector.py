import cv2
import numpy as np
from deepface import DeepFace


def detect_age(video_path):

    cap = cv2.VideoCapture(video_path)

    ages = []
    frame_count = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Every 15th frame
        if frame_count % 15 != 0:
            continue

        # Blur check
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if cv2.Laplacian(gray, cv2.CV_64F).var() < 100:
            continue

        try:

            result = DeepFace.analyze(
                frame,
                actions=["age"],
                detector_backend="retinaface",
                enforce_detection=True,
                silent=True
            )

            age = result[0]["age"] if isinstance(result, list) else result["age"]

            print(age)
            ages.append(float(age))

            # Enough good samples
            if len(ages) >= 20:
                break

        except Exception:
            continue

    cap.release()

    if not ages:
        return None

    ages = np.array(ages)

    q1 = np.percentile(ages, 25)
    q3 = np.percentile(ages, 75)

    iqr = q3 - q1

    ages = ages[
        (ages >= q1 - 1.5 * iqr) &
        (ages <= q3 + 1.5 * iqr)
    ]

    return max(0, int(np.percentile(ages, 10)) - 2)