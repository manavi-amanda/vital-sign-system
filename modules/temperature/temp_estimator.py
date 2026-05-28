import numpy as np
import cv2


class TemperatureEstimator:

    def __init__(self, smooth=True):
        self.temp_log = []
        self.smooth = smooth

    def extract_roi_temp(self, frame, x1, y1, x2, y2):
        # """
        # Extract temperature-like signal from face ROI
        # """

        roi = frame[y1:y2, x1:x2]

        if roi is None or roi.size == 0:
            return None

        # Convert to grayscale (brightness proxy)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Basic statistic: mean brightness
        temp_index = float(np.mean(gray))

        return temp_index

    def update(self, frame, face_box):
        # """
        # Call this for each frame.
        # face_box = (x1, y1, x2, y2)
        # """

        x1, y1, x2, y2 = face_box
        temp = self.extract_roi_temp(frame, x1, y1, x2, y2)

        if temp is not None:
            self.temp_log.append(temp)

        return temp

    def get_results(self):
        # """
        # Returns final temperature estimate
        # """

        if len(self.temp_log) == 0:
            return {
                "temp_index": None,
                "temp_c_estimate": None
            }

        temp_array = np.array(self.temp_log)

        # Smooth estimate
        mean_temp = float(np.mean(temp_array))
        std_temp = float(np.std(temp_array))

        # Optional fake mapping to Celsius scale (for demo only)
        temp_c = 33 + (mean_temp / 255.0) * 10
        #print("Body Temperature successfully estimated:", temp_c)
        return {
            "temp_index": mean_temp,
            "temp_std": std_temp,
            "temp_c_estimate": round(temp_c, 2)
        }