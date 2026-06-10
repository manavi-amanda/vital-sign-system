import joblib
import pandas as pd
import os
import numpy as np
import traceback

# ---------------------------
# MODEL PATH
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "blood_pressure_model.pkl")

print("\n[BP INIT] MODULE LOADED")

model = None

# ---------------------------
# LOAD MODEL SAFELY
# ---------------------------
try:
    #print("[INFO] Loading model from:", MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    print("[INFO] BP Model SUCCESS - Model loaded ✔")

except Exception as e:
    print("[ERROR] Failed to load model:")
    print(str(e))
    traceback.print_exc()

    model = None


# ---------------------------
# PREDICTION FUNCTION
# ---------------------------
def predict_blood_pressure(body_temp, heart_rate, age: None):

    try:
        # ---------------------------
        # CHECK MODEL
        # ---------------------------
        if model is None:
            return {
                "error": "BP model not loaded",
                "bp_category": None
            }

        # ---------------------------
        # FIXED AGE
        # ---------------------------
        if age is None:
            age = 30  # default age if not provided
        # ---------------------------
        # SAFE TYPE CONVERSION
        # ---------------------------
        body_temp = float(body_temp)
        heart_rate = float(heart_rate)

        # ---------------------------
        # VALIDATION
        # ---------------------------
        if np.isnan(body_temp) or np.isnan(heart_rate):
            return {
                "error": "Invalid NaN values in input",
                "bp_category": None
            }

        # ---------------------------
        # BUILD INPUT (MATCH TRAINING DATA EXACTLY)
        # ---------------------------
        data = {
            "Age": [age],
            "BodyTemp": [body_temp],
            "HeartRate": [heart_rate]
        }

        input_df = pd.DataFrame(data)

        #print("[BP DEBUG] Input:", input_df.to_dict())

        # ---------------------------
        # PREDICT
        # ---------------------------
        prediction_code = model.predict(input_df)[0]

        categories = {
            0: "Low",
            1: "Normal",
            2: "Elevated",
            3: "Stage 1 Hypertension",
            4: "Stage 2 Hypertension"
        }

        result = {
            "age": age,
            "body_temp": body_temp,
            "heart_rate": heart_rate,
            "bp_category": categories.get(int(prediction_code), "Unknown"),
            "prediction_code": int(prediction_code)
        }

        print("[INFO] Blood Pressure successfull. Prediction:", result["bp_category"])

        return result

    except Exception as e:
        print("[ERROR] Blood Pressure prediction failed:", str(e))
        traceback.print_exc()

        return {
            "error": str(e),
            "bp_category": None
        }