from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import tempfile
import os
from modules.respiratory.video_processor import process_video
from modules.heart_rate.hr_estimator import estimate_heart_rate
from modules.blood_pressure.predictor import predict_blood_pressure

router = APIRouter()


@router.get("/")
def home():
    return {"status": "running"}


@router.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):

    try:
        # IMPORT INSIDE FUNCTION
        

        file_bytes = await file.read()

        result = process_video(file_bytes)

        if not result or "final_bpm" not in result:
            raise HTTPException(status_code=500, detail="RR analysis failed")

        hr_result = estimate_heart_rate(result["final_bpm"])

        bp_result = predict_blood_pressure(
            body_temp=float(result["body_temperature"]["temp_c_estimate"]),
            heart_rate=float(hr_result["hr_estimated"])
        )

        return {
            "bpm": result["final_bpm"],
            "body_temperature": result["body_temperature"]["temp_c_estimate"],
            "heart_rate": hr_result["hr_estimated"],
            "blood_pressure": bp_result["bp_category"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/vitallens")
async def analyze_vitallens(
        file: UploadFile = File(..., description="Video file (mp4, avi, mov …)"),
        api_key: str = Form(..., description="VitalLens API key"),
        hybrid_x: float = Form(0.0, description="Hybrid mixing weight 0–1 (0=pure rPPG)"),
):
    """
    VitalLens rPPG pipeline with hybrid HR blending.

    Steps:
      1. VitalLens extracts HR_rgb and RR from the video's colour signal
      2. HR_resp  = RR × 4.5   (clinical ratio)
      3. HR_hybrid = (1 - x) × HR_rgb  +  x × HR_resp
      4. predict_blood_pressure using HR_hybrid

    Set hybrid_x = 0.0 to trust rPPG fully.
    Run hybrid_hr_tuner.py with ground-truth data to find the best x.
    """
    if not (0.0 <= hybrid_x <= 1.0):
        raise HTTPException(status_code=422, detail="hybrid_x must be between 0.0 and 1.0")

    try:
        from modules.heart_rate.vitals_api import process_vitallens
        from modules.heart_rate.hr_estimatorV2 import estimate_hybrid_heart_rate
        from modules.age_detector.age_detector import detect_age

        suffix = "." + (file.filename.rsplit(".", 1)[-1] if "." in file.filename else "mp4")
        file_bytes = await file.read()

        # Save uploaded video temporarily for age estimation
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file_bytes)
            temp_video_path = temp_file.name

        try:
            detected_age = detect_age(temp_video_path)
        finally:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

        vl_result = process_vitallens(file_bytes, api_key=api_key, suffix=suffix)
        
        hr_rgb = vl_result.get("heart_rate")
        rr_bpm = vl_result.get("respiratory_rate")
        
        if hr_rgb is None or rr_bpm is None:
            raise HTTPException(
                status_code=422,
                detail="VitalLens could not detect HR or RR — check lighting and face visibility."
            )
        
        # ── Step 2 & 3: Hybrid HR ───────────────────────────────────────────
        hr_result = estimate_hybrid_heart_rate(
            hr_rgb=hr_rgb,
            rr_bpm=rr_bpm,
            x=hybrid_x,
        )

        #── Step 4: Blood pressure (reuse existing predictor) ───────────────
        bp_result = predict_blood_pressure(
            body_temp=37.0,  # VitalLens doesn't provide temp;
            heart_rate=hr_result["hr_estimated"], 
            age=detected_age  # use clinical default 37 °C
        )

        return {
            # Core vitals
            "heart_rate": hr_result["hr_estimated"],
            "respiratory_rate": round(rr_bpm, 2),
            "hrv_sdnn_ms": vl_result.get("hrv_sdnn"),

            "age": detected_age,

            #Hybrid breakdown
            "hybrid_detail": {
                "hr_rgb": hr_result["hr_rgb"],
                "hr_resp": hr_result["hr_resp"],
                "hybrid_x": hr_result["hybrid_x"],
                "source": hr_result["source"],
            },
            
            # Signal quality
            "confidence": {
                "hr": vl_result.get("hr_confidence"),
                "rr": vl_result.get("rr_confidence"),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
