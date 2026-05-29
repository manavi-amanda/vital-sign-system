from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter()


@router.get("/")
def home():
    return {"status": "running"}


@router.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):

    try:
        # IMPORT INSIDE FUNCTION
        from modules.respiratory.video_processor import process_video
        from modules.heart_rate.hr_estimator import estimate_heart_rate
        from modules.blood_pressure.predictor import predict_blood_pressure

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