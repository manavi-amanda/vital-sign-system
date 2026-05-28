from fastapi import APIRouter, UploadFile, File, HTTPException
from modules.respiratory.video_processor import process_video
from modules.heart_rate.hr_estimator import estimate_heart_rate
from modules.blood_pressure.predictor import predict_blood_pressure

router = APIRouter()

@router.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):

    try:
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
            "blood_pressure": bp_result["bp_category"],
            #"details": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#REAL LIVE CAMERA STREAMING ENDPOINT
# from fastapi import APIRouter
# from pydantic import BaseModel
# import base64
# import numpy as np
# import cv2

# router = APIRouter()


# class FrameData(BaseModel):
#     image: str


# @router.post("/frame")
# async def receive_frame(data: FrameData):

#     # Decode base64 image
#     image_bytes = base64.b64decode(data.image)

#     np_array = np.frombuffer(image_bytes, np.uint8)

#     frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

#     # Process frame
#     print(frame.shape)

#     return {"message": "frame received"}