from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import cv2
import time

import state
from stream import connect, disconnect

app = FastAPI()

# Allow Electron to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Model
class ConnectRequest(BaseModel):
    rtspUrl: str

class PipelineRequest(BaseModel):
    pipelineType: str


# Connect Camera
@app.post("/connect")
def connect_camera(req: ConnectRequest):

    print("REQUEST RECEIVED")
    result = connect(req.rtspUrl)
    print("RETURNING RESPONSE")

    return {
        "connected": result,
        "status": "starting"
    }

@app.post("/start-pipeline")
def start_pipeline(req: PipelineRequest):

    if state.pipeline_running:
        return {
            "started": False,
            "reason": "Pipeline already running"
        }

    state.pipeline_type = req.pipelineType
    state.pipeline_running = True

    return {
        "started": True,
        "pipeline": state.pipeline_type
    }


# Disconnect Camera

@app.post("/disconnect")
def disconnect_camera():

    disconnect()
    return {
        "connected": False
    }

class ModeRequest(BaseModel):
    mode: int


@app.post("/set-mode")
def set_mode(req: ModeRequest):

    state.mode = req.mode

    return {
        "mode": state.mode
    }


# Connection Status
@app.get("/status")
def status():

    result = state.latest_result.copy()

    # One-time notification
    if result.get("thermal_completed"):
        state.latest_result.clear()

    return {
        "connected": state.connected,
        "running": state.pipeline_running,
        "result": result,
        "pipeline": state.pipeline_type
    }

# Live Video Stream
def generate():

    while True:

        if not state.connected:
            time.sleep(0.1)
            continue

        if state.latest_frame is None:
            time.sleep(0.01)
            continue

        success, buffer = cv2.imencode(
            ".jpg",
            state.latest_frame
        )

        if not success:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + buffer.tobytes()
            + b'\r\n'
        )


@app.get("/video")
def video():

    return StreamingResponse(

        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"

    )


@app.post("/reset-pipeline")
def reset_pipeline():

    state.pipeline_running = False
    state.pipeline_type = None

    time.sleep(0.2)  # allow worker to exit safely

    state.latest_result.clear()

    return {"reset": True}
# Health Check

@app.get("/")
def root():

    return {
        "message": "RTSP Backend Running"
    }