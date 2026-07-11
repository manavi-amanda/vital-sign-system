import asyncio
import fractions
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from av import VideoFrame
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

pcs = set()


# WebRTC Custom Media Track
class CameraStreamTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self):
        super().__init__()
        self.pts = 0

    async def recv(self):
        await asyncio.sleep(1 / 30)
        while state.latest_frame is None or not state.connected:
            await asyncio.sleep(0.05)

        frame = state.latest_frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")

        self.pts += int(90000 / 30)
        video_frame.pts = self.pts
        video_frame.time_base = fractions.Fraction(1, 90000)
        return video_frame


# Request Models
class ConnectRequest(BaseModel):
    rtspUrl: str


class PipelineRequest(BaseModel):
    pipelineType: str


class ModeRequest(BaseModel):
    mode: int


class WebRtcOffer(BaseModel):
    sdp: str
    type: str


# ==========================================
# 1. LIVE TELEMETRY STATUS VIA WEBSOCKET
# ==========================================
@app.websocket("/ws/status")
async def websocket_status(websocket: WebSocket):
    await websocket.accept()
    print("[INFO] Status WebSocket client connected")

    try:
        while True:
            try:
                # 1. Snapshot raw data safely using list() to prevent "dictionary changed size" exceptions
                raw_snapshot = dict(list(state.latest_result.items()))

                # 2. Sanitize data (auto-convert NumPy types to clean Python primitives)
                clean_result = {}
                for key, val in raw_snapshot.items():
                    if hasattr(val, "item") and not isinstance(val, (str, int, float, bool, list, dict)):
                        clean_result[key] = val.item()  # Converts np.float32, np.int64, etc. to native primitives
                    else:
                        clean_result[key] = val

                # 3. Handle automated state flag handovers
                if clean_result.get("thermal_completed"):
                    state.latest_result.clear()

                # 4. Construct payload
                payload = {
                    "connected": state.connected,
                    "running": state.pipeline_running,
                    "result": clean_result,
                    "pipeline": state.pipeline_type
                }

                # 5. Send out JSON frame
                await websocket.send_json(payload)

            except RuntimeError:
                # Catch block handles rare event where the dict shifts sizes while copying; skips frame safely
                pass
            except TypeError as json_err:
                print(f"[WS ERROR] JSON Serialization issue: {json_err}. Verify your AI worker thread variables.")
            except Exception as loop_err:
                print(f"[WS ERROR] Latent data stream exception: {loop_err}")

            # Push telemetry frames once every second
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        print("[INFO] Status WebSocket client disconnected")
    except Exception as critical_err:
        print(f"[WS CRITICAL] Connection dropped unexpectedly: {critical_err}")


# ==========================================
# 2. VIDEO STREAM VIA WEBRTC
# ==========================================
@app.post("/offer")
async def webrtc_offer(params: WebRtcOffer):
    offer = RTCSessionDescription(sdp=params.sdp, type=params.type)
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)

    video_track = CameraStreamTrack()
    pc.addTrack(video_track)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }


# ==========================================
# 3. STANDARD SYSTEM HTTP ENDPOINTS
# ==========================================
@app.post("/connect")
def connect_camera(req: ConnectRequest):
    result = connect(req.rtspUrl)
    return {"connected": result, "status": "starting"}


@app.post("/start-pipeline")
def start_pipeline(req: PipelineRequest):
    if state.pipeline_running:
        return {"started": False, "reason": "Pipeline already running"}
    state.pipeline_type = req.pipelineType
    state.pipeline_running = True
    return {"started": True, "pipeline": state.pipeline_type}


@app.post("/disconnect")
def disconnect_camera():
    disconnect()
    return {"connected": False}


@app.post("/set-mode")
def set_mode(req: ModeRequest):
    state.mode = req.mode
    return {"mode": state.mode}


@app.post("/reset-pipeline")
def reset_pipeline():
    state.pipeline_running = False
    state.pipeline_type = None
    time.sleep(0.2)
    state.latest_result.clear()
    return {"reset": True}


@app.on_event("shutdown")
async def on_shutdown():
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


@app.get("/")
def root():
    return {"message": "RTSP WebRTC + WebSocket Backend Running"}