from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread
import uvicorn

from app.routes import router
from app.rtsp_worker import start_rtsp_processing

app = FastAPI(
    title="Vital Signs API",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Existing API routes
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """
    Start RTSP processing in background
    while keeping FastAPI routes active.
    """

    print("Starting RTSP worker...")

    rtsp_thread = Thread(
        target=start_rtsp_processing,
        daemon=True
    )

    rtsp_thread.start()

    print("RTSP worker started.")


@app.get("/health")
def health():
    return {
        "status": "running",
        "rtsp_worker": "started"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )