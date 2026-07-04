import cv2
import threading
import tempfile
import os
import time
import copy

import state
from processor import run_pipeline

cap = None
frame_lock = threading.Lock()
latest_frame_local = None


# CONNECT
def connect(rtsp_url):
    global cap

    disconnect()

    print(f"[INFO] Connecting to {rtsp_url}")

    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("[ERROR] Cannot open RTSP stream")
        state.connected = False
        return False

    state.connected = True
    state.running = True

    print("[INFO] Connected successfully")

    # ONLY ONE reader thread
    threading.Thread(target=frame_reader, daemon=True).start()

    # pipeline worker separate
    threading.Thread(target=pipeline_worker, daemon=True).start()

    return True

# DISCONNECT
def disconnect():
    global cap

    state.running = False
    state.connected = False

    if cap is not None:
        cap.release()
        cap = None

    print("[INFO] Stream disconnected")


# SINGLE FRAME READER
def frame_reader():
    global cap, latest_frame_local

    print("[INFO] Frame reader started")
    while state.running and cap is not None:
        ret, frame = cap.read()
        if not ret:
            continue
        with frame_lock:
            latest_frame_local = frame.copy()
            state.latest_frame = latest_frame_local
    print("[INFO] Frame reader stopped")

# PIPELINE WORKER
def pipeline_worker():

    global latest_frame_local

    while state.running:

        if cap is None:
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

        temp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        filename = temp.name
        temp.close()

        writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        frame_count = int(fps * 10)

        print("[INFO] Recording 10 seconds for pipeline...")

        recorded = 0

        while recorded < frame_count and state.running:

            with frame_lock:
                frame = None if latest_frame_local is None else latest_frame_local.copy()

            if frame is None:
                continue

            writer.write(frame)
            recorded += 1

        writer.release()

        print("[INFO] Processing pipeline...")

        try:
            run_pipeline(filename)

        except Exception as e:
            print("[PIPELINE ERROR]", e)

        if os.path.exists(filename):
            os.remove(filename)

        time.sleep(2)

    print("[INFO] Pipeline worker stopped")