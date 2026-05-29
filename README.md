# 🩺 Vital Sign Monitoring System (AI-Based Video Analysis)
 
An AI-powered healthcare monitoring system that analyzes video input and estimates key physiological vital signs using computer vision and machine learning.
 
### It extracts:
 
- 🌬️ **Respiratory Rate** (BPM)
- ❤️ **Heart Rate** (estimated from RR / ML model)
- 🌡️ **Body Temperature** (estimated / proxy-based)
- 🩸 **Blood Pressure Category** (ML prediction)
---
 
## 🚀 Features
 
- 📹 Upload video for health analysis
- 👤 Face detection using YOLOv8
- 📊 Respiratory signal extraction from facial ROI
- ❤️ Heart rate estimation using RR-based logic / ML model
- 🌡️ Body temperature estimation module (proxy-based)
- 🩸 Blood pressure classification model
- ⚡ FastAPI REST API backend
- 🌐 Easily deployable to cloud platforms
---
 
## 🧠 System Workflow
 
```
Video Input
   ↓
Frame Extraction (OpenCV)
   ↓
Person / Face Detection (YOLOv8)
   ↓
ROI Intensity Signal Extraction
   ↓
Respiratory Rate Calculation
   ↓
Heart Rate Estimation (RR-based / ML model)
   ↓
Temperature Estimation Module
   ↓
Blood Pressure Prediction Model
   ↓
Final JSON Response
```
 
---
 
## 🛠️ Tech Stack
 
| Technology | Purpose |
|---|---|
| Python | Core language |
| FastAPI | REST API backend |
| OpenCV | Video/frame processing |
| YOLOv8 (Ultralytics) | Face & person detection |
| NumPy | Signal processing |
| Pandas | Data handling |
| Scikit-learn | ML models |
| PyTorch | Deep learning backend |
| Joblib | Model serialization |
 
---
 
## 📁 Project Structure
 
```
app/
│── main.py
│── routes.py
 
modules/
│
├── respiratory/
│   └── video_processor.py
│   └── respiratoryRate.py
│
├── heart_rate/
│   └── hr_estimator.py
│
├── temperature/
│   └── temp_estimator.py
│
├── blood_pressure/
│   └── predictor.py
 
models/
│── yolov8n.pt
│── yolov8n-face.pt
│── blood_pressure_model.pkl
```
 
---
 
## ⚙️ Installation
 
### 1. Clone Repository
 
```bash
git clone https://github.com/your-username/vital-sign-system.git
cd vital-sign-system
```
 
### 2. Create Virtual Environment
 
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```
 
**Linux / Mac:**
```bash
python -m venv venv
source venv/bin/activate
```
 
### 3. Install Dependencies
 
```bash
pip install -r requirements.txt
```
 
---
 
## ▶️ Run Locally
 
```bash
uvicorn main:app --reload
```
 
API will be available at:
 
```
http://127.0.0.1:8000
```
 
---
 
## 📡 API Endpoints
 
### 🔹 Health Check
 
**GET** `/`
 
**Response:**
```json
{
  "status": "running"
}
```
 
---
 
### 🔹 Analyze Video
 
**POST** `/analyze`
 
**Form Data:**
 
| Field | Type | Description |
|---|---|---|
| `file` | File | Video file (`.mp4`) |
 
**Example cURL Request:**
```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "file=@video.mp4"
```
 
**Example Response:**
```json
{
  "respiratory_rate": 18,
  "heart_rate": 78,
  "body_temperature": {
    "temp_index": 142.3,
    "temp_c_estimate": 36.7
  },
  "blood_pressure": "Normal"
}
```
 
---
 
## ☁️ Deployment Notes
 
> ⚠️ This project is computationally heavy due to YOLO + video processing.
 
**Minimum Requirements:**
 
- 1 GB RAM (2 GB+ recommended)
- CPU-only PyTorch recommended for cloud deployment
- Avoid long video uploads on free-tier servers
**Known Issues on Free Hosting:**
 
| Issue | Description |
|---|---|
| ⚠️ 502 Bad Gateway | Request timeouts during heavy processing |
| ⚠️ Memory Limit Exceeded | YOLO + video processing exceeds free-tier RAM |
| ⚠️ Slow Processing | Large videos cause significant delays |
 
---
 
## 🚨 Limitations
 
- Heavy processing load due to frame-by-frame analysis
- Accuracy depends on video quality and lighting conditions
- Temperature estimation is **not medically calibrated** (proxy-based)
- **Not suitable for real-time clinical diagnosis**
---
 
## 📄 License
 
This project is intended for research and educational purposes only. It is **not a certified medical device** and should not be used as a substitute for professional medical advice or diagnosis.
