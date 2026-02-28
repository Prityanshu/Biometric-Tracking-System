# Biometric Tracking System

A multi-modal biometric identification and campus tracking system using computer vision and deep learning.

## 🔹 Features Implemented (Phase 1)
- Real-time person detection using YOLOv8
- Live camera feed processing
- Bounding box visualization with confidence scores
## 🔹 Features Implemented (Phase 3)
- Person Re-Identification using body features
- ResNet50 deep feature embeddings
- Works even when face is not visible
## 🔹 Tech Stack
- Python
- OpenCV
- Ultralytics YOLOv8
- PyTorch

## 🔹 Project Structure
backend/
models/
iot_stream/
utils/
dashboard/

## 🔹 How to Run

1. Create virtual environment
2. Install dependencies:
   pip install -r requirements.txt

3. Run detection module:
   python -m iot_stream.camera_reader

## 🔹 Upcoming Features
- Face Recognition
- Person Re-Identification
- Gait Recognition
- Multi-camera tracking

## 🔹 Author
Prityanshu Yadav