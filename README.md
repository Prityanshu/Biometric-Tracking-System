markdown# 🧠 Biometric Tracking System using AI & Computer Vision

A **multi-modal biometric identification and tracking system** that detects, identifies, and tracks individuals across a campus environment using **face recognition, body re-identification, gait analysis, and visual attributes**.

---

## 🎯 Project Objective

To build an intelligent surveillance system capable of:

- Identifying individuals using facial features
- Recognizing people using body structure (ReID)
- Identifying individuals based on walking patterns (gait)
- Searching a person using image or description
- Tracking a person across multiple cameras
- Providing real-time location output

---

## 🏗️ System Architecture
```
Camera Input (IoT Streams)
        ↓
Person Detection (YOLOv8)
        ↓
Feature Extraction
├── Face Embedding
├── Body Embedding (ReID)
├── Gait Embedding
└── Attribute Extraction
        ↓
Feature Fusion + Matching Engine
        ↓
Identity / Appearance Matching
        ↓
Multi-Camera Tracking
        ↓
📍 Location Output + Dashboard
```

---

## 🚀 Features Implemented

### ✅ Phase 1 — Person Detection
- YOLOv8 based real-time human detection
- Bounding box extraction from video streams

### ✅ Phase 2 — Face Recognition
- Face embeddings using DeepFace (FaceNet)
- Identity registration and recognition
- Works in frontal and semi-profile views

### ✅ Phase 3 — Body Re-Identification (ReID)
- Body feature embeddings using ResNet50
- Identifies individuals even when face is not visible
- Robust across camera angles and clothing variation (partially)

### ✅ Phase 4 — Gait Recognition
- Silhouette-based gait feature extraction
- Temporal averaging of walking patterns
- Identifies individuals using walking style
- Works even when face is occluded

---

## 🧪 Testing Methodology

The system was evaluated using:

- **Positive tests**: Registered individual correctly identified
- **Negative tests**: Other individuals correctly rejected
- **Robustness tests**: Different clothes, side view walking, low light conditions

### Example Results

| Person | Score | Result |
|--------|-------|--------|
| Registered user | 0.85 – 0.95 | ✅ Correct |
| Other person 1 | 0.30 – 0.45 | ❌ Rejected |
| Other person 2 | 0.35 – 0.50 | ❌ Rejected |

---

## 🧰 Tech Stack

| Category | Tools |
|----------|-------|
| Programming | Python 3.10+ |
| Computer Vision | OpenCV, Ultralytics YOLOv8 |
| Deep Learning / AI | PyTorch, DeepFace (FaceNet), ResNet50 |
| Utilities | NumPy, Scikit-learn, FAISS (planned) |

---

## 📁 Project Structure
```
Biometric-Tracking-System/
│
├── backend/
│   ├── register.py
│   ├── recognize.py
│   ├── register_body.py
│   ├── recognize_body.py
│   ├── register_gait.py
│   └── recognize_gait.py
│
├── models/
│   ├── detector.py
│   ├── face_model.py
│   ├── reid_model.py
│   └── gait_model.py
│
├── utils/
│   ├── embeddings.py
│   ├── similarity.py
│   └── config.py
│
├── embeddings_db/
├── datasets/
├── iot_stream/
├── dashboard/
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository
```bash
git clone https://github.com/Prityanshu/Biometric-Tracking-System.git
cd Biometric-Tracking-System
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 🔹 Register Face
```bash
python -m backend.register
```

### 🔹 Recognize Face
```bash
python -m backend.recognize
```

### 🔹 Register Body (ReID)
```bash
python -m backend.register_body
```

### 🔹 Recognize Body
```bash
python -m backend.recognize_body
```

### 🔹 Register Gait
```bash
python -m backend.register_gait
```
> Walk in front of camera and press `S` to save.

### 🔹 Recognize Gait
```bash
python -m backend.recognize_gait
```

---

## 🎯 Current Capabilities

- ✔ Face-based identification
- ✔ Body-based identification
- ✔ Gait-based identification
- ✔ Real-time webcam inference
- ✔ Embedding-based similarity matching

---

## 🔮 Future Work (Upcoming Phases)

| Phase | Feature | Description |
|-------|---------|-------------|
| 🚧 Phase 5 | Search by Image | Input a snapshot → locate person across cameras |
| 🚧 Phase 6 | Attribute-Based Search | Search by shirt color, pant color, height, body type, accessories |
| 🚧 Phase 7 | Multi-Camera Tracking | Track identity across multiple streams with real-time location |
| 🚧 Phase 8 | Dashboard & Visualization | Live monitoring, detection overlay, campus map view |

---

## 🎓 Academic Relevance

This project demonstrates concepts from Computer Vision, Machine Learning, Deep Learning, Pattern Recognition, IoT Systems, Surveillance Systems, and Multi-modal Biometric Authentication.

---

## 📌 Key Concepts Used

`Feature Embeddings` `Cosine Similarity` `Object Detection` `Person Re-Identification` `Gait Signature Extraction` `Multi-modal Biometrics`

---

## 👨‍💻 Author

**Prityanshu Yadav** — B.Tech Final Year Project

---

## 📜 License

This project is for academic and research purposes.

---

## ⭐ Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [DeepFace](https://github.com/serengil/deepface)
- [PyTorch Community](https://pytorch.org)
- [OpenCV](https://opencv.org)