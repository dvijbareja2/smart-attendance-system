# Smart Attendance System

## 1. Project Title
**Smart Attendance System using Face Recognition on Jetson Nano**

---

## 2. Problem Statement
Manual attendance marking is time-consuming, error-prone, and vulnerable to proxy fraud. This project automatically detects and recognises faces from a live camera feed and records attendance in a CSV file with zero human intervention.

The system is designed for resource-constrained environments like schools and small offices where cloud connectivity cannot be guaranteed and real-time response is required.

---

## 3. Role of Edge Computing

| Component | Runs On |
|---|---|
| Face detection (Haar Cascade) | Jetson Nano (CPU) |
| Face preprocessing & SVM matching | Jetson Nano (CPU) |
| Attendance CSV logging | Jetson Nano (local storage) |
| Dataset capture | Jetson Nano |

**Why edge computing instead of cloud-only?**
- **Reduced latency**: inference happens in milliseconds locally with no round-trip to a remote server
- **Offline capability**: works without any internet connection
- **Privacy**: facial images and attendance data never leave the device
- **Efficiency**: lightweight SVM runs comfortably on Jetson Nano's 4GB RAM

---

## 4. Methodology / Approach

**Overall pipeline:**
```
Input (Webcam) → Preprocessing → SVM Model → Output (CSV + Live Display)
```

| Stage | Description |
|---|---|
| **Input** | OpenCV reads frames from webcam at native FPS |
| **Preprocessing** | Convert to greyscale → detect face ROI → resize to 100×100 → histogram equalisation |
| **Model** | SVM classifies face using learned feature vectors |
| **Output** | Annotated live video + attendance entry appended to attendance.csv |

---

## 5. Model Details

- **Final Model**: Support Vector Machine (SVM) with RBF kernel
- **Input size**: 100 × 100 pixels, single channel (greyscale), flattened to 10,000 features
- **Framework**: scikit-learn
- **Three models were trained and compared**:
  - MSE (baseline) — simple pixel matching
  - SVM — best performer at 97.7% accuracy
  - CNN — deep learning model, 97.7% accuracy

---

## 6. Training Details

- **Dataset**: 650 images across 13 people, collected using preprocessing_capture.py at 3 FPS
- **Training procedure**: 80% train / 20% test split across all models
- **Performance graphs**: See cnn_training_graph.png, svm_training_graph.png, model_comparison.png

---

## 7. Results / Output

**Project Demo Video**: https://youtu.be/F3zdZi5-uAA

- **System output**: Annotated live video showing detected faces with name and confidence score. Green box = recognised, Red box = unknown. FPS and inference time overlaid on frame.
- **Performance metrics**:

| Model | Accuracy | Type |
|---|---|---|
| MSE | 20.8% | Baseline |
| LBPH | 22.3% | Classic CV |
| Eigenfaces | 20.8% | Classic CV |
| CNN | 97.7% | Deep Learning |
| SVM | 97.7% | Machine Learning |

- **Performance comparison**:

| Metric | Normal PC (i5) | Jetson Nano |
|---|---|---|
| Inference time | ~10 ms | ~30 ms |
| FPS | ~25 FPS | ~12 FPS |
| Power draw | ~65 W | ~5 W |

---

## 8. Setup Instructions

### Installation of dependencies
```bash
git clone https://github.com/dvijbareja2/smart-attendance-system
cd smart_attendance
pip install -r Requirements.txt
```

### Commands to execute the project
```bash
# Step 1 - Collect face dataset for a new person
python preprocessing_capture.py

# Step 2 - Train all models
python training.py

# Step 3 - Run the attendance system
python main.py
# Press q to quit | Press r to reset attendance
```

**Output file**: attendance.csv is created automatically in the project root.