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