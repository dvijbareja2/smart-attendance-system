# ============================================================
#  inference.py  -  Run inference using SVM model
# ============================================================

import cv2
import time
import numpy as np
import pickle

from config import (INPUT_SOURCE, SCALE_FACTOR, MIN_NEIGHBORS, MIN_FACE_SIZE,
                    WINDOW_TITLE_ATTENDANCE)
from preprocessing import prepare_face
from utils import mark_attendance, reset_attendance, draw_face_box, open_camera
from logger import log_info, log_warning

SVM_MODEL_PATH = "svm_model.pkl"
SVM_THRESHOLD  = 0.6


def load_face_cascade():
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if cascade.empty():
        raise RuntimeError("Failed to load Haar cascade.")
    return cascade


def load_svm_model():
    with open(SVM_MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    log_info("SVM model loaded successfully.")
    return data["model"], data["label_encoder"]


def run_inference_on_frame(frame, gray, cascade, model, le):
    start = time.perf_counter()

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=MIN_FACE_SIZE,
    )

    results = []

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        face = prepare_face(roi)

        if face is None:
            continue

        flat = face.flatten().reshape(1, -1)
        proba = model.predict_proba(flat)[0]
        confidence = np.max(proba)
        label_idx  = np.argmax(proba)
        name = le.inverse_transform([label_idx])[0] if confidence >= SVM_THRESHOLD else "Unknown"

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        if name != "Unknown":
            mark_attendance(name)

        label = f"{name} ({confidence*100:.1f}%)"
        draw_face_box(frame, x, y, w, h, label, color)
        results.append((name, confidence, x, y, w, h))

    inference_time = time.perf_counter() - start
    return frame, results, inference_time


def run(model, le):
    cascade = load_face_cascade()
    cap     = open_camera(INPUT_SOURCE)

    log_info("Inference loop started. Press q=quit  r=reset.")

    frame_count = 0
    fps_display = 0.0
    t_prev = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            log_warning("Frame capture failed - exiting loop.")
            break

        frame = cv2.flip(frame, 1)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame, results, inf_time = run_inference_on_frame(
            frame, gray, cascade, model, le
        )

        frame_count += 1
        t_now = time.perf_counter()
        if t_now - t_prev >= 1.0:
            fps_display = frame_count / (t_now - t_prev)
            frame_count = 0
            t_prev = t_now

        cv2.putText(frame,
                    f"FPS: {fps_display:.1f}  Inf: {inf_time*1000:.1f}ms",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2)

        cv2.imshow(WINDOW_TITLE_ATTENDANCE, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            log_info("User quit.")
            break
        if key == ord("r"):
            reset_attendance()

    cap.release()
    cv2.destroyAllWindows()