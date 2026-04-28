# ============================================================
#  main.py  -  Main entry file for Smart Attendance System
# ============================================================

import sys
from logger import log_info, log_error


def initialise_system():
    log_info("=" * 55)
    log_info("Smart Attendance System  -  Starting up")
    log_info("=" * 55)
    try:
        import cv2, numpy, pandas
        log_info("Dependencies OK  (OpenCV, NumPy, Pandas)")
    except ImportError as e:
        log_error(f"Missing dependency: {e}")
        sys.exit(1)


def load_input_source():
    from inference import load_svm_model
    log_info("Loading SVM model...")
    model, le = load_svm_model()
    log_info("Model loaded successfully.")
    return model, le


def call_inference_pipeline(model, le):
    from inference import run
    log_info("Launching inference pipeline...")
    run(model, le)


def display_and_output():
    from config import ATTENDANCE_FILE
    import os
    log_info("Session ended.")
    if os.path.exists(ATTENDANCE_FILE):
        import pandas as pd
        df = pd.read_csv(ATTENDANCE_FILE)
        log_info(f"Attendance records today: {len(df)}")
        print("\n--- Today's Attendance ---")
        print(df.to_string(index=False))
    else:
        log_info("No attendance records found.")


if __name__ == "__main__":
    initialise_system()
    model, le = load_input_source()
    call_inference_pipeline(model, le)
    display_and_output()