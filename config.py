import os

MODEL_PATH = "haarcascade_frontalface_default.xml"

FACE_IMAGE_SIZE   = (100, 100)
CAPTURE_IMAGE_SIZE = (200, 200)

# 🔥 FIXED (was 2000)
MSE_THRESHOLD     = 6000

SCALE_FACTOR      = 1.1
MIN_NEIGHBORS     = 5
MIN_FACE_SIZE     = (80, 80)

INPUT_SOURCE = 0

DATASET_PATH     = "dataset"
ATTENDANCE_FILE  = "attendance.csv"
LOG_FILE_PATH    = os.path.join("logs", "system.log")

CAPTURE_TARGET_FPS   = 3
CAPTURE_MAX_IMAGES   = 50

WINDOW_TITLE_ATTENDANCE = "Smart Attendance System"
WINDOW_TITLE_CAPTURE    = "Face Capture"