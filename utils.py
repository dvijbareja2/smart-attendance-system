# ============================================================
#  utils.py  –  Helper & redundant utility functions
# ============================================================

import os
import cv2
import pandas as pd
from datetime import datetime
from config import ATTENDANCE_FILE
from logger import log_info, log_warning


# ── Attendance helpers ─────────────────────────────────────

def mark_attendance(name: str) -> bool:
    """
    Append an attendance entry for *name* if not already present today.
    Returns True when a new entry is written, False if already marked.
    """
    now      = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    already_marked = ((df["Name"] == name) & (df["Date"] == date_str)).any()
    if already_marked:
        return False

    new_row = pd.DataFrame({"Name": [name], "Date": [date_str], "Time": [time_str]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(ATTENDANCE_FILE, index=False)
    log_info(f"Attendance marked  →  {name}  {date_str}  {time_str}")
    return True


def reset_attendance() -> None:
    """Delete the attendance CSV file."""
    if os.path.exists(ATTENDANCE_FILE):
        os.remove(ATTENDANCE_FILE)
        log_info("Attendance file reset.")
    else:
        log_warning("Reset requested but attendance file not found.")


# ── Frame / display helpers ────────────────────────────────

def draw_face_box(frame, x: int, y: int, w: int, h: int,
                  label: str, color: tuple) -> None:
    """Draw a labelled bounding box on *frame* in-place."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
    cv2.putText(frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def open_camera(source) -> cv2.VideoCapture:
    """Open a VideoCapture and raise RuntimeError if it fails."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera / source: {source}")
    return cap
