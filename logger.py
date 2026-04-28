# ============================================================
#  logger.py  –  Logging utility for Smart Attendance System
# ============================================================

import logging
import os
from config import LOG_FILE_PATH

os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("SmartAttendance")


def log_info(msg: str):
    logger.info(msg)

def log_warning(msg: str):
    logger.warning(msg)

def log_error(msg: str):
    logger.error(msg)
