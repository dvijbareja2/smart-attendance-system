import cv2
import numpy as np
from config import FACE_IMAGE_SIZE


def prepare_face(gray_roi: np.ndarray,
                 target_size: tuple = FACE_IMAGE_SIZE):

    if gray_roi is None or gray_roi.size == 0:
        return None

    face = cv2.resize(gray_roi, target_size)

    # 🔥 KEY FIXES
    face = cv2.equalizeHist(face)
    face = cv2.GaussianBlur(face, (5, 5), 0)

    return face


def load_dataset(dataset_path: str,
                 target_size: tuple = FACE_IMAGE_SIZE):

    import os
    faces = []
    names = []

    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            processed = prepare_face(img, target_size)
            if processed is not None:
                faces.append(processed)
                names.append(person_name)

    return faces, names


def extract_mse_score(face_a: np.ndarray, face_b: np.ndarray) -> float:
    return float(np.mean((face_a.astype(np.float32) - face_b.astype(np.float32)) ** 2))
