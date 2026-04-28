import cv2
import os
import time

from config import (INPUT_SOURCE, SCALE_FACTOR, MIN_NEIGHBORS, MIN_FACE_SIZE,
                    CAPTURE_IMAGE_SIZE, CAPTURE_TARGET_FPS, CAPTURE_MAX_IMAGES,
                    DATASET_PATH, WINDOW_TITLE_CAPTURE)
from utils import open_camera
from logger import log_info


def capture_dataset(class_name: str) -> None:

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    output_folder = os.path.join(DATASET_PATH, class_name)
    os.makedirs(output_folder, exist_ok=True)

    cap = open_camera(INPUT_SOURCE)

    prev_time = 0.0
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_FACE_SIZE,
        )

        current_time = time.time()

        if current_time - prev_time >= 1 / CAPTURE_TARGET_FPS:
            if len(faces) == 1:
                (x, y, w, h) = faces[0]
                face_roi = gray[y:y+h, x:x+w]

                if face_roi.size == 0:
                    continue

                face_roi = cv2.resize(face_roi, CAPTURE_IMAGE_SIZE)
                face_roi = cv2.equalizeHist(face_roi)
                face_roi = cv2.GaussianBlur(face_roi, (5, 5), 0)

                file_path = os.path.join(output_folder, f"{count}.jpg")
                cv2.imwrite(file_path, face_roi)

                count += 1
                log_info(f"Saved {count}")

            prev_time = current_time

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(frame, f"{count}/{CAPTURE_MAX_IMAGES}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow(WINDOW_TITLE_CAPTURE, frame)

        if count >= CAPTURE_MAX_IMAGES:
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    name = input("Enter your name: ")
    capture_dataset(name)