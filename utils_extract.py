# utils_extract.py

import os
import cv2
import numpy as np
from pathlib import Path
from mtcnn import MTCNN
from PIL import Image

def extract_face_from_one_video(video_path, save_dir, num_images=10, image_size=(224, 224)):
    """
    Ekstrak wajah dari satu video dan simpan ke save_dir/folder_video.
    """
    detector = MTCNN()

    # Buat folder baru berdasarkan nama file video
    filename_stem = Path(video_path).stem
    save_path = Path(save_dir).joinpath(filename_stem)
    save_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(n_frames // num_images, 1)

    frame_count = 0
    face_count = 0

    while cap.isOpened() and face_count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % interval == 0:
            boxes = detector.detect_faces(frame)
            if boxes:
                x1, y1, width, height = boxes[0]['box']
                x2, y2 = x1 + width, y1 + height

                # Crop wajah dan resize
                face = frame[y1:y2, x1:x2]
                face = cv2.resize(face, image_size)

                # Save wajah
                face_filename = save_path.joinpath(f"face_{face_count}.jpg")
                cv2.imwrite(str(face_filename), face)
                face_count += 1

    cap.release()