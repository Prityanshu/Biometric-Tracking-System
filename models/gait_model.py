# models/gait_model.py

import cv2
import numpy as np


class GaitModel:
    def __init__(self):
        pass

    def get_silhouette(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
        silhouette = cv2.resize(thresh, (64, 128))
        return silhouette

    def extract_gait_embedding(self, frame_sequence):
        silhouettes = []
        for frame in frame_sequence:
            if frame is None or frame.size == 0:
                continue
            sil = self.get_silhouette(frame)
            silhouettes.append(sil)

        if len(silhouettes) == 0:
            return None

        silhouettes = np.array(silhouettes)
        avg_silhouette = np.mean(silhouettes, axis=0)
        avg_silhouette = avg_silhouette / 255.0
        embedding = avg_silhouette.flatten()

        # normalize so cosine similarity works correctly
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None
        return embedding / norm

    # Alias so tracker can call get_embedding consistently
    def get_embedding(self, frame_sequence):
        return self.extract_gait_embedding(frame_sequence)