# models/gait_model.py

import cv2
import numpy as np

class GaitModel:
    def __init__(self):
        pass

    def get_silhouette(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blur to remove noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold to binary silhouette
        _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

        # Resize to fixed size for consistency
        silhouette = cv2.resize(thresh, (64, 128))

        return silhouette

    def extract_gait_embedding(self, frame_sequence):
        silhouettes = []

        for frame in frame_sequence:
            sil = self.get_silhouette(frame)
            silhouettes.append(sil)

        silhouettes = np.array(silhouettes)

        # average silhouette over time (temporal aggregation)
        avg_silhouette = np.mean(silhouettes, axis=0)

        # normalize
        avg_silhouette = avg_silhouette / 255.0

        # flatten to vector
        embedding = avg_silhouette.flatten()

        return embedding