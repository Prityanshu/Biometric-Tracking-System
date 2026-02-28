# models/face_model.py

from deepface import DeepFace
import numpy as np

class FaceRecognizer:
    def __init__(self):
        self.model_name = "Facenet"

    def get_embedding(self, face_img):
        """
        Input: cropped face image (numpy array)
        Output: embedding vector (numpy array)
        """
        try:
            result = DeepFace.represent(
                face_img,
                model_name=self.model_name,
                enforce_detection=False
            )

            embedding = result[0]["embedding"]
            return np.array(embedding)

        except Exception as e:
            print("Face embedding error:", e)
            return None