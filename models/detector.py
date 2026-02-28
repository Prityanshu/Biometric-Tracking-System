# models/detector.py

from ultralytics import YOLO
import cv2
from utils.config import MODEL_PATH, CONF_THRESHOLD

class PersonDetector:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)

    def detect(self, frame):
        """
        Input: frame (numpy image)
        Output: list of detections (dict)
        """

        results = self.model(frame)

        detections = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                # class 0 = person
                if cls == 0:
                    conf = float(box.conf[0])

                    if conf < CONF_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    detection = {
                        "bbox": (x1, y1, x2, y2),
                        "confidence": conf,
                        "class": "person"
                    }

                    detections.append(detection)

        return detections