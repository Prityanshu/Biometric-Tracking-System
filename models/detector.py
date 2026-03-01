# models/detector.py

from ultralytics import YOLO
from utils.config import MODEL_PATH, CONF_THRESHOLD


class PersonDetector:
    def __init__(self):
        # Use ByteTrack so each person gets a persistent track_id
        # This means gait buffers stay per-person even when they cross paths
        self.model = YOLO(MODEL_PATH)

    def detect(self, frame):
        """
        Input:  frame (numpy image)
        Output: list of dicts with bbox, confidence, track_id
        """
        # persist=True enables ByteTrack across frames
        results = self.model.track(frame, persist=True, verbose=False)

        detections = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls = int(box.cls[0])

                # class 0 = person only
                if cls != 0:
                    continue

                conf = float(box.conf[0])
                if conf < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # track_id from ByteTrack — falls back to None if tracking unavailable
                track_id = int(box.id[0]) if box.id is not None else None

                detections.append({
                    "bbox":       (x1, y1, x2, y2),
                    "confidence": conf,
                    "class":      "person",
                    "track_id":   track_id
                })

        return detections