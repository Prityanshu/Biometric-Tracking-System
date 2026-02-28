# backend/register_gait.py

import cv2
from models.detector import PersonDetector
from models.gait_model import GaitModel
from utils.embeddings import save_embedding

def register_gait(person_id):
    cap = cv2.VideoCapture(0)

    detector = PersonDetector()
    gait_model = GaitModel()

    frame_sequence = []

    print("Walk naturally in front of camera. Press 's' to save gait.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            # safe crop
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            body_crop = frame[y1:y2, x1:x2]

            frame_sequence.append(body_crop)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow("Register Gait", frame)

        key = cv2.waitKey(1)

        if key == ord('s') and len(frame_sequence) > 15:
            emb = gait_model.extract_gait_embedding(frame_sequence[-20:])
            save_embedding(person_id, emb, "gait")
            print(f"Saved gait embedding for {person_id}")
            break

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    person_id = input("Enter Person ID: ")
    register_gait(person_id)