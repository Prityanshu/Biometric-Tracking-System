# backend/register_body.py

import cv2
from models.reid_model import ReIDModel
from models.detector import PersonDetector
from utils.embeddings import save_embedding

def register_body(person_id):
    cap = cv2.VideoCapture(0)

    detector = PersonDetector()
    reid = ReIDModel()

    print("Press 's' to capture body")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            body_crop = frame[y1:y2, x1:x2]

            cv2.rectangle(frame, (x1,y1),(x2,y2),(255,0,0),2)

        cv2.imshow("Register Body", frame)

        key = cv2.waitKey(1)

        if key == ord('s') and len(detections) > 0:
            emb = reid.get_embedding(body_crop)
            save_embedding(person_id, emb, "body")
            print(f"Saved body embedding for {person_id}")
            break

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    person_id = input("Enter Person ID: ")
    register_body(person_id)