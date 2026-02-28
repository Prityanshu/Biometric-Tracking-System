# backend/register.py

import cv2
from models.face_model import FaceRecognizer
from utils.embeddings import save_embedding

def register_person(person_id):
    cap = cv2.VideoCapture(0)
    face_model = FaceRecognizer()

    print("Press 's' to capture face")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Register Face", frame)

        key = cv2.waitKey(1)

        if key == ord('s'):
            emb = face_model.get_embedding(frame)

            if emb is not None:
                save_embedding(person_id, emb, "face")
                print(f"Saved face embedding for {person_id}")
                break

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    person_id = input("Enter Person ID: ")
    register_person(person_id)