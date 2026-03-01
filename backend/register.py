# backend/register.py

import cv2
import numpy as np
from models.face_model import FaceRecognizer
from models.detector import PersonDetector
from utils.embeddings import save_embedding


def register_person(person_id):
    cap = cv2.VideoCapture(0)
    face_model = FaceRecognizer()
    detector = PersonDetector()

    print(f"\n[REGISTER] Registering face for: {person_id}")
    print("  - Stand 1-1.5 metres from camera")
    print("  - Face well lit, looking straight at camera")
    print("  - Press 's' to capture samples (need 15)")
    print("  - ESC to quit\n")

    collected = []
    TARGET = 15   # more samples = more stable embedding

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        detections = detector.detect(frame)
        face_found = False

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_h, crop_w = crop.shape[:2]

            emb = face_model.get_embedding(crop)

            if emb is not None:
                face_found = True
                color = (0, 255, 0)
                status = f"✅ Face OK ({crop_w}x{crop_h}) — press 's' [{len(collected)}/{TARGET}]"
            else:
                color = (0, 165, 255)
                status = f"⚠️  Crop too small ({crop_w}x{crop_h}) — move closer!"

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, status, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if not face_found and not detections:
            cv2.putText(display, "No person detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow(f"Register Face — {person_id}", display)
        key = cv2.waitKey(1)

        if key == ord('s'):
            if face_found:
                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    crop = frame[y1:y2, x1:x2]
                    emb = face_model.get_embedding(crop)
                    if emb is not None:
                        collected.append(emb)
                        print(f"  Sample {len(collected)}/{TARGET} captured  crop=({x2-x1}x{y2-y1})")
                        break

                if len(collected) >= TARGET:
                    final_emb = np.mean(collected, axis=0)
                    final_emb = final_emb / np.linalg.norm(final_emb)
                    save_embedding(person_id, final_emb, "face")
                    print(f"\n[REGISTER] ✅ Saved face for {person_id} ({TARGET} samples averaged)")
                    break
            else:
                print("  ⚠️  Move closer to camera — crop too small for reliable embedding")

        elif key == 27:
            print("[REGISTER] Cancelled")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    person_id = input("Enter Person ID: ")
    register_person(person_id)