# backend/register_body.py

import cv2
import numpy as np
from models.reid_model import ReIDModel
from models.detector import PersonDetector
from utils.embeddings import save_embedding


def register_body(person_id):
    cap = cv2.VideoCapture(0)
    detector = PersonDetector()
    reid = ReIDModel()

    print(f"\n[REGISTER] Registering body for: {person_id}")
    print("  - Stand fully in frame (head to toe if possible)")
    print("  - Press 's' multiple times from slightly different angles")
    print("  - Press ESC to quit\n")

    collected_embeddings = []
    TARGET_SAMPLES = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        detections = detector.detect(frame)
        person_found = False

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            body_crop = frame[y1:y2, x1:x2]
            if body_crop.size == 0:
                continue

            person_found = True
            color = (0, 255, 0)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, f"Press 's' ({len(collected_embeddings)}/{TARGET_SAMPLES})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if not person_found:
            cv2.putText(display, "No person detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow(f"Register Body — {person_id}", display)
        key = cv2.waitKey(1)

        if key == ord('s') and person_found:
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                body_crop = frame[y1:y2, x1:x2]
                if body_crop.size == 0:
                    continue
                emb = reid.get_embedding(body_crop)
                if emb is not None:
                    collected_embeddings.append(emb)
                    print(f"  Sample {len(collected_embeddings)}/{TARGET_SAMPLES} captured")
                    break

            if len(collected_embeddings) >= TARGET_SAMPLES:
                final_emb = np.mean(collected_embeddings, axis=0)
                final_emb = final_emb / np.linalg.norm(final_emb)
                save_embedding(person_id, final_emb, "body")
                print(f"\n[REGISTER] ✅ Saved body embedding for {person_id} (averaged {TARGET_SAMPLES} samples)")
                break

        elif key == ord('s') and not person_found:
            print("  ⚠️  No person detected")

        elif key == 27:
            print("[REGISTER] Cancelled")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    person_id = input("Enter Person ID: ")
    register_body(person_id)