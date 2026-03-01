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
    print("  - Press 's' multiple times from slightly different positions/angles")
    print("  - Vary distance slightly between captures for robustness")
    print("  - Press ESC to quit\n")

    collected = []
    TARGET_SAMPLES = 15   # Fix 1 — increased from 10, stored as stack not average

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
            cv2.putText(display,
                        f"Press 's' [{len(collected)}/{TARGET_SAMPLES}]",
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
                    # Normalize each sample individually before storing
                    emb = emb / np.linalg.norm(emb)
                    collected.append(emb)
                    print(f"  Sample {len(collected)}/{TARGET_SAMPLES} captured  "
                          f"crop=({x2-x1}x{y2-y1})")
                    break

            if len(collected) >= TARGET_SAMPLES:
                # Fix 1 — save full stack (N, 512) instead of averaged vector
                # Matcher will score against each exemplar and take max
                stack = np.stack(collected)   # shape (TARGET_SAMPLES, 512)
                save_embedding(person_id, stack, "body")
                print(f"\n[REGISTER] ✅ Saved {TARGET_SAMPLES} body exemplars for {person_id}")
                print(f"           Shape: {stack.shape} — multi-exemplar matching enabled")
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