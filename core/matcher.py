import os
import numpy as np
from core.fusion_engine import FusionEngine


def cosine_similarity(a, b):
    if a is None or b is None:
        return None
    a = np.array(a, dtype=np.float32).flatten()
    b = np.array(b, dtype=np.float32).flatten()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None
    return float(np.dot(a / na, b / nb))


class Matcher:
    def __init__(self, db_path="embeddings_db"):
        self.db_path = db_path
        self.database = {}
        self.fusion = FusionEngine()
        self.load_database()

        # When face IS available: threshold well above face cross-sim (0.15)
        # When face is NOT available: body/gait cross-sim > 0.83, refuse to guess
        self.THRESHOLD_WITH_FACE    = 0.45
        self.THRESHOLD_WITHOUT_FACE = 0.99   # effectively disabled
        self.MARGIN = 0.05

    def load_database(self):
        self.database = {}

        if not os.path.exists(self.db_path):
            print(f"[DB LOAD] ⚠️  Folder not found: {self.db_path}")
            return

        files = os.listdir(self.db_path)
        print(f"\n[DB LOAD] Files: {files}\n")

        for file in files:
            if not file.endswith(".npy"):
                continue
            parts    = file.replace(".npy", "").split("_")
            if len(parts) < 2:
                continue
            name     = parts[0]
            modality = parts[1].lower()
            if modality not in ("face", "body", "gait"):
                continue
            try:
                emb = np.load(os.path.join(self.db_path, file))
                if name not in self.database:
                    self.database[name] = {}
                self.database[name][modality] = emb
                print(f"[DB LOAD] ✅  {name} → {modality}  shape={emb.shape}")
            except Exception as e:
                print(f"[DB LOAD] ❌  {file}: {e}")

        print(f"\n[DB LOAD] Loaded: {list(self.database.keys())}\n")

    def identify(self, face_emb=None, body_emb=None, gait_emb=None):
        if not self.database:
            return "Unknown", 0.0

        scores = []

        for person, data in self.database.items():
            face_sim = cosine_similarity(face_emb, data.get("face"))
            body_sim = cosine_similarity(body_emb, data.get("body"))
            gait_sim = cosine_similarity(gait_emb, data.get("gait"))

            final_score, trusted = self.fusion.compute_final_score(
                face_score=face_sim,
                body_score=body_sim,
                gait_score=gait_sim,
                verbose=True
            )

            parts = []
            if face_sim is not None: parts.append(f"face={face_sim:.3f}")
            if body_sim is not None: parts.append(f"body={body_sim:.3f}")
            if gait_sim is not None: parts.append(f"gait={gait_sim:.3f}")
            print(f"[MATCHER] {person:15s} | {' | '.join(parts)} | final={final_score:.3f}")

            scores.append((person, final_score, trusted))

        scores.sort(key=lambda x: x[1], reverse=True)

        best_person, best_score, best_trusted = scores[0]
        second_score = scores[1][1] if len(scores) > 1 else 0.0

        threshold = self.THRESHOLD_WITH_FACE if best_trusted else self.THRESHOLD_WITHOUT_FACE

        print(f"[MATCHER] Best={best_person} ({best_score:.3f}) | 2nd={second_score:.3f} | "
              f"threshold={threshold} | trusted={best_trusted}")

        if best_score >= threshold and (best_score - second_score) >= self.MARGIN:
            print(f"[MATCHER] ✅  → {best_person}")
            return best_person, best_score
        else:
            reason = "no face — refusing to guess" if not best_trusted else \
                     "score too low" if best_score < threshold else "margin too small"
            print(f"[MATCHER] ❌  → Unknown ({reason})")
            return "Unknown", best_score