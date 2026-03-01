# utils/embeddings.py

import numpy as np
import os

DB_PATH = "embeddings_db"
os.makedirs(DB_PATH, exist_ok=True)


def save_embedding(person_id, embedding, modality="face"):
    """
    Save embedding(s) for a person.

    embedding can be:
      - shape (512,)    — single averaged vector (legacy, still supported)
      - shape (N, 512)  — stack of N exemplars (Fix 1 — multi-exemplar matching)

    Multi-exemplar is preferred: at match time we score against each exemplar
    and take the max, giving much better coverage of appearance variation.
    This matters most when scaling to 8+ registered people.
    """
    file_path = os.path.join(DB_PATH, f"{person_id}_{modality}.npy")
    np.save(file_path, embedding)
    shape = np.array(embedding).shape
    print(f"[EMBEDDINGS] Saved {person_id}_{modality}.npy  shape={shape}")


def load_all_embeddings(modality="face"):
    """
    Load all embeddings for a given modality.
    Returns dict of person_id → np.array (shape (512,) or (N, 512))
    Matcher.cosine_similarity handles both shapes automatically.
    """
    db = {}

    for file in os.listdir(DB_PATH):
        if file.endswith(f"_{modality}.npy"):
            person_id = file.split("_")[0]
            emb = np.load(os.path.join(DB_PATH, file))
            db[person_id] = emb

    return db