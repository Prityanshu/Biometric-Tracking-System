# utils/embeddings.py

import numpy as np
import os

DB_PATH = "embeddings_db"
os.makedirs(DB_PATH, exist_ok=True)

def save_embedding(person_id, embedding, modality="face"):
    file_path = os.path.join(DB_PATH, f"{person_id}_{modality}.npy")
    np.save(file_path, embedding)

def load_all_embeddings(modality="face"):
    db = {}

    for file in os.listdir(DB_PATH):
        if file.endswith(f"_{modality}.npy"):
            person_id = file.split("_")[0]
            emb = np.load(os.path.join(DB_PATH, file))
            db[person_id] = emb

    return db