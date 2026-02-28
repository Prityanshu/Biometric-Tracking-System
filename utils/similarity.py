# utils/similarity.py

import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_best_match(query_embedding, db_embeddings, threshold=0.6):
    best_match = "Unknown"
    best_score = -1

    for person_id, emb in db_embeddings.items():
        score = cosine_similarity(query_embedding, emb)

        if score > best_score:
            best_score = score
            best_match = person_id

    if best_score < threshold:
        return "Unknown", best_score

    return best_match, best_score