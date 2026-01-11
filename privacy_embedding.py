import numpy as np
from sentence_transformers import SentenceTransformer

ROLE_RANK = {
    "intern": 0,
    "developer": 1,
    "manager": 2
}

model = SentenceTransformer("all-MiniLM-L6-v2")

def make_projection(dim, keep_ratio):
    k = int(dim * keep_ratio)
    P = np.zeros((dim, dim))
    P[:k, :k] = np.eye(k)
    return P

def build_projections(dim):
    return {
        "intern": make_projection(dim, 0.3),
        "developer": make_projection(dim, 0.6),
        "manager": make_projection(dim, 1.0)
    }

def embed(text, role, projections):
    vec = model.encode([text], normalize_embeddings=True)[0]
    vec = projections[role] @ vec
    vec = vec / (np.linalg.norm(vec) + 1e-9)
    return vec.astype("float32")