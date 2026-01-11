import faiss
import numpy as np
import os
from privacy_embedding import embed, build_projections, ROLE_RANK, model

def load_documents_from_folder(folder_path="docs"):
    documents = []

    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(folder_path, filename)

        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        role = None
        doc_id = None
        content_lines = []

        for line in lines:
            if line.startswith("ROLE:"):
                role = line.replace("ROLE:", "").strip().lower()
            elif line.startswith("ID:"):
                doc_id = line.replace("ID:", "").strip()
            else:
                content_lines.append(line)

        if role is None or doc_id is None:
            raise ValueError(f"Missing ROLE or ID in {filename}")

        documents.append({
            "id": doc_id,
            "role": role,
            "text": "\n".join(content_lines).strip()
        })

    return documents

DOCUMENTS = load_documents_from_folder("docs")

dim = model.get_sentence_embedding_dimension()
PROJECTIONS = build_projections(dim)

indices = {}
doc_maps = {}

for user_role in ROLE_RANK:
    allowed = [
        d for d in DOCUMENTS
        if ROLE_RANK[d["role"]] <= ROLE_RANK[user_role]
    ]

    vectors, mapping = [], []
    for d in allowed:
        vectors.append(embed(d["text"], d["role"], PROJECTIONS))
        mapping.append(d)

    vectors = np.vstack(vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    indices[user_role] = index
    doc_maps[user_role] = mapping

    faiss.write_index(index, f"{user_role}.index")

np.save("doc_maps.npy", doc_maps, allow_pickle=True)
print("✅ Permission-aware vector DB built")