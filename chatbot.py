import faiss
import numpy as np
import google.generativeai as genai
from privacy_embedding import embed, build_projections, ROLE_RANK, model
import os
from dotenv import load_dotenv

load_dotenv()


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

llm = genai.GenerativeModel("gemini-2.5-flash")

indices = {
    "intern": faiss.read_index("intern.index"),
    "developer": faiss.read_index("developer.index"),
    "manager": faiss.read_index("manager.index"),
}

doc_maps = np.load("doc_maps.npy", allow_pickle=True).item()

dim = model.get_sentence_embedding_dimension()
PROJECTIONS = build_projections(dim)

def retrieve_context(query, role, k=3):
    q_emb = embed(query, role, PROJECTIONS).reshape(1, -1)

    index = indices[role]
    docs = doc_maps[role]

    k = min(k, index.ntotal)
    scores, idxs = index.search(q_emb, k)

    return [docs[i]["text"] for i in idxs[0]]


def chat(role):
    print(f"\n🤖 Gemini Chatbot ({role.upper()})")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        context = retrieve_context(user_input, role)

        prompt = f"""
You are an assistant answering questions using ONLY the context below.
If the answer is not contained, say "I do not have access to that information."

Context:
{chr(10).join(context)}

User question:
{user_input}
"""

        response = llm.generate_content(prompt)
        print("\nAssistant:", response.text, "\n")


role = input("Select role (intern / developer / manager): ").strip().lower()
chat(role)