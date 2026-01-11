# 🔐 Permission-Aware Vector Database with Privacy-Conditional Embeddings

## Overview

This repository implements a **Permission-Aware Vector Database** that enforces access control at the *semantic representation level* rather than through traditional metadata filtering or prompt-based restrictions. The system is designed to support **secure Retrieval-Augmented Generation (RAG)** with modern Large Language Models (LLMs) such as Gemini, ensuring that sensitive information is never retrieved or exposed to unauthorized users.

The core novelty of this work lies in **privacy-conditional embeddings**, where both documents and queries are embedded using transformations that depend on predefined permission levels (e.g., intern, developer, manager). Retrieval is restricted to permission-bounded vector indices, guaranteeing that the LLM only receives context the user is authorized to access.

---

## Research Motivation and Novelty

### Problem with Existing Vector Databases

Conventional vector databases used in RAG pipelines embed all documents into a single semantic space and apply access control *after* retrieval via metadata filtering or prompt constraints. This approach suffers from several limitations:

* Semantic leakage due to similarity overlap in high-dimensional spaces
* Vulnerability to prompt-injection attacks
* Implicit trust placed in LLMs to obey access rules
* Access control implemented outside the embedding and retrieval process

### Proposed Research Contribution

This project is based on the following research ideas:

* **Privacy-Conditional Embedding**: Embedding representations are conditioned on document and user permission levels, encoding access constraints directly into the vector representation.
* **Hierarchical Semantic Projection**: Higher-permission embeddings retain richer semantic information, while lower-permission embeddings expose only restricted semantic subspaces.
* **Permission-Bounded Retrieval**: Retrieval is performed within role-specific vector indices, preventing unauthorized vectors from ever being considered during similarity search.
* **LLM-Agnostic Security**: The LLM is treated as an untrusted component and is supplied only with pre-authorized contextual data.

This research demonstrates that access control can be enforced *before* similarity computation and LLM invocation, significantly reducing privacy risks in AI-driven systems.

---

## System Architecture

The system consists of three primary layers:

1. **Representation Layer** – Generates privacy-conditioned embeddings for documents and queries.
2. **Retrieval Layer** – Maintains permission-bounded vector indices and performs secure semantic search.
3. **Generation Layer** – Uses a pretrained LLM (e.g., Gemini-Flash) to generate responses strictly from authorized context.

Access control is enforced at the representation and retrieval layers, independent of LLM behavior.

---

## Project Structure

```
project/
│
├── privacy_embedding.py   # Privacy-conditional embedding logic
├── build_index.py         # Document ingestion and vector index construction
├── chatbot.py             # Role-based chatbot using Gemini LLM
├── docs/                  # Text documents with role assignments
│   ├── intern_doc.txt
│   ├── developer_doc.txt
│   └── manager_doc.txt
├── .env                   # API keys (not committed)
└── README.md
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd project
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install sentence-transformers faiss-cpu google-generativeai python-dotenv
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

> **Important:** Never commit the `.env` file to version control.

---

## Running the System

### Step 1: Prepare Documents

Place your text documents inside the `docs/` folder. Each document should specify its role at the top:

```
ROLE: intern
ID: DOC1

<document content>
```

### Step 2: Build the Permission-Aware Vector Database

```bash
python build_index.py
```

This step embeds documents using privacy-conditional embeddings and constructs role-specific vector indices.

### Step 3: Start the Chatbot

```bash
python chatbot.py
```

Select a role (`intern`, `developer`, or `manager`) and start chatting.

---

## Example Demo Behavior

| User Role | Query                           | Expected Behavior                     |
| --------- | ------------------------------- | ------------------------------------- |
| Intern    | "What is the company budget?"   | Access denied or insufficient context |
| Developer | "How does deployment work?"     | Technical workflow explanation        |
| Manager   | "What are the financial risks?" | Strategic and financial response      |

---

## Security Properties

* Unauthorized documents are never retrieved
* LLM never receives restricted context
* Prompt-injection attacks cannot bypass access control
* Access enforcement is independent of LLM compliance

---

## Limitations and Future Work

* Perfect semantic separation is mathematically impossible in continuous embedding spaces
* Projection matrices are currently static and not learned
* Future work includes training privacy-aware embedding heads and integrating encrypted vector search

---

## Citation

If you use or build upon this work, please cite it as:

> "Permission-Aware Vector Databases Using Privacy-Conditional Embeddings for Secure LLM Interaction"

---

## License

This project is intended for academic and research use only. Patent filing may be pending.
