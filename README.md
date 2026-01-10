# Sanskrit RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for Sanskrit texts, designed for **CPU-only inference**. This system ingests Sanskrit moral stories, indexes them using a hybrid strategy (BM25 + Semantic Vectors), and generates context-aware answers using a local Large Language Model.

## 🚀 Unique Features

*   **Cross-Script Support**: Seamlessly handles **Devanagari** (संस्कृत), **IAST** (saṃskṛta), and **Loose Roman** (sanskrit) inputs.
*   **Hybrid Retrieval**: Combines Lexical (BM25) precision with Semantic (embedding) understanding.
*   **CPU Optimized**: efficient inference using `llama.cpp` and `quantized` models.
*   **Citation Aware**: Every answer cites the source story title.

---

## 🛠️ Installation

### Prerequisites
*   Python 3.10+
*   Allowed: 8GB+ RAM
*   OS: Windows, Linux, or MacOS

### Setup
1.  **Clone the repository**
    ```bash
    git clone <repository_url>
    cd RAG_Sanskrit_SarvagyaJain
    ```

2.  **Create Virtual Environment**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Models**
    The system requires two models. Place them in the `models/` directory:
    
    *   **LLM**: [Qwen2.5-3B-Instruct-Q5_K_M.gguf](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF)
        *   Save to: `models/llm/Qwen2.5-3B-Instruct-Q5_K_M.gguf`
    *   **Embedding**: [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) 
        *   *Note: This is downloaded automatically by `sentence-transformers` on first run.*

---

## 🏃 Usage

The system exposes a unified CLI via `code/main.py`.

### 1. Indexing (First Time Setup)
Ingest and index the raw Sanskrit stories:
```bash
python code/main.py --mode index --data ./data/raw
```

### 2. Interactive Query Mode (Recommended)
Start the chatbot interface:
```bash
python code/main.py --interactive
```
*   *Type your question in English or Sanskrit (e.g., "Who was Shankhanada?" or "शंखनादः कः आसीत्?")*

### 3. Quick Query
Run a single query from command line:
```bash
python code/main.py --mode query --query "कालीदासस्य विषये किम् वर्णितम्?"
```

---

## 📂 Project Structure

```
RAG_Sanskrit_SarvagyaJain/
├── code/
│   ├── main.py               # Main CLI Entry Point
│   ├── config/               # System configuration (YAML)
│   ├── src/                  # Source Code
│   │   ├── ingestion/        # Document loading & segmentation
│   │   ├── preprocessing/    # Script normalization (SLP1)
│   │   ├── indexing/         # BM25 & Vector Indexing
│   │   ├── retrieval/        # Hybrid Retrieval Logic
│   │   └── generation/       # LLM Integration (Qwen)
│   └── scripts/              # Utility scripts (eval, debug)
├── data/
│   ├── raw/                  # Original stories.txt
│   └── processed/            # Indexed artifacts (FAISS, BM25)
├── models/                   # Local GGUF models
└── report/                   # Documentation
    └── Technical_Report.md   # Detailed System Report
```

## 📄 Documentation

For a deep dive into the architecture, retrieval strategy, and performance metrics, please read the **[Technical Report](report/Technical_Report.md)**.

## ⚖️ License
MIT License