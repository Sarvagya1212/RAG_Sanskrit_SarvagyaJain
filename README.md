# Sanskrit RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for Sanskrit texts, designed for **CPU-only inference**. This system ingests Sanskrit moral stories, indexes them using a hybrid strategy (BM25 + Semantic Vectors), and generates context-aware answers using a local Large Language Model.

## 🚀 Unique Features

*   **Cross-Script Support**: Seamlessly handles **Devanagari** (संस्कृत), **IAST** (saṃskṛta), and **Loose Roman** (sanskrit) inputs.
*   **Hierarchical Chunking**: Parent-child strategy (600-800 token parents, 150-200 token children) for optimal retrieval precision + context richness.
*   **Hybrid Retrieval**: Combines Lexical (BM25) precision with Semantic (embedding) understanding.
*   **CPU Optimized**: Efficient inference using `llama.cpp` and `quantized` models.
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

    Create the models directory structure:
    ```bash
    mkdir -p models/llm
    ```

    **LLM Model** (Required):
    - Download: [Qwen2.5-3B-Instruct-Q5_K_M.gguf](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q5_k_m.gguf)
    - Save to: `models/llm/Qwen2.5-3B-Instruct-Q5_K_M.gguf`
    - Size: ~2GB

    **Embedding Model** (Auto-downloaded):
    - Model: `intfloat/multilingual-e5-small` (384 dimensions)
    - Downloaded automatically by `sentence-transformers` on first run

5.  **Verify Installation**
    ```bash
    python code/main.py --help
    ```

---

## 🏃 Usage

### Quick Start

**1. Index your documents (first time):**
```bash
python code/main.py --mode index --data ./data/raw
```

**2. Query the system:**
```bash
# Devanagari
python code/main.py --mode query --query "कालिदासः कः आसीत्?" --top-k 2

# IAST or Roman
python code/main.py --mode query --query "kalidasa kaun tha?" --top-k 2
```

**3. Interactive mode:**
```bash
python code/main.py --interactive
```

### CLI Options

```bash
python code/main.py [OPTIONS]

Options:
  --mode {index,query}    Operation mode
  --query TEXT            Query text (any script)
  --data PATH             Data directory (default: ./data/raw)
  --top-k INT             Number of results (default: 2)
  --interactive           Start interactive REPL
  --config PATH           Config file (default: code/config/config.yaml)
```

### Examples

```bash
# Index with custom path
python code/main.py --mode index --data /my/texts

# Query with more results  
python code/main.py --mode query --query "राजा" --top-k 5

# Interactive session
python code/main.py --interactive
Query> कालिदासः कः आसीत्?
[results...]
Query> exit
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
    └── Technical_Report.pdf  # Detailed System Report
```

## 📄 Documentation

For a deep dive into the architecture, retrieval strategy, and performance metrics, please read the **[Technical Report](report/Technical_Report.pdf)**.

## ⚖️ License
MIT License