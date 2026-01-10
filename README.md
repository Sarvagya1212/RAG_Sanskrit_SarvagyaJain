# Sanskrit RAG System

Production-ready Retrieval-Augmented Generation system for Sanskrit texts using CPU-only inference.

## System Overview

This system processes Sanskrit moral stories (Panchatantra-style) and enables intelligent question-answering through:
- Multi-script support (Devanagari, IAST, loose Roman)
- Hybrid retrieval (BM25 + Vector search)
- LLM-powered answer generation (Qwen 2.5 3B)

## Requirements

- Python 3.9+
- 8GB RAM minimum
- 10GB disk space (for models)
- CPU with 4+ cores recommended

## Installation

```bash
# Clone repository
git clone 
cd RAG_Sanskrit_YourName

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download LLM model (see Installation Guide in docs)
```

## Quick Start

```bash
# Index documents
python code/main.py --mode index --data ./data/raw

# Query system
python code/main.py --mode query

# Interactive mode
python code/main.py --interactive
```

## Project Structure

```
RAG_Sanskrit_YourName/
├── code/
│   ├── src/              # Source modules
│   ├── config/           # Configuration files
│   ├── tests/            # Unit tests
│   └── main.py           # Entry point
├── data/                 # Data directories
├── models/               # LLM and embeddings
├── report/               # Technical report
└── logs/                 # System logs
```

## Development Status

- [x] Day 1: Environment setup ✓
- [ ] Day 2: Data loading & story segmentation
- [ ] Day 3: Preprocessing pipeline
- [ ] Day 4: Chunking system
- [ ] Day 5: Vector indexing

## Documentation

See `report/` directory for detailed technical documentation.

## License

MIT License