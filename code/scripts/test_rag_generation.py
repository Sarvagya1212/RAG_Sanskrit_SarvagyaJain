import sys
from pathlib import Path
import yaml
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from code.src.preprocessing.preprocessor import SanskritPreprocessor
from code.src.indexing.bm25_indexer import BM25Indexer
from code.src.indexing.vector_indexer import VectorIndexer
from code.src.indexing.metadata_store import MetadataStore
from code.src.indexing.embedding_generator import EmbeddingGenerator
from code.src.retrieval.hybrid_retriever import HybridRetriever
from code.src.generation.llm_generator import LLMGenerator

def test_rag():
    print("=== Testing Full RAG Pipeline ===")
    
    # 1. Load Config
    with open("code/config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    print("\n[1/5] Loading Configuration...")
    embedding_model = config["models"]["embedding"]["name"]
    llm_path = config["models"]["llm"]["path"]
    print(f"  Embedding: {embedding_model}")
    print(f"  LLM Path: {llm_path}")
    
    # 2. Initialize Retriever Components
    print("\n[2/5] Initializing Retrieval System...")
    preprocessor = SanskritPreprocessor()
    
    bm25 = BM25Indexer()
    bm25.load("data/processed/bm25_index.pkl")
    
    vector_indexer = VectorIndexer()
    vector_indexer.load("data/processed/faiss_index.bin")
    
    embedder = EmbeddingGenerator(model_name=embedding_model)
    embedder.load_model()
    
    metadata_store = MetadataStore()
    
    retriever = HybridRetriever(
        bm25_indexer=bm25,
        vector_indexer=vector_indexer,
        embedding_generator=embedder,
        metadata_store=metadata_store,
        preprocessor=preprocessor
    )
    
    # 3. Initialize Generator
    print("\n[3/5] Initializing LLM Generator...")
    # Override thread count for test if needed
    generator = LLMGenerator(
        model_path=llm_path,
        n_ctx=2048,
        n_threads=4,
        max_tokens=256
    )
    generator.load_model()
    
    # 4. Perform Retrieval
    query = "Who was Shankhanada?"
    print(f"\n[4/5] Retrieving context for: '{query}'")
    
    chunks = retriever.retrieve(query, top_k=3)
    print(f"  Retrieved {len(chunks)} chunks:")
    for i, c in enumerate(chunks, 1):
        print(f"    {i}. {c['story_title']} (score: {c['retrieval_score']:.3f})")
    
    if not chunks:
        print("  ERROR: No chunks retrieved! Cannot proceed to generation.")
        return
    
    # 5. Generate Answer
    print(f"\n[5/5] Generating Answer...")
    result = generator.generate(query, chunks)
    
    print("\n" + "="*50)
    print("GENERATED ANSWER:")
    print("="*50)
    print(result['answer'])
    print("="*50)
    
    print("\nCitations:")
    for source in result['sources']:
        print(f" - {source['story_title']}")
        
    metadata_store.close()

if __name__ == "__main__":
    test_rag()
