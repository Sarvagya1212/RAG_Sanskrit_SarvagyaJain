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
from code.src.utils.logger import setup_logger

def verify_cross_script():
    print("=== Cross-Script Retrieval Consistency Test ===\n")
    
    # 1. Load System
    with open("code/config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    preprocessor = SanskritPreprocessor()
    bm25 = BM25Indexer()
    bm25.load("data/processed/bm25_index.pkl")
    vector = VectorIndexer()
    vector.load("data/processed/faiss_index.bin")
    embedder = EmbeddingGenerator(model_name=config["models"]["embedding"]["name"])
    embedder.load_model()
    metadata = MetadataStore()
    
    retriever = HybridRetriever(bm25, vector, embedder, metadata, preprocessor)
    
    # 2. Define Test Cases
    queries = [
        ("Devanagari", "शंखनादः कः आसीत्?"),
        ("IAST", "śaṅkhanādaḥ kaḥ āsīt?"),
        ("Loose Roman", "shankhanaadah kah aseet?")
    ]
    
    results_data = []
    
    # 3. Run Tests
    for script_name, query_text in queries:
        print(f"Testing {script_name}: {query_text}")
        
        # Get SLP1
        prep = preprocessor.process(query_text)
        slp1 = prep.slp1
        
        # Retrieve
        results = retriever.retrieve(query_text, top_k=1)
        
        if results:
            top_result = results[0]
            chunk_id = top_result['chunk_id']
            score = top_result.get('retrieval_score', 0.0)
        else:
            chunk_id = "NO_RESULT"
            score = 0.0
            
        results_data.append({
            "Script": script_name,
            "Query": query_text,
            "Preprocessed Form": slp1,
            "Top Chunk ID": chunk_id,
            "Score": f"{score:.2f}"
        })
        print(f"  -> SLP1: {slp1}")
        print(f"  -> Top: {chunk_id} (Score: {score:.2f})\n")

    # 4. Generate Markdown Table
    print("\n### Cross-Script Retrieval Consistency Test\n")
    print("| Script | Query | Preprocessed Form | Top Chunk ID | Score |")
    print("|--------|-------|-------------------|--------------|-------|")
    for r in results_data:
        print(f"| {r['Script']} | {r['Query']} | {r['Preprocessed Form']} | {r['Top Chunk ID']} | {r['Score']} |")
    
    # 5. Ad-hoc Tests
    additional_queries = [
        ("Event Retrieval", "शर्करा कुत्र स्त्रवत्?"),
        ("Verse Retrieval", "उद्यमः साहसम् धैर्यम्")
    ]
    
    print("\n\n### Functional Tests\n")
    for name, query in additional_queries:
        print(f"Testing {name}: {query}")
        results = retriever.retrieve(query, top_k=1)
        if results:
            top = results[0]
            print(f"  -> Found: {top['story_title']}")
            print(f"  -> Text snippet: {top['text_original'][:60]}...")
        else:
            print("  -> No results found")
        print()

    metadata.close()

if __name__ == "__main__":
    verify_cross_script()
