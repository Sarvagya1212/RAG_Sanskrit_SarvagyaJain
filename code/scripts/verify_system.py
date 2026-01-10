"""Comprehensive 6-step verification per user's checklist."""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 70)
print("COMPLETE SYSTEM VERIFICATION")
print("=" * 70)

# STEP 1: Check Index Files
print("\n[STEP 1] INDEX FILE VERIFICATION")
print("-" * 70)

files_to_check = [
    "data/processed/faiss_index.bin",
    "data/processed/metadata.db",
    "data/processed/bm25_index.pkl",
    "data/processed/embeddings.npy",
    "data/processed/chunks_fixed.json"
]

for filepath in files_to_check:
    if os.path.exists(filepath):
        size_kb = os.path.getsize(filepath) / 1024
        status = "✅" if size_kb > 0.1 else "❌ EMPTY"
        print(f"  {status} {filepath}: {size_kb:.2f} KB")
    else:
        print(f"  ❌ MISSING: {filepath}")

# STEP 2: Check Chunk Preprocessing
print("\n[STEP 2] CHUNK PREPROCESSING VERIFICATION")
print("-" * 70)

import json
with open("data/processed/chunks_fixed.json", encoding="utf-8") as f:
    data = json.load(f)
    
print(f"  Total chunks: {data['total_chunks']}")
chunk = data['chunks'][0]
print(f"  Sample chunk keys: {list(chunk.keys())}")
print(f"  Original text: {chunk['text'][:80]}...")
print(f"  SLP1 text:     {chunk['slp1_text'][:80]}...")

# Verify SLP1 is actually ASCII
is_ascii = all(ord(c) < 128 for c in chunk['slp1_text'][:100].replace(' ', ''))
print(f"  SLP1 is ASCII: {'✅' if is_ascii else '❌ STILL DEVANAGARI'}")

# STEP 3: Test Retrieval Components
print("\n[STEP 3] RETRIEVAL COMPONENT TESTING")
print("-" * 70)

from code.src.preprocessing import SanskritPreprocessor
from code.src.indexing import BM25Indexer, VectorIndexer, EmbeddingGenerator, MetadataStore

preprocessor = SanskritPreprocessor()

# Test query preprocessing
test_query = "शंखनादः"
result = preprocessor.process(test_query)
print(f"  Query: {test_query}")
print(f"  Script: {result.script}")
print(f"  SLP1: {result.slp1}")

# Load BM25 and test
print("\n  [3a] BM25 Search Test:")
bm25 = BM25Indexer()
bm25.load("data/processed/bm25_index.pkl")
print(f"    Chunks in BM25: {len(bm25.chunks)}")

# Check what's actually in the BM25 index
print(f"    First chunk slp1_text (first 80 chars):")
first_slp1 = bm25.chunks[0].get('slp1_text', 'MISSING')[:80]
print(f"      {first_slp1}")

# Search
bm25_results = bm25.search(result.slp1, top_k=5)
print(f"    BM25 results for '{result.slp1}': {len(bm25_results)}")
for i, (idx, score) in enumerate(bm25_results[:3], 1):
    print(f"      {i}. idx={idx}, score={score:.4f}")

# Load vector and test
print("\n  [3b] Vector Search Test:")
embedder = EmbeddingGenerator()
embedder.load_model()
vector_indexer = VectorIndexer()
vector_indexer.load("data/processed/faiss_index.bin")
print(f"    Vectors in FAISS: {vector_indexer.index.ntotal}")

query_emb = embedder.generate_query_embedding(result.slp1)
vector_results = vector_indexer.search(query_emb, top_k=5)
print(f"    Vector results: {len(vector_results)}")
for i, (idx, dist) in enumerate(vector_results[:3], 1):
    print(f"      {i}. idx={idx}, distance={dist:.4f}")

# Load metadata and test
print("\n  [3c] Metadata Store Test:")
store = MetadataStore()
all_chunks = store.get_all_chunks()
print(f"    Chunks in metadata: {len(all_chunks)}")
if all_chunks:
    print(f"    First chunk ID: {all_chunks[0].get('chunk_id')}")
    print(f"    First chunk story: {all_chunks[0].get('story_title')}")

# STEP 4: Cross-Script Equivalence
print("\n[STEP 4] CROSS-SCRIPT EQUIVALENCE")
print("-" * 70)

test_words = [
    ("Devanagari", "शंखनाद"),
    ("IAST", "śaṅkhanāda"),
    ("Loose Roman", "shankhanaada")
]

slp1_results = []
for script_name, word in test_words:
    res = preprocessor.process(word)
    slp1_results.append(res.slp1)
    print(f"  {script_name:15} '{word}' → '{res.slp1}'")

all_equal = len(set(slp1_results)) == 1
print(f"\n  All equivalent: {'✅' if all_equal else '❌ MISMATCH'}")

# STEP 5: Diagnose Retrieval Issue
print("\n[STEP 5] RETRIEVAL DIAGNOSIS")
print("-" * 70)

# Check if the query SLP1 appears in any chunk
query_slp1 = "SaMKanAda"
matches_found = 0
for i, chunk in enumerate(data['chunks']):
    if query_slp1.lower() in chunk['slp1_text'].lower():
        matches_found += 1
        if matches_found == 1:
            print(f"  Found '{query_slp1}' in chunk {i}:")
            print(f"    {chunk['slp1_text'][:100]}...")

print(f"\n  Chunks containing '{query_slp1}': {matches_found}")

if matches_found > 0:
    print(f"  ✅ Query term exists in chunks")
else:
    print(f"  ❌ Query term NOT FOUND in any chunk!")
    print(f"     Checking what terms ARE in the chunks...")
    sample = data['chunks'][0]['slp1_text'][:200]
    print(f"     Sample: {sample}")

# STEP 6: Summary
print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

print(f"""
INDEX FILES:
  - FAISS vectors: {vector_indexer.index.ntotal}
  - Metadata chunks: {len(all_chunks)}
  - BM25 chunks: {len(bm25.chunks)}
  - Match: {'✅' if len(all_chunks) == vector_indexer.index.ntotal == len(bm25.chunks) else '❌'}

PREPROCESSING:
  - Chunks have SLP1: {'✅' if is_ascii else '❌'}
  - Cross-script works: {'✅' if all_equal else '❌'}

RETRIEVAL:
  - BM25 returns results: {'✅' if bm25_results[0][1] > 0 else '❌ (scores are 0)'}
  - Vector returns results: {'✅' if len(vector_results) > 0 else '❌'}
  - Query term in chunks: {'✅' if matches_found > 0 else '❌'}
""")

store.close()
print("Verification complete.")
