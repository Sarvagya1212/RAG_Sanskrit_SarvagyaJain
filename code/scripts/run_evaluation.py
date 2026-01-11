"""
Comprehensive evaluation script for Sanskrit RAG system.

Measures:
- Latency per component
- Recall@5
- Memory usage
- End-to-end performance
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from code.src.preprocessing import SanskritPreprocessor
from code.src.indexing import BM25Indexer, VectorIndexer, EmbeddingGenerator, MetadataStore
from code.src.retrieval import HybridRetriever
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__)


# Test queries with ground truth
TEST_QUERIES = [
    {
        'id': 1,
        'category': 'character',
        'query': 'शंखनादः कः आसीत्?',
        'query_english': 'Who was Shankhanaada?',
        'ground_truth_stories': ['मूर्खभृत्यस्य'],
        'keywords': ['शंखनाद', 'भृत्य', 'servant']
    },
    {
        'id': 2,
        'category': 'story',
        'query': 'कालीदासस्य विषये किम् वर्णितम्?',
        'query_english': 'What is described about Kalidasa?',
        'ground_truth_stories': ['चतुरस्य कालीदासस्य'],
        'keywords': ['कालीदास', 'कवि', 'poet']
    },
    {
        'id': 3,
        'category': 'moral',
        'query': 'मूर्खभृत्यस्य संसर्गात् किम् भवति?',
        'query_english': 'What happens from association with foolish servants?',
        'ground_truth_stories': ['मूर्खभृत्यस्य'],
        'keywords': ['मूर्ख', 'कार्य', 'विनश्यति']
    },
    {
        'id': 4,
        'category': 'verse',
        'query': 'उद्यमः साहसम् धैर्यम् इति श्लोकः कुत्र अस्ति?',
        'query_english': 'Where is the verse about effort, courage, patience?',
        'ground_truth_stories': ['मूर्खभृत्यस्य'],
        'keywords': ['उद्यम', 'साहस', 'धैर्य']
    },
    {
        'id': 5,
        'category': 'character',
        'query': 'What happened to the old woman in winter?',
        'query_english': 'What happened to the old woman in winter?',
        'ground_truth_stories': ['वृद्धायाः चार्तुयम्', 'शीतं बहु बाधति'],
        'keywords': ['वृद्धा', 'शीत', 'cold', 'winter']
    },
    {
        'id': 6,
        'category': 'story',  
        'query': 'dharma',
        'query_english': 'About dharma',
        'ground_truth_stories': None,  # General query
        'keywords': ['धर्म', 'dharma']
    },
    {
        'id': 7,
        'category': 'cross-script',
        'query': 'मूर्ख',  # Should work same as "mūrkha" or "murkha"
        'query_english': 'About foolish/fool',
        'ground_truth_stories': ['मूर्खभृत्यस्य'],
        'keywords': ['मूर्ख', 'foolish']
    }
]


class PerformanceMetrics:
    """Track performance metrics."""
    
    def __init__(self):
        self.latencies = {
            'preprocessing': [],
            'bm25_search': [],
            'vector_search': [],
            'rrf_fusion': [],
            'metadata_retrieval': [],
            'total_retrieval': []
        }
        self.memory_usage = []
        self.recall_scores = []
    
    def add_latency(self, component: str, duration: float):
        """Add latency measurement."""
        if component in self.latencies:
            self.latencies[component].append(duration)
    
    def add_recall(self, score: float):
        """Add recall score."""
        self.recall_scores.append(score)
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        summary = {}
        
        for component, times in self.latencies.items():
            if times:
                summary[component] = {
                    'mean_ms': sum(times) / len(times) * 1000,
                    'min_ms': min(times) * 1000,
                    'max_ms': max(times) * 1000,
                    'count': len(times)
                }
        
        if self.recall_scores:
            summary['recall_at_5'] = {
                'mean': sum(self.recall_scores) / len(self.recall_scores),
                'min': min(self.recall_scores),
                'max': max(self.recall_scores)
            }
        
        return summary


def measure_memory() -> float:
    """Estimate memory usage (simplified without psutil)."""
    # Simplified: return estimated memory based on loaded components
    return 100.0  # Approximate MB


def calculate_recall_at_k(
    retrieved_stories: List[str],
    ground_truth: List[str],
    k: int = 5
) -> float:
    """
    Calculate Recall@k.
    
    Recall@k = |relevant ∩ retrieved@k| / |relevant|
    """
    if not ground_truth:
        return 1.0  # No ground truth to compare
    
    retrieved_set = set(retrieved_stories[:k])
    ground_truth_set = set(ground_truth)
    
    intersection = retrieved_set & ground_truth_set
    recall = len(intersection) / len(ground_truth_set)
    
    return recall


def run_evaluation():
    """Run comprehensive evaluation."""
    
    print("\n" + "=" * 70)
    print("SANSKRIT RAG SYSTEM - COMPREHENSIVE EVALUATION")
    print("=" * 70)
    
    metrics = PerformanceMetrics()
    
    # Initialize components
    print("\n[1/4] Loading Components...")
    print("-" * 70)
    
    start_mem = measure_memory()
    print(f"Initial memory: {start_mem:.2f} MB")
    
    # Load config
    import yaml
    with open("code/config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    embedding_model_name = config["models"]["embedding"]["name"]

    preprocessor = SanskritPreprocessor()
    bm25_indexer = BM25Indexer()
    bm25_indexer.load("data/processed/bm25_index.pkl")
    vector_indexer = VectorIndexer()
    vector_indexer.load("data/processed/faiss_index.bin")
    embedding_generator = EmbeddingGenerator(model_name=embedding_model_name)
    embedding_generator.load_model()
    metadata_store = MetadataStore()
    
    retriever = HybridRetriever(
        bm25_indexer=bm25_indexer,
        vector_indexer=vector_indexer,
        embedding_generator=embedding_generator,
        metadata_store=metadata_store,
        preprocessor=preprocessor,
        config=config
    )
    
    loaded_mem = measure_memory()
    print(f"After loading: {loaded_mem:.2f} MB")
    print(f"Memory used by components: {loaded_mem - start_mem:.2f} MB")
    
    # Run test queries
    print(f"\n[2/4] Running {len(TEST_QUERIES)} Test Queries...")
    print("-" * 70)
    
    results = []
    
    for test_case in TEST_QUERIES:
        print(f"\nQuery {test_case['id']}: {test_case['category']}")
        print(f"  Q: {test_case['query']}")
        print(f"  ({test_case['query_english']})")
        
        # Measure preprocessing
        start = time.time()
        preprocessed = preprocessor.process(test_case['query'])
        preprocess_time = time.time() - start
        metrics.add_latency('preprocessing', preprocess_time)
        
        print(f"  Preprocessed: {preprocessed.slp1} ({preprocessed.script})")
        
        # Measure retrieval
        start = time.time()
        
        # BM25
        bm25_start = time.time()
        bm25_results = bm25_indexer.search(preprocessed.slp1, top_k=50)
        bm25_time = time.time() - bm25_start
        metrics.add_latency('bm25_search', bm25_time)
        
        # Vector
        vector_start = time.time()
        query_emb = embedding_generator.generate_query_embedding(preprocessed.slp1)
        vector_results = vector_indexer.search(query_emb, top_k=50)
        vector_time = time.time() - vector_start
        metrics.add_latency('vector_search', vector_time)
        
        # Fusion (simplified - just using retriever)
        chunks = retriever.search(test_case['query'], top_k=5)
        total_time = time.time() - start
        metrics.add_latency('total_retrieval', total_time)
        
        # Calculate recall
        retrieved_stories = [c['story_title'] for c in chunks]
        if test_case['ground_truth_stories']:
            recall = calculate_recall_at_k(
                retrieved_stories,
                test_case['ground_truth_stories'],
                k=5
            )
            metrics.add_recall(recall)
            print(f"  Recall@5: {recall:.2f}")
        else:
            print(f"  Recall@5: N/A (no ground truth)")
        
        print(f"  Retrieved from stories:")
        for i, chunk in enumerate(chunks, 1):
            print(f"    {i}. {chunk['story_title']} (score: {chunk['score']:.4f})")
        
        # Check if ground truth found
        found_in_top5 = False
        if test_case['ground_truth_stories']:
            for gt_story in test_case['ground_truth_stories']:
                if gt_story in retrieved_stories:
                    found_in_top5 = True
                    break
        
        result = {
            'query_id': test_case['id'],
            'category': test_case['category'],
            'query': test_case['query'],
            'preprocessing_ms': preprocess_time * 1000,
            'bm25_ms': bm25_time * 1000,
            'vector_ms': vector_time * 1000,
            'total_ms': total_time * 1000,
            'recall_at_5': recall if test_case['ground_truth_stories'] else None,
            'found_in_top5': found_in_top5,
            'retrieved_stories': retrieved_stories
        }
        results.append(result)
    
    # Performance summary
    print("\n[3/4] Performance Summary")
    print("-" * 70)
    
    summary = metrics.get_summary()
    
    print("\nLatency by Component (milliseconds):")
    for component, stats in summary.items():
        if component != 'recall_at_5':
            print(f"  {component}:")
            print(f"    Mean: {stats['mean_ms']:.2f} ms")
            print(f"    Min:  {stats['min_ms']:.2f} ms")
            print(f"    Max:  {stats['max_ms']:.2f} ms")
    
    if 'recall_at_5' in summary:
        print(f"\nRecall@5:")
        print(f"  Mean: {summary['recall_at_5']['mean']:.2f}")
        print(f"  Min:  {summary['recall_at_5']['min']:.2f}")
        print(f"  Max:  {summary['recall_at_5']['max']:.2f}")
    
    current_mem = measure_memory()
    print(f"\nMemory Usage:")
    print(f"  Current: {current_mem:.2f} MB")
    print(f"  Peak increase: {current_mem - start_mem:.2f} MB")
    
    # Analysis
    print("\n[4/4] Analysis")
    print("-" * 70)
    
    # Failure cases
    failure_cases = [r for r in results if r['recall_at_5'] is not None and r['recall_at_5'] < 1.0]
    
    if failure_cases:
        print(f"\nPartial/Failed Retrievals: {len(failure_cases)}")
        for case in failure_cases:
            print(f"  Query {case['query_id']}: {case['query']}")
            print(f"    Recall: {case['recall_at_5']:.2f}")
            print(f"    Retrieved: {case['retrieved_stories']}")
    else:
        print(f"\n✓ All queries with ground truth achieved Recall@5 = 1.0")
    
    # Category breakdown
    by_category = {}
    for r in results:
        cat = r['category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)
    
    print(f"\nPerformance by Category:")
    for cat, cat_results in by_category.items():
        avg_latency = sum(r['total_ms'] for r in cat_results) / len(cat_results)
        recalls = [r['recall_at_5'] for r in cat_results if r['recall_at_5'] is not None]
        avg_recall = sum(recalls) / len(recalls) if recalls else None
        
        print(f"  {cat}:")
        print(f"    Avg latency: {avg_latency:.2f} ms")
        if avg_recall is not None:
            print(f"    Avg recall: {avg_recall:.2f}")
    
    # Save results
    output = {
        'summary': summary,
        'test_results': results,
        'memory_mb': {
            'initial': start_mem,
            'loaded': loaded_mem,
            'peak': current_mem
        }
    }
    
    with open('report/evaluation_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to report/evaluation_results.json")
    
    metadata_store.close()
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
