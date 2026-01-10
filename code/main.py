"""
Sanskrit RAG System - Main CLI Interface

Complete command-line interface with interactive query mode,
indexing mode, and beautiful result formatting.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
import json
from datetime import datetime
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.src.ingestion.pipeline import IngestionPipeline
from code.src.preprocessing.preprocessor import SanskritPreprocessor
from code.src.chunking.chunker import SanskritChunker  # Updated import path
from code.src.indexing.indexing_pipeline import IndexingPipeline
from code.src.indexing.bm25_indexer import BM25Indexer
from code.src.indexing.embedding_generator import EmbeddingGenerator
from code.src.indexing.vector_indexer import VectorIndexer
from code.src.indexing.metadata_store import MetadataStore
from code.src.retrieval.hybrid_retriever import HybridRetriever
from code.src.generation.llm_generator import LLMGenerator
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__, log_file="logs/main.log")


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class SanskritRAGCLI:
    """Main CLI interface for Sanskrit RAG system."""
    
    def __init__(self, config_path: str = "code/config/config.yaml"):
        """
        Initialize CLI interface.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.preprocessor = SanskritPreprocessor()
        
        # Initialize retrieval components (loaded lazily)
        self.bm25_indexer = None
        self.vector_indexer = None
        self.embedding_generator = None
        self.metadata_store = None
        self.retriever = None
        self.llm_generator = None
        
        self.is_initialized = False
    
    def _load_config(self, path: str) -> Dict:
        """Load configuration from YAML."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            print(f"{Colors.RED}Error loading config: {e}{Colors.END}")
            sys.exit(1)

    def print_banner(self):
        """Print welcome banner."""
        banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║          Sanskrit RAG System v1.0                         ║
║          Cross-Script Retrieval System                    ║
║                                                           ║
║          Supports: Devanagari | IAST | Loose Roman       ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
{Colors.END}
"""
        print(banner)
    
    def load_indexes(self):
        """Load all indexes and prepare for querying."""
        if self.is_initialized:
            return
        
        print(f"{Colors.YELLOW}Loading indexes...{Colors.END}")
        
        try:
            # Load BM25 index
            print("  → Loading BM25 index...", end=" ")
            self.bm25_indexer = BM25Indexer()
            self.bm25_indexer.load("data/processed/bm25_index.pkl")
            print(f"{Colors.GREEN}✓{Colors.END}")
            
            # Load vector index
            print("  → Loading FAISS index...", end=" ")
            self.vector_indexer = VectorIndexer()
            self.vector_indexer.load("data/processed/faiss_index.bin")
            print(f"{Colors.GREEN}✓{Colors.END}")
            
            # Load embedding model
            print("  → Loading embedding model...", end=" ")
            model_name = self.config['models']['embedding']['name']
            self.embedding_generator = EmbeddingGenerator(model_name=model_name)
            self.embedding_generator.load_model()
            print(f"{Colors.GREEN}✓{Colors.END}")
            
            # Connect to metadata store
            print("  → Connecting to metadata store...", end=" ")
            self.metadata_store = MetadataStore()
            print(f"{Colors.GREEN}✓{Colors.END}")
            
            # Initialize Hybrid Retriever
            print("  → Initializing Hybrid Retriever...", end=" ")
            self.retriever = HybridRetriever(
                bm25_indexer=self.bm25_indexer,
                vector_indexer=self.vector_indexer,
                embedding_generator=self.embedding_generator,
                metadata_store=self.metadata_store,
                preprocessor=self.preprocessor
            )
            print(f"{Colors.GREEN}✓{Colors.END}")
            
            # Initialize LLM Generator
            print("  → Initializing LLM Generator...", end=" ")
            try:
                llm_config = self.config['models']['llm']
                self.llm_generator = LLMGenerator(
                    model_path=llm_config['path'],
                    n_ctx=llm_config.get('n_ctx', 4096),
                    n_threads=llm_config.get('n_threads', 4),
                    temperature=llm_config.get('temperature', 0.7),
                    max_tokens=llm_config.get('max_tokens', 512)
                )
                print(f"{Colors.GREEN}✓{Colors.END}")
            except Exception as e:
                print(f"{Colors.RED}✗ (Skipping LLM: {e}){Colors.END}")
                self.llm_generator = None
            
            self.is_initialized = True
            print(f"\n{Colors.GREEN}All indexes loaded successfully!{Colors.END}\n")
            
        except FileNotFoundError as e:
            print(f"\n{Colors.RED}✗ Error: Index files not found!{Colors.END}")
            print(f"{Colors.YELLOW}Please run indexing first:{Colors.END}")
            print(f"  python code/main.py --mode index --data ./data/raw\n")
            sys.exit(1)
        except Exception as e:
            print(f"\n{Colors.RED}✗ Error loading indexes: {e}{Colors.END}")
            logger.error(f"Failed to load indexes: {e}", exc_info=True)
            sys.exit(1)
    
    def detect_input_script(self, text: str) -> str:
        """
        Detect and display input script type.
        
        Args:
            text: Input text
            
        Returns:
            Script name (devanagari/iast/loose_roman)
        """
        result = self.preprocessor.process(text)
        script = result.script
        
        script_colors = {
            'devanagari': Colors.CYAN,
            'iast': Colors.BLUE,
            'loose_roman': Colors.YELLOW
        }
        
        color = script_colors.get(script, Colors.END)
        print(f"{color}Detected script: {script.upper()}{Colors.END}")
        
        # Display SLP1 conversion for the query
        print(f"  → SLP1: {Colors.CYAN}{result.slp1}{Colors.END}")
        
        return script
    
    def search(self, query: str, top_k: int = 2) -> List[Dict]:
        """
        Perform hybrid search using HybridRetriever.
        
        Args:
            query: Query text (any script)
            top_k: Number of results to return
            
        Returns:
            List of retrieved chunks with scores
        """
        print(f"\n{Colors.YELLOW}Performing hybrid retrieval...{Colors.END}")
        
        # Use the robust HybridRetriever which handles:
        # 1. Preprocessing
        # 2. BM25 Search
        # 3. Vector Embedding generation (with 'query:' prefix)
        # 4. Vector Search
        # 5. RRF Fusion
        # 6. Metadata lookup (safe int casting)
        
        results = self.retriever.retrieve(query, top_k=top_k)
        
        print(f"  → Found {len(results)} relevant results\n")
        
        return results
    
    def display_results(self, results: List[Dict], query: str):
        """
        Display search results with beautiful formatting.
        
        Args:
            results: List of chunks with metadata
            query: Original query text
        """
        print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.GREEN}SEARCH RESULTS{Colors.END}")
        print(f"{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.END}\n")
        
        print(f"{Colors.BOLD}Query:{Colors.END} {Colors.CYAN}{query}{Colors.END}")
        print(f"{Colors.BOLD}Found:{Colors.END} {len(results)} relevant chunks\n")
        
        for i, chunk in enumerate(results, 1):
            self._display_single_result(i, chunk)
        
        print(f"{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.END}\n")
    
    def _display_single_result(self, rank: int, chunk: Dict):
        """
        Display a single search result.
        
        Args:
            rank: Result rank (1, 2, 3, ...)
            chunk: Chunk dictionary with metadata
        """
        # Header
        print(f"{Colors.BOLD}{Colors.BLUE}Result #{rank}{Colors.END}")
        print(f"{Colors.BOLD}├─{Colors.END} Story: {Colors.CYAN}{chunk.get('story_title', 'Unknown')}{Colors.END}")
        print(f"{Colors.BOLD}├─{Colors.END} Chunk ID: {chunk.get('chunk_id', 'Unknown')}")
        print(f"{Colors.BOLD}├─{Colors.END} Type: {chunk.get('content_type', 'Unknown')}")
        score = chunk.get('retrieval_score', chunk.get('rrf_score', 0))
        print(f"{Colors.BOLD}├─{Colors.END} Score: {score:.4f}")
        
        # Text content
        text = chunk.get('text_original', chunk.get('text_slp1', ''))
        
        # Truncate if too long
        max_length = 300
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        print(f"{Colors.BOLD}└─{Colors.END} Text:")
        print(f"   {Colors.YELLOW}{text}{Colors.END}")
        
    def generate_answer(self, query: str, context_chunks: List[Dict]):
        """
        Generate answer using LLM.
        
        Args:
            query: User query
            context_chunks: Retrieved context
        """
        if not self.llm_generator:
            print(f"\n{Colors.YELLOW}LLM not initialized. Skipping generation.{Colors.END}")
            return

        print(f"\n{Colors.YELLOW}Generating answer...{Colors.END}")
        
        try:
            result = self.llm_generator.generate(
                query=query, 
                context_chunks=context_chunks,
                include_citations=True
            )
            
            answer = result['answer']
            sources = result['sources']
            
            print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.END}")
            print(f"{Colors.BOLD}{Colors.GREEN}GENERATED ANSWER{Colors.END}")
            print(f"{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.END}\n")
            
            print(f"{Colors.BOLD}{answer}{Colors.END}\n")
            
            if sources:
                print(f"{Colors.BOLD}Sources:{Colors.END}")
                for source in sources:
                    print(f"- {source['story_title']}")
            
            print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.END}\n")
            
        except Exception as e:
            print(f"\n{Colors.RED}Error generating answer: {e}{Colors.END}\n")
            logger.error(f"Generation error: {e}", exc_info=True)

    def interactive_mode(self):
        """Run interactive query mode."""
        self.print_banner()
        self.load_indexes()
        
        print(f"{Colors.BOLD}Interactive Query Mode{Colors.END}")
        print(f"Enter your queries in any script (Devanagari, IAST, or loose Roman)")
        print(f"Type {Colors.RED}'quit'{Colors.END} or {Colors.RED}'exit'{Colors.END} to stop\n")
        
        while True:
            # Get user input
            try:
                query = input(f"{Colors.BOLD}{Colors.GREEN}Query >{Colors.END} ").strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n\n{Colors.YELLOW}Exiting...{Colors.END}")
                break
            
            # Check for exit commands
            if query.lower() in ['quit', 'exit', 'q']:
                print(f"\n{Colors.YELLOW}Goodbye!{Colors.END}\n")
                break
            
            # Skip empty queries
            if not query:
                continue
            
            # Detect script and display SLP1
            self.detect_input_script(query)
            
            # Search
            try:
                results = self.search(query, top_k=5)
                
                # Display results
                if results:
                    self.display_results(results, query)
                    # Generate Answer
                    self.generate_answer(query, results)
                else:
                    print(f"\n{Colors.RED}No results found.{Colors.END}\n")
            
            except Exception as e:
                print(f"\n{Colors.RED}Error during search: {e}{Colors.END}\n")
                logger.error(f"Search error: {e}", exc_info=True)
    
    def index_mode(self, data_path: str):
        """
        Run complete indexing pipeline.
        
        Args:
            data_path: Path to data directory or file
        """
        self.print_banner()
        
        print(f"{Colors.BOLD}Indexing Mode{Colors.END}")
        print(f"Data source: {data_path}\n")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Ingestion
            print(f"{Colors.BOLD}{Colors.BLUE}[1/3] Document Ingestion{Colors.END}")
            print(f"{Colors.YELLOW}{'─'*60}{Colors.END}")
            
            ingestion_pipeline = IngestionPipeline(output_dir="data/processed")
            
            if Path(data_path).is_file():
                stories = ingestion_pipeline.process_file(data_path)
            else:
                stories = ingestion_pipeline.process_directory(data_path)
            
            ingestion_pipeline.save_stories(stories)
            
            print(f"{Colors.GREEN}✓ Loaded {len(stories)} stories{Colors.END}\n")
            
            # Step 2: Running run_chunking logic directly (since Pipeline class isn't in main src yet)
            # Or assume we can just run the script command via subprocess if we don't want to duplicate logic.
            # But let's try to use what we have.
            # Wait, user's code imported `ChunkingPipeline`. Does it exist?
            # List dir check says `preprocessor.py` and `script_detector.py` in preprocessing.
            # `chunker.py` is in `code/src/chunking/chunker.py`.
            # User's main.py imported `from src.preprocessing.chunking_pipeline import ChunkingPipeline`.
            # I don't see `chunking_pipeline.py` in `code/src/preprocessing`.
            
            # I will use subprocess to call existing scripts to be safe and avoid missing dependencies.
            print(f"{Colors.BOLD}{Colors.BLUE}[2/3] Chunking (via run_chunking.py){Colors.END}")
            import subprocess
            subprocess.run([sys.executable, "code/scripts/run_chunking.py"], check=True)
            
            print(f"{Colors.GREEN}✓ Chunking complete{Colors.END}\n")
            
            # Step 3: Indexing
            print(f"{Colors.BOLD}{Colors.BLUE}[3/3] Building Indexes (via run_indexing.py){Colors.END}")
            subprocess.run([sys.executable, "code/scripts/run_indexing.py"], check=True)
            
            print(f"{Colors.GREEN}✓ All indexes built successfully{Colors.END}\n")
            
            # Summary
            elapsed = (datetime.now() - start_time).total_seconds()
            
            print(f"{Colors.BOLD}{Colors.GREEN}{'='*60}{Colors.END}")
            print(f"{Colors.BOLD}{Colors.GREEN}INDEXING COMPLETE{Colors.END}")
            print(f"{Colors.BOLD}{Colors.GREEN}{'='*60}{Colors.END}")
            print(f"Total time: {elapsed:.2f} seconds")
            print(f"{Colors.BOLD}{Colors.GREEN}{'='*60}{Colors.END}\n")
            
            print(f"{Colors.YELLOW}You can now run queries with:{Colors.END}")
            print(f"  python code/main.py --mode query\n")
            
        except Exception as e:
            print(f"\n{Colors.RED}✗ Indexing failed: {e}{Colors.END}\n")
            logger.error(f"Indexing failed: {e}", exc_info=True)
            sys.exit(1)
    
    def single_query_mode(self, query: str, top_k: int = 2):
        """
        Run a single query and exit.
        
        Args:
            query: Query text
            top_k: Number of results
        """
        self.load_indexes()
        
        print(f"\n{Colors.BOLD}Query:{Colors.END} {query}\n")
        
        # Detect script
        self.detect_input_script(query)
        
        # Search
        results = self.search(query, top_k=top_k)
        
        # Display results
        if results:
            self.display_results(results, query)
            # Generate Answer
            self.generate_answer(query, results)
        else:
            print(f"\n{Colors.RED}No results found.{Colors.END}\n")
    
    def cleanup(self):
        """Cleanup resources."""
        if self.metadata_store:
            self.metadata_store.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sanskrit RAG System - Cross-Script Retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index documents
  python code/main.py --mode index --data ./data/raw
  
  # Interactive query mode
  python code/main.py --mode query
  
  # Single query
  python code/main.py --mode query --query "धर्मः किम्?"
  
  # Interactive mode (shorthand)
  python code/main.py --interactive
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['index', 'query'],
        help='Operation mode: index or query'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        help='Path to data directory or file (for index mode)'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        help='Query text (for single query mode)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=2,
        help='Number of results to return (default: 2)'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive query mode'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='code/config/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = SanskritRAGCLI(config_path=args.config)
    
    try:
        # Determine mode
        if args.interactive or (args.mode == 'query' and not args.query and not args.mode == 'index'):
             # Default to interactive if just 'python main.py' or 'python main.py --mode query'
            cli.interactive_mode()
        
        elif args.mode == 'query' and args.query:
            # Single query mode
            cli.single_query_mode(args.query, top_k=args.top_k)
        
        elif args.mode == 'index':
            # Index mode
            if not args.data:
                print(f"{Colors.RED}Error: --data is required for index mode{Colors.END}")
                parser.print_help()
                sys.exit(1)
            
            cli.index_mode(args.data)
        
        else:
            # No mode specified, show help
            parser.print_help()
    
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Exiting...{Colors.END}")
    
    finally:
        cli.cleanup()


if __name__ == "__main__":
    main()