"""
LLM Generator for Sanskrit RAG System.
Uses parent-child chunking strategy for context.
"""

import os
import time
from typing import List, Dict, Optional
from llama_cpp import Llama
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__)


def get_optimal_threads():
    """
    Get optimal thread count for CPU inference.
    Uses physical cores (not hyperthreaded) for best performance.
    """
    try:
        import psutil
        physical_cores = psutil.cpu_count(logical=False)
    except ImportError:
        physical_cores = None
    
    # Fallback if psutil not available
    if physical_cores is None:
        total_cores = os.cpu_count()
        physical_cores = total_cores // 2 if total_cores else 4
    
    # Leave 1 core for OS, minimum 1 thread
    return max(1, physical_cores - 1)


class PromptTemplate:
    """Prompt template for Sanskrit RAG."""
    
    SYSTEM_PROMPT = """<|im_start|>system
You are an expert Sanskrit scholar and teacher specializing in classical Sanskrit literature.

Your role is to answer questions based STRICTLY on the provided Sanskrit text passages. Follow these rules carefully:

1. Answer based only on provided context
2. Quote Sanskrit when relevant
3. Be concise and direct (2-4 sentences)
4. Cite your sources
5. If information is not in passages, say so explicitly
6. Never fabricate information
<|im_end|>"""

    def __init__(self, config: Optional[Dict] = None) -> None:
        self.config: Dict = config or {}
        
        # Get LLM config
        llm_config = self.config.get('models', {}).get('llm', {})
        
        # System prompt from config or default
        if 'system_prompt' in llm_config:
            self.system_prompt = f"<|im_start|>system\n{llm_config['system_prompt'].strip()}\n<|im_end|>"
        else:
            self.system_prompt = self.SYSTEM_PROMPT
        
        # User instruction from config or default
        self.user_instruction = llm_config.get(
            'user_instruction', 
            "Provide a detailed answer based on the passage above."
        )
        
        # Assistant prefix from config or default
        self.assistant_prefix = llm_config.get(
            'assistant_prefix', 
            "Based on the passage, "
        )

    def format_context(self, chunks: List[Dict], query: str = "") -> str:
        """Format chunks for LLM context."""
        if not chunks:
            return "No context."
        
        MAX_CHARS = 1200
        
        parts = []
        for i, chunk in enumerate(chunks, 1):
            title = chunk.get('story_title', '')
            text = chunk.get('parent_text') or chunk.get('text', '')
            
            if len(text) > MAX_CHARS:
                text = text[:MAX_CHARS] + "..."
            
            parts.append(f"[Passage {i} - {title}]\n{text}")
        
        return "\n\n".join(parts)
    
    def build_prompt(self, query: str, chunks: List[Dict]) -> str:
        """Build prompt for LLM using config values."""
        context = self.format_context(chunks, query)
        
        return f"""{self.system_prompt}

<|im_start|>user
Here is the Sanskrit passage:

{context}

Question: {query}

{self.user_instruction}
<|im_end|>

<|im_start|>assistant
{self.assistant_prefix}"""


class LLMGenerator:
    """Optimized LLM-based answer generator."""
    
    def __init__(self, config: dict, model_path: Optional[str] = None, **kwargs):
        """
        Initialize LLM generator.
        Supports both new (config dict) and old (individual args) initialization.
        """
        self.config = config
        # Handle both config dict and direct args for path
        self.model_path = model_path or config['models']['llm']['path']
        
        # Optimized generation parameters from config
        llm_conf = config['models']['llm']
        self.n_ctx = llm_conf.get('n_ctx', 2048)
        self.temperature = llm_conf.get('temperature', 0.1)
        self.max_tokens = llm_conf.get('max_tokens', 300)
        self.top_p = llm_conf.get('top_p', 0.9)
        
        # CPU optimization: Auto-detect physical cores if not specified in config
        self.n_threads = llm_conf.get('n_threads') or get_optimal_threads()
        
        self.model = None
        self.prompt_template = PromptTemplate(config=config)
        
        logger.info(f"LLMGenerator initialized (threads: {self.n_threads}, ctx: {self.n_ctx})")

    def load_model(self):
        """Load LLM model with optimizations."""
        if self.model is not None:
            return
            
        logger.info(f"Loading model from {self.model_path}...")
        start_time = time.time()
        
        try:
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_batch=512,      # Optimized batch size
                verbose=False,
                n_gpu_layers=0,   # Explicit CPU only
                use_mmap=True,
                use_mlock=False
            )
            logger.info(f"Model loaded in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate(self, query: str, context_chunks: List[Dict], include_citations: bool = True) -> Dict:
        """
        Main generation method (Backward compatible wrapper).
        """
        return self.generate_answer(query, context_chunks)

    def generate_answer(self, query: str, context_chunks: List[Dict], max_context_chars: int = 0) -> Dict:
        """Generate answer from context chunks."""
        if self.model is None:
            self.load_model()
            
        prompt = self.prompt_template.build_prompt(query, context_chunks)
        
        try:
            response = self.model(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repeat_penalty=1.2,
                stop=["<|im_end|>", "\n\nQuestion:", "Sources:"],
                echo=False
            )
            
            answer_text = response['choices'][0]['text'].strip()
            
            # Build sources list
            sources = []
            seen = set()
            for chunk in context_chunks:
                title = chunk.get('story_title', 'Unknown')
                if title not in seen:
                    sources.append({'story_title': title})
                    seen.add(title)
            
            return {
                'answer': answer_text,
                'sources': sources,
                'prompt': prompt,
                'metadata': response.get('usage', {})
            }
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {
                'answer': f"Error: {str(e)}",
                'sources': [],
                'metadata': {'error': str(e)}
            }

    def quick_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Quick answer for diagnostics."""
        res = self.generate_answer(query, context_chunks[:2])
        return res['answer']