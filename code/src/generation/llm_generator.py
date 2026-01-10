"""LLM generator using llama-cpp for Sanskrit RAG responses."""

from typing import List, Dict, Optional
from llama_cpp import Llama
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PromptTemplate:
    """Prompt template for Sanskrit RAG system."""
    
    SYSTEM_PROMPT = """You are a helpful assistant specialized in Sanskrit literature and stories. 
You answer questions based on the provided context from Sanskrit texts.
When answering:
- Use the context provided to give accurate answers
- If the answer is not in the context, say so clearly
- Cite the story name when referencing information
- Be concise but informative
"""
    
    def format_context(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks as context.
        
        Args:
            chunks: List of retrieved chunks with metadata
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            story_title = chunk.get('story_title', 'Unknown')
            text = chunk.get('text_original', chunk.get('text', ''))
            
            context_parts.append(
                f"[Source {i}: {story_title}]\n{text}\n"
            )
        
        return "\n".join(context_parts)
    
    def build_prompt(self, query: str, chunks: List[Dict]) -> str:
        """
        Build complete prompt with system, context, and query.
        
        Args:
            query: User query
            chunks: Retrieved chunks
            
        Returns:
            Complete prompt string
        """
        context = self.format_context(chunks)
        
        prompt = f"""{self.SYSTEM_PROMPT}

Context from Sanskrit texts:
{context}

User Question: {query}

Answer:"""
        
        return prompt


class LLMGenerator:
    """
    LLM-based answer generation using llama-cpp.
    
    Loads Qwen model and generates answers with context injection.
    """
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 4,
        temperature: float = 0.7,
        max_tokens: int = 512
    ):
        """
        Initialize LLM generator.
        
        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size
            n_threads: Number of CPU threads
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = None
        self.prompt_template = PromptTemplate()
        
        logger.info(f"LLMGenerator initialized (model: {model_path})")
    
    def load_model(self):
        """Load the Qwen model."""
        logger.info(f"Loading model from {self.model_path}...")
        
        self.model = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=False
        )
        
        logger.info("Model loaded successfully")
    
    def generate(
        self,
        query: str,
        context_chunks: List[Dict],
        include_citations: bool = True
    ) -> Dict:
        """
        Generate answer with context injection.
        
        Args:
            query: User query
            context_chunks: Retrieved chunks for context
            include_citations: Whether to include source citations
            
        Returns:
            {
                'answer': generated answer,
                'sources': list of source citations,
                'prompt': full prompt used,
                'metadata': generation metadata
            }
        """
        if self.model is None:
            self.load_model()
        
        # Build prompt with context
        prompt = self.prompt_template.build_prompt(query, context_chunks)
        
        logger.debug(f"Generating answer for query: '{query}'")
        logger.debug(f"Using {len(context_chunks)} context chunks")
        
        # Generate
        response = self.model(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["User Question:", "\n\n\n"],
            echo=False
        )
        
        # Extract answer
        answer_text = response['choices'][0]['text'].strip()
        
        # Build citations
        sources = []
        if include_citations and context_chunks:
            seen_stories = set()
            for chunk in context_chunks:
                story_title = chunk.get('story_title', 'Unknown')
                if story_title not in seen_stories:
                    sources.append({
                        'story_title': story_title,
                        'story_id': chunk.get('story_id'),
                        'chunk_id': chunk.get('chunk_id')
                    })
                    seen_stories.add(story_title)
        
        result = {
            'answer': answer_text,
            'sources': sources,
            'prompt': prompt,
            'metadata': {
                'num_context_chunks': len(context_chunks),
                'tokens_generated': response['usage']['completion_tokens'],
                'model': self.model_path
            }
        }
        
        logger.info(f"Generated answer ({result['metadata']['tokens_generated']} tokens)")
        return result
    
    def generate_with_explanation(
        self,
        query: str,
        context_chunks: List[Dict]
    ) -> Dict:
        """
        Generate answer with detailed explanation of sources.
        
        Returns:
            {
                'answer': generated answer,
                'sources_detail': detailed source information,
                'context_used': formatted context,
                'metadata': generation metadata
            }
        """
        if self.model is None:
            self.load_model()
        
        # Build prompt
        prompt = self.prompt_template.build_prompt(query, context_chunks)
        
        # Generate
        response = self.model(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["User Question:", "\n\n\n"],
            echo=False
        )
        
        answer_text = response['choices'][0]['text'].strip()
        
        # Detailed source information
        sources_detail = []
        for i, chunk in enumerate(context_chunks, 1):
            sources_detail.append({
                'rank': i,
                'story_title': chunk.get('story_title', 'Unknown'),
                'story_id': chunk.get('story_id'),
                'chunk_id': chunk.get('chunk_id'),
                'text_snippet': chunk.get('text_original', '')[:100] + '...',
                'retrieval_score': chunk.get('retrieval_score', 0.0)
            })
        
        return {
            'answer': answer_text,
            'sources_detail': sources_detail,
            'context_used': self.prompt_template.format_context(context_chunks),
            'metadata': {
                'num_context_chunks': len(context_chunks),
                'tokens_generated': response['usage']['completion_tokens'],
                'prompt_length': len(prompt),
                'model': self.model_path
            }
        }
