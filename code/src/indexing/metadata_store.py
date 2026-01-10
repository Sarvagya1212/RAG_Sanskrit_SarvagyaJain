"""Metadata storage using SQLite."""

import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MetadataStore:
    """SQLite-based metadata storage for chunks."""
    
    def __init__(self, db_path: str = "data/processed/metadata.db"):
        """
        Initialize metadata store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._create_schema()
        logger.info(f"MetadataStore initialized: {db_path}")
    
    def _connect(self):
        """Connect to SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Return rows as dicts
    
    def _create_schema(self):
        """Create database schema."""
        schema = """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            story_id INTEGER,
            story_title TEXT,
            chunk_index INTEGER,
            text_original TEXT,
            text_slp1 TEXT,
            content_type TEXT,
            token_count INTEGER,
            vector_index INTEGER,
            created_at TIMESTAMP
        )
        """
        self.conn.execute(schema)
        self.conn.commit()
        logger.debug("Database schema created")
    
    def insert_chunks(self, chunks: List[Dict]):
        """
        Insert chunks into database.
        
        Args:
            chunks: List of chunk dictionaries
        """
        cursor = self.conn.cursor()
        
        for i, chunk in enumerate(chunks):
            # Generate globally unique chunk_id using story_id and index
            story_id = chunk.get('story_id', 0)
            local_id = chunk.get('chunk_id', i)
            chunk_id = f"s{story_id}_c{local_id}_i{i}"  # Globally unique
            
            cursor.execute(
                """
                INSERT OR REPLACE INTO chunks VALUES 
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    story_id,
                    chunk.get('story_title'),
                    i,  # chunk_index = position in overall list
                    chunk.get('text', chunk.get('original_text', '')),
                    chunk.get('slp1_text', chunk.get('text_slp1', '')),
                    chunk.get('content_type', chunk.get('type', '')),
                    chunk.get('token_count'),
                    i,  # vector_index = same as chunk_index
                    datetime.now()
                )
            )
        
        self.conn.commit()
        logger.info(f"Inserted {len(chunks)} chunks into metadata DB")
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Retrieve chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Chunk dictionary or None
        """
        cursor = self.conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?",
            (chunk_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_chunk_by_index(self, index: int) -> Optional[Dict]:
        """
        Retrieve chunk by vector index.
        
        Args:
            index: Vector index position
            
        Returns:
            Chunk dictionary or None
        """
        cursor = self.conn.execute(
            "SELECT * FROM chunks WHERE vector_index = ?",
            (index,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_chunks_by_story(self, story_id: int) -> List[Dict]:
        """
        Retrieve all chunks from a story.
        
        Args:
            story_id: Story identifier
            
        Returns:
            List of chunk dictionaries
        """
        cursor = self.conn.execute(
            "SELECT * FROM chunks WHERE story_id = ? ORDER BY chunk_index",
            (story_id,)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_all_chunks(self) -> List[Dict]:
        """
        Retrieve all chunks.
        
        Returns:
            List of all chunk dictionaries
        """
        cursor = self.conn.execute(
            "SELECT * FROM chunks ORDER BY chunk_index"
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with counts by content type and story
        """
        cursor = self.conn.execute("SELECT COUNT(*) FROM chunks")
        total = cursor.fetchone()[0]
        
        cursor = self.conn.execute(
            "SELECT content_type, COUNT(*) FROM chunks GROUP BY content_type"
        )
        by_type = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor = self.conn.execute(
            "SELECT COUNT(DISTINCT story_id) FROM chunks"
        )
        num_stories = cursor.fetchone()[0]
        
        return {
            'total_chunks': total,
            'num_stories': num_stories,
            'by_content_type': by_type
        }
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
