"""Metadata storage using SQLite with parent-child chunk support."""

import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MetadataStore:
    """SQLite-based metadata storage for parent-child chunks."""
    
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
        """Create database schema for parent-child chunks."""
        schema = """
        CREATE TABLE IF NOT EXISTS child_chunks (
            chunk_id TEXT PRIMARY KEY,
            parent_id TEXT NOT NULL,
            story_id TEXT NOT NULL,
            story_title TEXT,
            
            -- Child chunk data (indexed for search)
            text TEXT NOT NULL,
            preprocessed_text TEXT NOT NULL,
            
            -- Parent chunk data (context)
            parent_text TEXT NOT NULL,
            parent_preprocessed TEXT NOT NULL,
            
            -- Hierarchy metadata
            child_index INTEGER,
            total_children INTEGER,
            
            -- Additional metadata
            token_count INTEGER,
            vector_index INTEGER,
            created_at TIMESTAMP,
            
            FOREIGN KEY (parent_id) REFERENCES parent_chunks(parent_id)
        );
        
        CREATE TABLE IF NOT EXISTS parent_chunks (
            parent_id TEXT PRIMARY KEY,
            story_id TEXT NOT NULL,
            story_title TEXT,
            
            -- Parent chunk data
            text TEXT NOT NULL,
            preprocessed_text TEXT NOT NULL,
            
            -- Metadata
            token_count INTEGER,
            child_count INTEGER,
            created_at TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_parent_id ON child_chunks(parent_id);
        CREATE INDEX IF NOT EXISTS idx_story_id ON child_chunks(story_id);
        """
        self.conn.executescript(schema)
        self.conn.commit()
        logger.debug("Database schema created for parent-child chunks")
    
    def insert_parent_chunks(self, parents: List[Dict]):
        """
        Insert parent chunks into database.
        
        Args:
            parents: List of parent chunk dictionaries
        """
        cursor = self.conn.cursor()
        
        for parent in parents:
            cursor.execute(
                """
                INSERT OR REPLACE INTO parent_chunks VALUES 
                (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    parent['parent_id'],
                    parent['story_id'],
                    parent['story_title'],
                    parent['text'],
                    parent['preprocessed_text'],
                    parent['token_count'],
                    parent.get('child_count', 0),
                    datetime.now()
                )
            )
        
        self.conn.commit()
        logger.info(f"Inserted {len(parents)} parent chunks")
    
    def insert_child_chunks(self, children: List[Dict]):
        """
        Insert child chunks into database.
        
        Args:
            children: List of child chunk dictionaries
        """
        cursor = self.conn.cursor()
        
        for i, child in enumerate(children):
            cursor.execute(
                """
                INSERT OR REPLACE INTO child_chunks VALUES 
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    child['chunk_id'],
                    child['parent_id'],
                    child['story_id'],
                    child['story_title'],
                    child['text'],
                    child['preprocessed_text'],
                    child['parent_text'],
                    child['parent_preprocessed'],
                    child['child_index'],
                    child['total_children'],
                    child['token_count'],
                    i,  # vector_index = position in overall list
                    datetime.now()
                )
            )
        
        self.conn.commit()
        logger.info(f"Inserted {len(children)} child chunks")
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Retrieve a child chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Chunk dictionary or None
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM child_chunks WHERE chunk_id = ?",
            (chunk_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
        
    def get_chunk_by_vector_index(self, index: int) -> Optional[Dict]:
        """
        Retrieve a child chunk by its vector/list index.
        
        Args:
            index: Integer index (from BM25 or FAISS)
            
        Returns:
            Chunk dictionary or None
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM child_chunks WHERE vector_index = ?",
            (index,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_parent_by_id(self, parent_id: str) -> Optional[Dict]:
        """
        Retrieve a parent chunk by ID.
        
        Args:
            parent_id: Parent chunk identifier
            
        Returns:
            Parent chunk dictionary or None
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM parent_chunks WHERE parent_id = ?",
            (parent_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_chunks_by_parent(self, parent_id: str) -> List[Dict]:
        """
        Get all child chunks for a parent.
        
        Args:
            parent_id: Parent chunk identifier
            
        Returns:
            List of child chunk dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM child_chunks WHERE parent_id = ? ORDER BY child_index",
            (parent_id,)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_all_child_chunks(self) -> List[Dict]:
        """
        Get all child chunks.
        
        Returns:
            List of all child chunk dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM child_chunks ORDER BY vector_index")
        return [dict(row) for row in cursor.fetchall()]
    
    def get_statistics(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Statistics dictionary
        """
        cursor = self.conn.cursor()
        
        # Parent count
        cursor.execute("SELECT COUNT(*) as count FROM parent_chunks")
        parent_count = cursor.fetchone()['count']
        
        # Child count
        cursor.execute("SELECT COUNT(*) as count FROM child_chunks")
        child_count = cursor.fetchone()['count']
        
        # Unique stories
        cursor.execute("SELECT COUNT(DISTINCT story_id) as count FROM child_chunks")
        story_count = cursor.fetchone()['count']
        
        return {
            'parent_chunks': parent_count,
            'child_chunks': child_count,
            'stories': story_count,
            'avg_children_per_parent': child_count / parent_count if parent_count > 0 else 0
        }
    
    def clear_all(self):
        """Clear all data from database."""
        self.conn.execute("DELETE FROM child_chunks")
        self.conn.execute("DELETE FROM parent_chunks")
        self.conn.commit()
        logger.info("Cleared all chunks from database")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
