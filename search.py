#!/usr/bin/env python3
"""
Hybrid search:  combine semantic (FAISS) + lexical (BM25) retrieval. 
"""

import logging
import pickle
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
import orjson
from rank_bm25 import BM25Okapi

from indexer import EmbeddingGenerator
from utils import load_config, setup_logging

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """Hybrid search combining FAISS and BM25."""
    
    def __init__(self, config: dict):
        self.config = config
        self. embedding_gen = EmbeddingGenerator(config)
        
        output_dir = Path(config["indexing"]["output_dir"])
        
        # Load FAISS index
        faiss_path = output_dir / config["indexing"]["faiss_index"]
        logger.info(f"Loading FAISS index from {faiss_path}")
        self.faiss_index = faiss.read_index(str(faiss_path))
        
        # Load BM25 index
        bm25_path = output_dir / config["indexing"]["bm25_index"]
        logger.info(f"Loading BM25 index from {bm25_path}")
        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)
        
        # Load metadata
        meta_path = output_dir / config["indexing"]["metadata_file"]
        logger.info(f"Loading metadata from {meta_path}")
        with open(meta_path, "rb") as f:
            self.chunks = orjson.loads(f.read())
        
        logger.info(f"Loaded {len(self.chunks)} chunks")
    
    def search_semantic(self, query: str, k: int = 10) -> List[tuple]:
        """Semantic search using FAISS."""
        query_emb = self.embedding_gen. embed_batch([query])
        distances, indices = self.faiss_index.search(query_emb, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                results.append((idx, float(dist)))
        return results
    
    def search_lexical(self, query: str, k: int = 10) -> List[tuple]:
        """Lexical search using BM25."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        results = [(int(idx), float(scores[idx])) for idx in top_indices]
        return results
    
    def search_hybrid(self, query: str, k:  int = 10, alpha: float = 0.5) -> List[Dict]:
        """
        Hybrid search with score fusion.
        alpha: weight for semantic (1-alpha for lexical).
        """
        # Get candidates from both
        sem_results = self.search_semantic(query, k=k*2)
        lex_results = self.search_lexical(query, k=k*2)
        
        # Normalize scores to [0, 1]
        def normalize_scores(results):
            if not results:
                return {}
            scores = [s for _, s in results]
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s: 
                return {idx: 0.5 for idx, _ in results}
            return {idx: (s - min_s) / (max_s - min_s) for idx, s in results}
        
        sem_scores = normalize_scores(sem_results)
        lex_scores = normalize_scores(lex_results)
        
        # Combine scores
        all_indices = set(sem_scores.keys()) | set(lex_scores.keys())
        combined = {}
        for idx in all_indices:
            sem_score = sem_scores.get(idx, 0.0)
            lex_score = lex_scores.get(idx, 0.0)
            combined[idx] = alpha * sem_score + (1 - alpha) * lex_score
        
        # Sort by combined score
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Return chunks with scores
        results = []
        for idx, score in ranked: 
            chunk = self.chunks[idx]. copy()
            chunk["_score"] = score
            results. append(chunk)
        
        return results


def main():
    config = load_config()
    setup_logging(config)
    
    engine = HybridSearchEngine(config)
    
    # Example queries
    queries = [
        "How is authentication implemented?",
        "Find the JWT token verification function",
        "Where are the database migrations?",
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        results = engine.search_hybrid(query, k=5, alpha=0.6)
        
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] Score: {result['_score']:. 3f}")
            print(f"File: {result['path']}:{result['start_line']}-{result['end_line']}")
            print(f"Symbols: {', '.join(result['symbols']) if result['symbols'] else 'N/A'}")
            print(f"Code preview:\n{result['code'][:200]}...")


if __name__ == "__main__":
    main()