#!/usr/bin/env python3
"""
Build vector index (FAISS) and BM25 index from chunked JSONL. 
"""

import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
import orjson
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from utils import load_config, setup_logging

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using HuggingFace or OpenAI."""
    
    def __init__(self, config: dict):
        self.provider = config["embedding"]["provider"]
        self.model_name = config["embedding"]["model"]
        self.batch_size = config["embedding"]["batch_size"]
        self.device = config["embedding"]["device"]
        
        if self.provider == "huggingface":
            self._init_huggingface()
        elif self.provider == "openai":
            self._init_openai()
        else:
            raise ValueError(f"Unknown provider:  {self.provider}")
    
    def _init_huggingface(self):
        """Initialize HuggingFace model."""
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        logger.info(f"Loading HuggingFace model:  {self.model_name}")
        self.tokenizer = AutoTokenizer. from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self.torch = torch
        logger.info("HuggingFace model loaded")
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        import openai
        self.client = openai.OpenAI()
        logger.info(f"OpenAI client initialized for model: {self.model_name}")
    
    def embed_batch_hf(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using HuggingFace."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with self.torch. no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy()
    
    def embed_batch_openai(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        response = self.client.embeddings.create(
            input=texts,
            model=self. model_name
        )
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype=np.float32)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts."""
        if self.provider == "huggingface":
            return self.embed_batch_hf(texts)
        elif self.provider == "openai":
            return self.embed_batch_openai(texts)
    
    def embed_all(self, texts: List[str]) -> np.ndarray:
        """Embed all texts in batches."""
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding"):
            batch = texts[i:i + self. batch_size]
            emb = self.embed_batch(batch)
            all_embeddings.append(emb)
        
        return np.vstack(all_embeddings)


class IndexBuilder:
    """Build FAISS and BM25 indices."""
    
    def __init__(self, config: dict):
        self.config = config
        self.embedding_gen = EmbeddingGenerator(config)
    
    def load_chunks(self, jsonl_path:  str) -> tuple[List[Dict], List[str]]:
        """Load chunks from JSONL."""
        chunks = []
        texts = []
        
        logger.info(f"Loading chunks from {jsonl_path}")
        with open(jsonl_path, "rb") as f:
            for line in f:
                chunk = orjson.loads(line)
                chunks.append(chunk)
                
                # Combine code + docstring + symbols for embedding
                text_parts = [chunk["code"]]
                if chunk. get("docstring"):
                    text_parts.append(chunk["docstring"])
                if chunk. get("symbols"):
                    text_parts.append(" ".join(chunk["symbols"]))
                
                texts.append("\n".join(text_parts))
        
        logger.info(f"Loaded {len(chunks)} chunks")
        return chunks, texts
    
    def build_faiss(self, embeddings: np.ndarray, output_path: str):
        """Build and save FAISS index."""
        dimension = embeddings.shape[1]
        logger.info(f"Building FAISS index with dimension {dimension}")
        
        # Use HNSW for better performance
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.add(embeddings)
        
        faiss.write_index(index, output_path)
        logger.info(f"FAISS index saved to {output_path}")
    
    def build_bm25(self, texts: List[str], output_path: str):
        """Build and save BM25 index."""
        logger.info("Building BM25 index")
        
        # Tokenize
        tokenized = [text.lower().split() for text in texts]
        bm25 = BM25Okapi(tokenized)
        
        with open(output_path, "wb") as f:
            pickle. dump(bm25, f)
        
        logger.info(f"BM25 index saved to {output_path}")
    
    def save_metadata(self, chunks: List[Dict], output_path: str):
        """Save chunk metadata."""
        with open(output_path, "wb") as f:
            f.write(orjson.dumps(chunks, option=orjson.OPT_INDENT_2))
        logger.info(f"Metadata saved to {output_path}")
    
    def run(self):
        """Run full indexing pipeline."""
        output_dir = Path(self.config["indexing"]["output_dir"])
        jsonl_path = output_dir / self.config["indexing"]["jsonl_file"]
        
        # Load chunks
        chunks, texts = self.load_chunks(str(jsonl_path))
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedding_gen. embed_all(texts)
        
        # Build FAISS
        faiss_path = output_dir / self.config["indexing"]["faiss_index"]
        self.build_faiss(embeddings, str(faiss_path))
        
        # Build BM25
        bm25_path = output_dir / self.config["indexing"]["bm25_index"]
        self.build_bm25(texts, str(bm25_path))
        
        # Save metadata
        meta_path = output_dir / self.config["indexing"]["metadata_file"]
        self.save_metadata(chunks, str(meta_path))
        
        logger. info("Indexing complete!")


def main():
    config = load_config()
    setup_logging(config)
    
    builder = IndexBuilder(config)
    builder.run()


if __name__ == "__main__":
    main()