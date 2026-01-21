#!/usr/bin/env python3
"""
Production-grade code chunker with AST parsing, multi-language support,
parallel processing, and incremental indexing.
"""

import json
import logging
import multiprocessing as mp
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Iterator

import orjson
from git import Repo
from tqdm import tqdm
from tree_sitter_languages import get_language, get_parser

from utils import (
    load_config,
    setup_logging,
    sha256_hex,
    should_exclude,
    is_binary,
    extract_imports_python,
    extract_imports_js,
    detect_file_role,
    normalize_code,
    TokenCounter,
)

logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """Represents a single code chunk."""
    id: str
    repo: str
    path: str
    commit: str
    branch: str
    start_line: int
    end_line: int
    language: str
    code: str
    code_normalized: str
    symbols: List[str]
    docstring: str
    imports: List[str]
    file_role: str
    is_test: bool
    size_tokens: int
    size_chars: int
    content_hash: str
    
    def to_dict(self):
        return asdict(self)


class ASTChunker:
    """AST-based chunker using tree-sitter."""
    
    # Language-specific query patterns for functions/classes
    QUERIES = {
        "python": """
            (function_definition) @function
            (class_definition) @class
        """,
        "javascript": """
            (function_declaration) @function
            (class_declaration) @class
            (method_definition) @method
        """,
        "typescript": """
            (function_declaration) @function
            (class_declaration) @class
            (method_definition) @method
        """,
        "java": """
            (method_declaration) @method
            (class_declaration) @class
        """,
        "go": """
            (function_declaration) @function
            (method_declaration) @method
        """,
        "rust": """
            (function_item) @function
            (impl_item) @impl
        """,
    }
    
    def __init__(self, language: str):
        self.language = language
        try:
            self.parser = get_parser(language)
            self.ts_language = get_language(language)
            self.available = True
            logger.debug(f"Initialized tree-sitter parser for {language}")
        except Exception as e: 
            logger.warning(f"Could not load tree-sitter for {language}: {e}")
            self.available = False
    
    def extract_chunks(self, code: str, path: str) -> List[Tuple[int, int, str, List[str], str]]:
        """
        Extract code chunks using AST. 
        Returns list of (start_line, end_line, code_text, symbols, docstring).
        """
        if not self.available:
            return []
        
        try:
            tree = self.parser.parse(bytes(code, "utf-8"))
            root = tree.root_node
            
            chunks = []
            query_str = self.QUERIES. get(self.language)
            if not query_str:
                return []
            
            query = self.ts_language.query(query_str)
            captures = query.captures(root)
            
            for node, _ in captures:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                chunk_code = code. split("\n")[start_line - 1:end_line]
                chunk_text = "\n".join(chunk_code)
                
                # Extract symbol name
                symbols = self._extract_symbols(node, code)
                
                # Extract docstring
                docstring = self._extract_docstring(node, code)
                
                chunks. append((start_line, end_line, chunk_text, symbols, docstring))
            
            return chunks
            
        except Exception as e:
            logger.error(f"AST parsing failed for {path}: {e}")
            return []
    
    def _extract_symbols(self, node, code: str) -> List[str]:
        """Extract function/class names from node."""
        symbols = []
        # Look for identifier child node
        for child in node. children:
            if child.type in ["identifier", "type_identifier"]:
                name = code[child.start_byte:child.end_byte]
                symbols.append(name)
        return symbols
    
    def _extract_docstring(self, node, code: str) -> str:
        """Extract docstring/comment from function/class."""
        # Python:  first string in body
        if self. language == "python":
            for child in node.children:
                if child.type == "block": 
                    for stmt in child.children:
                        if stmt.type == "expression_statement":
                            for expr in stmt.children:
                                if expr.type == "string":
                                    return code[expr.start_byte:expr.end_byte]. strip('"\'')
        # TODO: Add javadoc, jsdoc extraction for other languages
        return ""


class LineChunker:
    """Fallback line-based chunker with sliding window."""
    
    def __init__(self, max_lines: int = 100, overlap: int = 20):
        self.max_lines = max_lines
        self. overlap = overlap
    
    def extract_chunks(self, code: str) -> List[Tuple[int, int, str]]:
        """Returns (start_line, end_line, code_text)."""
        lines = code.splitlines()
        chunks = []
        i = 0
        n = len(lines)
        
        while i < n:
            end = min(i + self.max_lines, n)
            chunk_lines = lines[i:end]
            chunk_text = "\n".join(chunk_lines)
            chunks.append((i + 1, end, chunk_text))
            
            if end == n:
                break
            i += self.max_lines - self.overlap
        
        return chunks


class CodeChunkerPipeline:
    """Main pipeline for chunking repository code."""
    
    def __init__(self, config: dict):
        self.config = config
        self.repo_path = Path(config["repo"]["path"])
        self.repo_name = config["repo"]["name"]
        self.branch = config["repo"]["branch"]
        
        self.max_tokens = config["chunking"]["max_tokens"]
        self.min_tokens = config["chunking"]["min_tokens"]
        self. prefer_ast = config["chunking"]["prefer_ast"]
        
        self.exclude_patterns = config["exclude"]
        self.languages = set(config["languages"])
        
        # Initialize git repo for incremental support
        try:
            self.git_repo = Repo(self. repo_path)
            self.current_commit = self.git_repo.head.commit. hexsha
            logger.info(f"Git repo loaded, current commit: {self.current_commit[: 8]}")
        except Exception as e:
            logger.warning(f"Not a git repo or error:  {e}")
            self.git_repo = None
            self. current_commit = ""
        
        # Token counter
        embed_model = config["embedding"]["model"]
        self.token_counter = TokenCounter(embed_model)
        
        # Cache of AST parsers
        self.ast_parsers = {}
    
    def get_parser(self, language: str) -> Optional[ASTChunker]:
        """Get or create AST parser for language."""
        if language not in self.ast_parsers:
            self.ast_parsers[language] = ASTChunker(language)
        return self.ast_parsers[language]
    
    def chunk_file(self, file_path: Path) -> List[CodeChunk]: 
        """Chunk a single file."""
        if is_binary(file_path):
            return []
        
        try: 
            code = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return []
        
        if not code.strip():
            return []
        
        # Determine language
        ext = file_path.suffix.lstrip(".")
        if ext not in self.languages:
            return []
        
        language = ext
        if ext == "ts":
            language = "typescript"
        elif ext == "js": 
            language = "javascript"
        elif ext == "py":
            language = "python"
        
        # Try AST first
        chunks_data = []
        if self.prefer_ast:
            parser = self.get_parser(language)
            if parser and parser.available:
                ast_chunks = parser.extract_chunks(code, str(file_path))
                if ast_chunks:
                    for start, end, chunk_code, symbols, docstring in ast_chunks:
                        chunks_data.append({
                            "start":  start,
                            "end":  end,
                            "code":  chunk_code,
                            "symbols": symbols,
                            "docstring": docstring,
                        })
        
        # Fallback to line chunker
        if not chunks_data: 
            fallback = LineChunker(
                max_lines=self.config["chunking"]["fallback_lines"],
                overlap=20
            )
            for start, end, chunk_code in fallback.extract_chunks(code):
                chunks_data.append({
                    "start": start,
                    "end": end,
                    "code": chunk_code,
                    "symbols": [],
                    "docstring": "",
                })
        
        # Build CodeChunk objects
        results = []
        rel_path = file_path.relative_to(self.repo_path)
        file_role = detect_file_role(rel_path)
        
        for chunk_info in chunks_data:
            chunk_code = chunk_info["code"]
            code_norm = normalize_code(chunk_code)
            token_count = self.token_counter.count(chunk_code)
            
            # Skip too small chunks
            if token_count < self.min_tokens:
                continue
            
            # Extract imports
            imports = []
            if language == "python":
                imports = extract_imports_python(chunk_code)
            elif language in ["javascript", "typescript"]:
                imports = extract_imports_js(chunk_code)
            
            chunk_hash = sha256_hex(code_norm. encode("utf-8"))
            chunk_id = f"{self.repo_name}:{rel_path}:{chunk_info['start']}:{chunk_info['end']}:{chunk_hash[: 12]}"
            
            chunk_obj = CodeChunk(
                id=chunk_id,
                repo=self.repo_name,
                path=str(rel_path),
                commit=self.current_commit,
                branch=self.branch,
                start_line=chunk_info["start"],
                end_line=chunk_info["end"],
                language=language,
                code=chunk_code,
                code_normalized=code_norm,
                symbols=chunk_info["symbols"],
                docstring=chunk_info["docstring"],
                imports=imports,
                file_role=file_role,
                is_test=(file_role == "test"),
                size_tokens=token_count,
                size_chars=len(chunk_code),
                content_hash=chunk_hash,
            )
            results.append(chunk_obj)
        
        return results
    
    def get_files_to_process(self) -> List[Path]:
        """Get list of files to process."""
        all_files = []
        for p in self.repo_path.rglob("*"):
            if p.is_file():
                if should_exclude(p. relative_to(self.repo_path), self.exclude_patterns):
                    continue
                all_files.append(p)
        logger.info(f"Found {len(all_files)} files to process")
        return all_files
    
    def process_parallel(self, files:  List[Path], num_workers: int) -> Iterator[CodeChunk]:
        """Process files in parallel."""
        with mp.Pool(num_workers) as pool:
            for chunks in tqdm(
                pool.imap_unordered(self.chunk_file, files),
                total=len(files),
                desc="Chunking files"
            ):
                for chunk in chunks:
                    yield chunk
    
    def run(self, output_path: str, num_workers: int = 4):
        """Run the chunking pipeline."""
        files = self.get_files_to_process()
        
        logger.info(f"Starting chunking with {num_workers} workers...")
        count = 0
        
        with open(output_path, "wb") as f:
            for chunk in self.process_parallel(files, num_workers):
                f.write(orjson.dumps(chunk.to_dict()) + b"\n")
                count += 1
        
        logger.info(f"Wrote {count} chunks to {output_path}")


def main():
    config = load_config()
    setup_logging(config)
    
    output_dir = Path(config["indexing"]["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    jsonl_path = output_dir / config["indexing"]["jsonl_file"]
    
    pipeline = CodeChunkerPipeline(config)
    pipeline.run(
        output_path=str(jsonl_path),
        num_workers=config["performance"]["num_workers"]
    )


if __name__ == "__main__":
    main()