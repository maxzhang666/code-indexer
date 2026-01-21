#!/usr/bin/env python3
"""Utility functions for code indexing."""

import hashlib
import logging
import re
from pathlib import Path
from typing import List, Optional

import yaml


def setup_logging(config:  dict):
    """Setup logging configuration."""
    level = getattr(logging, config. get("logging", {}).get("level", "INFO"))
    log_file = config.get("logging", {}).get("file", "indexer.log")
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging. FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def sha256_hex(data: bytes) -> str:
    """Compute SHA256 hash."""
    return hashlib.sha256(data).hexdigest()


def should_exclude(path: Path, patterns: List[str]) -> bool:
    """Check if path matches any exclusion pattern."""
    path_str = str(path)
    for pattern in patterns:
        if Path(path_str).match(pattern):
            return True
    return False


def is_binary(path: Path) -> bool:
    """Detect binary files."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(1024)
            return b"\0" in chunk
    except Exception:
        return True


def extract_imports_python(code: str) -> List[str]:
    """Extract import statements from Python code."""
    imports = []
    for line in code.split("\n"):
        line = line.strip()
        if line.startswith("import ") or line.startswith("from "):
            imports.append(line)
    return imports


def extract_imports_js(code: str) -> List[str]:
    """Extract import/require from JavaScript."""
    imports = []
    patterns = [
        r'import\s+.*?from\s+["\'](.+?)["\']',
        r'require\(["\'](.+?)["\']\)',
    ]
    for pattern in patterns: 
        imports.extend(re.findall(pattern, code))
    return imports


def detect_file_role(path: Path) -> str:
    """Detect file role (test/doc/code)."""
    path_lower = str(path).lower()
    if any(x in path_lower for x in ["test", "spec", "__test__"]):
        return "test"
    if any(x in path_lower for x in ["readme", "doc", "docs"]):
        return "doc"
    return "code"


def normalize_code(code: str) -> str:
    """Normalize code for hashing/dedup."""
    lines = [line.rstrip() for line in code.splitlines()]
    return "\n".join(line for line in lines if line)  # remove empty lines


class TokenCounter:
    """Token counter with multiple backend support."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.backend = None
        self.model_name = model_name
        
        # Try tiktoken first (for OpenAI models)
        try:
            import tiktoken
            self.tokenizer = tiktoken.encoding_for_model(model_name)
            self.backend = "tiktoken"
        except Exception:
            pass
        
        # Fallback to transformers
        if not self.backend:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.backend = "transformers"
            except Exception:
                self.backend = "simple"
    
    def count(self, text: str) -> int:
        """Count tokens in text."""
        if self.backend == "tiktoken":
            return len(self.tokenizer.encode(text))
        elif self.backend == "transformers":
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        else:
            # Simple word-based approximation
            return max(1, len(text.split()))