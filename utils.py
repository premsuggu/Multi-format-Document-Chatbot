# utils.py

import re

def clean_text(text: str) -> str:
    """
    Clean input text by removing extra whitespace, line breaks, and non-printable chars.
    """
    text = ''.join(c for c in text if c.isprintable())
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Split text into word-based chunks with overlap.
    
    Args:
        text (str): Text to split.
        chunk_size (int): Number of words in each chunk.
        overlap (int): Number of words to overlap between chunks.
        
    Returns:
        List of text chunks.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
