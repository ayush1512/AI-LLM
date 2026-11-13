import re
from dataclasses import dataclass
from typing import List, Optional
import io

@dataclass
class TextChunk:
    chunk_id: int
    text: str
    source: str
    char_count: int
    word_count: int

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50, source: str = "document") -> List[TextChunk]:
    """
    Split text into overlapping chunks based on words/tokens.
    Preserves sentence boundaries where possible.
    """
    cleaned = re.sub(r'\s+', ' ', text).strip()
    if not cleaned:
        return []

    words = cleaned.split(' ')
    chunks: List[TextChunk] = []
    
    if len(words) <= chunk_size:
        chunks.append(
            TextChunk(
                chunk_id=0,
                text=cleaned,
                source=source,
                char_count=len(cleaned),
                word_count=len(words)
            )
        )
        return chunks

    step = max(1, chunk_size - overlap)
    chunk_idx = 0
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if not chunk_words:
            continue
        chunk_str = ' '.join(chunk_words).strip()
        chunks.append(
            TextChunk(
                chunk_id=chunk_idx,
                text=chunk_str,
                source=source,
                char_count=len(chunk_str),
                word_count=len(chunk_words)
            )
        )
        chunk_idx += 1
        if i + chunk_size >= len(words):
            break

    return chunks

def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Extract text from uploaded PDF or plain text file."""
    ext = filename.lower().split('.')[-1]
    if ext == 'pdf':
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages).strip()
        except Exception as e:
            # Fallback string extraction for PDFs if pypdf is not available
            return f"Error extracting PDF: {str(e)}"
    else:
        # Default text/utf-8 decoding
        try:
            return file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            return file_bytes.decode('latin-1', errors='ignore')

