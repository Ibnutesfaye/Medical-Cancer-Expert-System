"""
Text chunking module for Medical Cancer Expert System.

Splits text into overlapping chunks using character-based sizing
(no external vocab files required — avoids MemoryError from tiktoken BPE).
"""

from typing import List
from pydantic import BaseModel


# Approximate chars-per-token for English medical text
_CHARS_PER_TOKEN = 4


class TextChunk(BaseModel):
    """Represents a chunk of text with metadata."""
    text: str
    document_name: str
    page_number: int
    chunk_index: int
    token_count: int


class TextChunker:
    """
    Splits text into overlapping chunks with character-based sizing.

    Token counts are approximated as len(text) // 4, which is accurate
    enough for English medical prose and requires no large vocab files.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        encoding_name: str = "cl100k_base",   # kept for API compatibility, ignored
    ):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Maximum *approximate* tokens per chunk
            overlap: Token overlap between consecutive chunks
            encoding_name: Ignored (kept for backward compatibility)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Convert token counts to character counts
        self._char_size = chunk_size * _CHARS_PER_TOKEN
        self._char_overlap = overlap * _CHARS_PER_TOKEN

    def chunk_text(
        self, text: str, document_name: str, page_number: int = 1
    ) -> List[TextChunk]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            document_name: Source document name
            page_number: Starting page number

        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []

        chunks: List[TextChunk] = []
        chunk_index = 0
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self._char_size, text_len)
            chunk_text = text[start:end]
            approx_tokens = max(1, len(chunk_text) // _CHARS_PER_TOKEN)

            chunks.append(
                TextChunk(
                    text=chunk_text,
                    document_name=document_name,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    token_count=approx_tokens,
                )
            )

            step = self._char_size - self._char_overlap
            start += step if step > 0 else self._char_size
            chunk_index += 1

        return chunks

    def count_tokens(self, text: str) -> int:
        """Approximate token count for text."""
        return max(1, len(text) // _CHARS_PER_TOKEN)


def chunk_document(
    text: str,
    document_name: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[TextChunk]:
    """
    Convenience function to chunk a document.

    Args:
        text: Document text
        document_name: Source document name
        chunk_size: Maximum approximate tokens per chunk
        overlap: Token overlap between chunks

    Returns:
        List of TextChunk objects
    """
    chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
    return chunker.chunk_text(text, document_name)

