"""
Unit tests for the text chunking module.

Tests specific examples, edge cases, and error conditions for text chunking.
"""

import pytest
from text_chunker import TextChunker, TextChunk


class TestTextChunker:
    """Unit tests for TextChunker class."""
    
    def test_initialization_with_defaults(self):
        """Test that TextChunker initializes with default parameters."""
        chunker = TextChunker()
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200
    
    def test_initialization_with_custom_parameters(self):
        """Test that TextChunker accepts custom parameters."""
        chunker = TextChunker(chunk_size=500, chunk_overlap=100)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 100
    
    def test_initialization_with_invalid_chunk_size(self):
        """Test that negative or zero chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextChunker(chunk_size=0)
        
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextChunker(chunk_size=-100)
    
    def test_initialization_with_invalid_overlap(self):
        """Test that negative overlap raises ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            TextChunker(chunk_overlap=-50)
    
    def test_initialization_with_overlap_exceeding_chunk_size(self):
        """Test that overlap >= chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            TextChunker(chunk_size=100, chunk_overlap=100)
        
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            TextChunker(chunk_size=100, chunk_overlap=150)
    
    def test_count_tokens_with_simple_text(self):
        """Test token counting with simple text."""
        chunker = TextChunker()
        text = "Hello world"
        token_count = chunker.count_tokens(text)
        assert token_count > 0
        assert isinstance(token_count, int)
    
    def test_count_tokens_with_empty_string(self):
        """Test token counting with empty string."""
        chunker = TextChunker()
        assert chunker.count_tokens("") == 0
    
    def test_chunk_text_with_short_text(self):
        """Test chunking text shorter than chunk_size returns single chunk."""
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        text = "This is a short text that should fit in one chunk."
        
        chunks = chunker.chunk_text(text, "test.pdf", page_number=1)
        
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].document_name == "test.pdf"
        assert chunks[0].page_number == 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].token_count <= 1000
    
    def test_chunk_text_with_empty_string(self):
        """Test chunking empty string returns empty list."""
        chunker = TextChunker()
        chunks = chunker.chunk_text("", "test.pdf")
        assert chunks == []
    
    def test_chunk_text_with_whitespace_only(self):
        """Test chunking whitespace-only string returns empty list."""
        chunker = TextChunker()
        chunks = chunker.chunk_text("   \n\t  ", "test.pdf")
        assert chunks == []
    
    def test_chunk_text_preserves_metadata(self):
        """Test that chunks preserve document name and page number."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        # Create text long enough to generate multiple chunks
        text = " ".join(["word"] * 100)
        
        chunks = chunker.chunk_text(text, "medical_doc.pdf", page_number=5)
        
        for chunk in chunks:
            assert chunk.document_name == "medical_doc.pdf"
            assert chunk.page_number == 5
    
    def test_chunk_text_assigns_sequential_indices(self):
        """Test that chunks have sequential indices starting from 0."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = " ".join(["word"] * 100)
        
        chunks = chunker.chunk_text(text, "test.pdf")
        
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
    
    def test_chunk_text_respects_chunk_size_limit(self):
        """Test that no chunk exceeds the chunk_size token limit."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        # Create long text
        text = " ".join(["medical"] * 200)
        
        chunks = chunker.chunk_text(text, "test.pdf")
        
        for chunk in chunks:
            assert chunk.token_count <= 100
    
    def test_chunk_text_creates_multiple_chunks_for_long_text(self):
        """Test that long text is split into multiple chunks."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        # Create text that will require multiple chunks
        text = " ".join(["word"] * 200)
        
        chunks = chunker.chunk_text(text, "test.pdf")
        
        assert len(chunks) > 1
    
    def test_chunk_text_with_medical_terminology(self):
        """Test chunking preserves medical terminology."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "Carcinoma is a type of cancer. Chemotherapy and radiotherapy are common treatments."
        
        chunks = chunker.chunk_text(text, "oncology.pdf")
        
        # Should be single chunk since text is short
        assert len(chunks) == 1
        assert "Carcinoma" in chunks[0].text
        assert "Chemotherapy" in chunks[0].text
    
    def test_chunk_text_by_pages_with_single_page(self):
        """Test chunking by pages with a single page."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        pages = [("This is page one content.", 1)]
        
        chunks = chunker.chunk_text_by_pages(pages, "test.pdf")
        
        assert len(chunks) == 1
        assert chunks[0].page_number == 1
        assert chunks[0].chunk_index == 0
    
    def test_chunk_text_by_pages_with_multiple_pages(self):
        """Test chunking by pages preserves page numbers."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        pages = [
            ("Content from page one.", 1),
            ("Content from page two.", 2),
            ("Content from page three.", 3)
        ]
        
        chunks = chunker.chunk_text_by_pages(pages, "test.pdf")
        
        # Verify chunks exist
        assert len(chunks) > 0
        
        # Verify page numbers are preserved
        page_numbers = [chunk.page_number for chunk in chunks]
        assert 1 in page_numbers
        assert 2 in page_numbers
        assert 3 in page_numbers
    
    def test_chunk_text_by_pages_reindexes_globally(self):
        """Test that chunk_text_by_pages assigns global sequential indices."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        # Create pages with enough content to generate multiple chunks
        pages = [
            (" ".join(["word"] * 50), 1),
            (" ".join(["word"] * 50), 2)
        ]
        
        chunks = chunker.chunk_text_by_pages(pages, "test.pdf")
        
        # Verify indices are sequential across all pages
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
    
    def test_chunk_text_with_special_characters(self):
        """Test chunking preserves special characters and symbols."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "Dosage: 50mg/day. Temperature: 37°C. Survival rate: 85%."
        
        chunks = chunker.chunk_text(text, "test.pdf")
        
        assert len(chunks) == 1
        assert "50mg/day" in chunks[0].text
        assert "37°C" in chunks[0].text
        assert "85%" in chunks[0].text
    
    def test_chunk_text_token_count_accuracy(self):
        """Test that token_count field matches actual token count."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a test sentence for token counting."
        
        chunks = chunker.chunk_text(text, "test.pdf")
        
        for chunk in chunks:
            actual_count = chunker.count_tokens(chunk.text)
            assert chunk.token_count == actual_count
