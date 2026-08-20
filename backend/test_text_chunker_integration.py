"""
Integration tests for text chunking with realistic medical content.

Tests the chunker with actual medical text to verify it works correctly
in real-world scenarios.
"""

import pytest
from text_chunker import TextChunker


class TestTextChunkerIntegration:
    """Integration tests with realistic medical text."""
    
    def test_chunk_medical_text_realistic(self):
        """Test chunking with realistic medical text about cancer."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        
        medical_text = """
        Cancer is a group of diseases involving abnormal cell growth with the potential 
        to invade or spread to other parts of the body. These contrast with benign tumors, 
        which do not spread. Possible signs and symptoms include a lump, abnormal bleeding, 
        prolonged cough, unexplained weight loss, and a change in bowel movements. While 
        these symptoms may indicate cancer, they can also have other causes. Over 100 types 
        of cancers affect humans.
        
        Tobacco use is the cause of about 22% of cancer deaths. Another 10% are due to 
        obesity, poor diet, lack of physical activity or excessive drinking of alcohol. 
        Other factors include certain infections, exposure to ionizing radiation, and 
        environmental pollutants. In the developing world, 15% of cancers are due to 
        infections such as Helicobacter pylori, hepatitis B, hepatitis C, human 
        papillomavirus infection, Epstein-Barr virus and human immunodeficiency virus (HIV).
        """
        
        chunks = chunker.chunk_text(medical_text, "cancer_overview.pdf", page_number=1)
        
        # Verify chunks were created
        assert len(chunks) > 0
        
        # Verify all chunks respect token limit
        for chunk in chunks:
            assert chunk.token_count <= 100
            assert chunk.token_count > 0
        
        # Verify metadata is preserved
        for chunk in chunks:
            assert chunk.document_name == "cancer_overview.pdf"
            assert chunk.page_number == 1
        
        # Verify sequential indexing
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
        
        # Verify medical terms are preserved
        full_text = " ".join([chunk.text for chunk in chunks])
        assert "cancer" in full_text.lower()
        assert "tumor" in full_text.lower() or "tumors" in full_text.lower()
    
    def test_chunk_long_medical_document(self):
        """Test chunking a longer medical document."""
        chunker = TextChunker(chunk_size=200, chunk_overlap=50)
        
        long_text = """
        Breast cancer is cancer that develops from breast tissue. Signs of breast cancer 
        may include a lump in the breast, a change in breast shape, dimpling of the skin, 
        fluid coming from the nipple, a newly inverted nipple, or a red or scaly patch of skin.
        
        In those with distant spread of the disease, there may be bone pain, swollen lymph nodes, 
        shortness of breath, or yellow skin. Risk factors for developing breast cancer include 
        being female, obesity, lack of physical exercise, drinking alcohol, hormone replacement 
        therapy during menopause, ionizing radiation, early age at first menstruation, having 
        children late or not at all, older age, and family history.
        
        About 5–10% of cases are due to genes inherited from a person's parents, including BRCA1 
        and BRCA2 among others. Breast cancer most commonly develops in cells from the lining of 
        milk ducts and the lobules that supply the ducts with milk. Cancers developing from the 
        ducts are known as ductal carcinomas, while those developing from lobules are known as 
        lobular carcinomas.
        
        Screening is recommended in those between 50 and 75 years of age. The benefits versus 
        harms of screening in those less than 50 years old are not clear. Screening is typically 
        done using mammography. In those at high risk, magnetic resonance imaging (MRI) may be 
        recommended. If a lump is found, a biopsy is typically done to confirm the diagnosis.
        """ * 2  # Repeat to make it longer
        
        chunks = chunker.chunk_text(long_text, "breast_cancer.pdf", page_number=3)
        
        # Should create multiple chunks
        assert len(chunks) > 2
        
        # All chunks should respect limits
        for chunk in chunks:
            assert chunk.token_count <= 200
        
        # Verify overlap exists between consecutive chunks
        # (This is a basic check - the property test will verify exact overlap)
        if len(chunks) > 1:
            # Check that consecutive chunks share some content
            for i in range(len(chunks) - 1):
                # At least some words should overlap
                words1 = set(chunks[i].text.split())
                words2 = set(chunks[i + 1].text.split())
                overlap_words = words1.intersection(words2)
                assert len(overlap_words) > 0, "Consecutive chunks should have overlapping content"
    
    def test_chunk_with_abbreviations_and_numbers(self):
        """Test chunking preserves medical abbreviations and numerical data."""
        chunker = TextChunker(chunk_size=150, chunk_overlap=30)
        
        text = """
        The patient's CBC showed WBC 12.5 x10^9/L, RBC 4.2 x10^12/L, and Hgb 13.5 g/dL.
        CT scan revealed a 3.2cm mass in the right upper lobe. PET-CT showed SUV max of 8.5.
        Biopsy confirmed adenocarcinoma, stage IIIA (T2N2M0). Treatment plan includes 
        4 cycles of cisplatin/pemetrexed followed by radiation therapy (60 Gy in 30 fractions).
        EGFR mutation testing was negative. PD-L1 expression was 45%.
        """
        
        chunks = chunker.chunk_text(text, "patient_report.pdf", page_number=2)
        
        # Combine all chunks to verify preservation
        combined_text = " ".join([chunk.text for chunk in chunks])
        
        # Verify abbreviations are preserved
        assert "WBC" in combined_text or "wbc" in combined_text.lower()
        assert "CT" in combined_text or "ct" in combined_text.lower()
        
        # Verify numbers are preserved (at least some of them)
        assert any(char.isdigit() for char in combined_text)
    
    def test_chunk_empty_and_whitespace_handling(self):
        """Test that empty strings and whitespace are handled correctly."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        
        # Empty string
        chunks = chunker.chunk_text("", "empty.pdf")
        assert chunks == []
        
        # Only whitespace
        chunks = chunker.chunk_text("   \n\n\t  ", "whitespace.pdf")
        assert chunks == []
        
        # Text with lots of whitespace
        text = "Cancer    is    a    disease.    \n\n\n    Treatment    is    available."
        chunks = chunker.chunk_text(text, "spaced.pdf")
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.text.strip()) > 0
    
    def test_chunk_by_pages_integration(self):
        """Test chunking multiple pages with realistic content."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        
        pages = [
            ("Lung cancer is the leading cause of cancer death worldwide.", 1),
            ("Symptoms include persistent cough, chest pain, and shortness of breath.", 2),
            ("Diagnosis typically involves imaging studies and tissue biopsy.", 3),
            ("Treatment options include surgery, chemotherapy, radiation, and immunotherapy.", 4)
        ]
        
        chunks = chunker.chunk_text_by_pages(pages, "lung_cancer_guide.pdf")
        
        # Should have chunks from all pages
        assert len(chunks) >= 4
        
        # Verify page numbers are preserved
        page_numbers = [chunk.page_number for chunk in chunks]
        assert 1 in page_numbers
        assert 2 in page_numbers
        assert 3 in page_numbers
        assert 4 in page_numbers
        
        # Verify global indexing
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
        
        # Verify all chunks have same document name
        for chunk in chunks:
            assert chunk.document_name == "lung_cancer_guide.pdf"
