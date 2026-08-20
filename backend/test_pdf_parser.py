"""
Test script for PDF parser module.

Tests the PDF parser with the provided medical documents.
"""

import os
from pdf_parser import PDFParser, PDFParserError


def test_parser_with_medical_docs():
    """Test PDF parser with the medical documents in backend directory."""
    parser = PDFParser()
    
    # List of medical documents to test
    documents = [
        "breast_cancer.pdf.docx",
        "prevention.pdf.pdf",
        "screening.pdf.docx",
        "world_cancer.pdf.pdf"
    ]
    
    print("Testing PDF Parser with Medical Documents")
    print("=" * 60)
    
    for doc_name in documents:
        doc_path = doc_name
        
        if not os.path.exists(doc_path):
            print(f"\n❌ {doc_name}: File not found")
            continue
        
        try:
            print(f"\n📄 Processing: {doc_name}")
            
            # Read file
            with open(doc_path, 'rb') as f:
                file_bytes = f.read()
            
            # Parse PDF
            text = parser.parse_pdf(file_bytes, filename=doc_name)
            
            # Display results
            text_length = len(text)
            word_count = len(text.split())
            lines = text.split('\n')
            non_empty_lines = [l for l in lines if l.strip()]
            
            print(f"   ✓ Successfully extracted text")
            print(f"   - Text length: {text_length:,} characters")
            print(f"   - Word count: {word_count:,} words")
            print(f"   - Lines: {len(non_empty_lines):,} non-empty lines")
            print(f"   - Preview (first 200 chars):")
            print(f"     {text[:200].replace(chr(10), ' ')[:200]}...")
            
        except PDFParserError as e:
            print(f"   ❌ Parser error: {e}")
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
    
    print("\n" + "=" * 60)
    print("Testing complete!")


if __name__ == "__main__":
    test_parser_with_medical_docs()
