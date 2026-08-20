"""
PDF and document parser module for Medical Cancer RAG Chatbot.

This module handles text extraction from PDF documents with support for:
- Text-based PDFs using PyPDF2
- Complex layouts using PyMuPDF (fitz) as fallback
- Multi-column layouts with paragraph boundary preservation
- Whitespace normalization while preserving medical terminology
- Error handling for encrypted/password-protected PDFs
"""

import re
from typing import Tuple, Optional
from pathlib import Path
import PyPDF2
import pdfplumber


class PDFParserError(Exception):
    """Custom exception for PDF parsing errors."""
    pass


class PDFParser:
    """
    Parser for extracting text from PDF documents.
    
    Handles various PDF formats and layouts while preserving medical terminology
    and document structure.
    """
    
    def __init__(self):
        """Initialize the PDF parser."""
        pass
    
    def parse_pdf(self, pdf_file: bytes, filename: str = "") -> str:
        """
        Extract text from PDF file.
        
        Args:
            pdf_file: PDF file content as bytes
            filename: Optional filename for error messages
            
        Returns:
            Extracted and normalized text content
            
        Raises:
            PDFParserError: If PDF cannot be parsed (encrypted, corrupted, etc.)
        """
        # Try PyPDF2 first (faster for simple PDFs)
        try:
            text = self._extract_with_pypdf2(pdf_file)
            if text and len(text.strip()) > 100:  # Reasonable content extracted
                return self._normalize_text(text)
        except Exception as e:
            # Fall through to pdfplumber
            pass
        
        # Try pdfplumber as fallback (better for complex layouts)
        try:
            text = self._extract_with_pdfplumber(pdf_file)
            if text and len(text.strip()) > 0:
                return self._normalize_text(text)
            else:
                raise PDFParserError(f"No text content could be extracted from PDF{f' {filename}' if filename else ''}")
        except PDFParserError:
            raise
        except Exception as e:
            raise PDFParserError(f"Failed to parse PDF{f' {filename}' if filename else ''}: {str(e)}")
    
    def _extract_with_pypdf2(self, pdf_file: bytes) -> str:
        """
        Extract text using PyPDF2.
        
        Args:
            pdf_file: PDF file content as bytes
            
        Returns:
            Extracted text
            
        Raises:
            PDFParserError: If PDF is encrypted or cannot be read
        """
        from io import BytesIO
        
        pdf_stream = BytesIO(pdf_file)
        reader = PyPDF2.PdfReader(pdf_stream)
        
        # Check if PDF is encrypted
        if reader.is_encrypted:
            raise PDFParserError("PDF is password-protected or encrypted and cannot be processed")
        
        # Extract text from all pages
        text_parts = []
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            except Exception as e:
                # Continue with other pages if one fails
                continue
        
        return "\n\n".join(text_parts)
    
    def _extract_with_pdfplumber(self, pdf_file: bytes) -> str:
        """
        Extract text using pdfplumber with better layout handling.
        
        Args:
            pdf_file: PDF file content as bytes
            
        Returns:
            Extracted text with preserved structure
            
        Raises:
            PDFParserError: If PDF is encrypted or cannot be read
        """
        from io import BytesIO
        
        pdf_stream = BytesIO(pdf_file)
        
        try:
            with pdfplumber.open(pdf_stream) as pdf:
                text_parts = []
                
                for page in pdf.pages:
                    # Extract text with layout preservation
                    page_text = page.extract_text()
                    
                    if page_text and page_text.strip():
                        text_parts.append(page_text.strip())
                
                return "\n\n".join(text_parts)
                
        except Exception as e:
            if "password" in str(e).lower() or "encrypted" in str(e).lower():
                raise PDFParserError("PDF is password-protected or encrypted and cannot be processed")
            raise
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize extracted text while preserving medical terminology.
        
        Performs:
        - Excessive whitespace removal (multiple spaces, tabs)
        - Excessive newline normalization (more than 2 consecutive)
        - Preserves single spaces, medical abbreviations, special characters
        
        Args:
            text: Raw extracted text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        
        # Normalize multiple spaces to single space (but preserve medical terms)
        # This regex replaces 2+ spaces with a single space
        text = re.sub(r' {2,}', ' ', text)
        
        # Normalize excessive newlines (more than 2 consecutive) to 2 newlines
        # This preserves paragraph boundaries (double newline) but removes excessive spacing
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove spaces at the beginning and end of lines
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)
        
        # Remove any leading/trailing whitespace from the entire text
        text = text.strip()
        
        return text
    
    def parse_file(self, file_path: str) -> str:
        """
        Parse a PDF file from disk.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted and normalized text
            
        Raises:
            PDFParserError: If file cannot be read or parsed
        """
        try:
            with open(file_path, 'rb') as f:
                pdf_bytes = f.read()
            return self.parse_pdf(pdf_bytes, filename=Path(file_path).name)
        except PDFParserError:
            raise
        except Exception as e:
            raise PDFParserError(f"Failed to read file {file_path}: {str(e)}")


def parse_pdf_file(file_path: str) -> str:
    """
    Convenience function to parse a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted and normalized text
        
    Raises:
        PDFParserError: If file cannot be parsed
    """
    parser = PDFParser()
    return parser.parse_file(file_path)


def parse_pdf_bytes(pdf_bytes: bytes, filename: str = "") -> str:
    """
    Convenience function to parse PDF from bytes.
    
    Args:
        pdf_bytes: PDF file content as bytes
        filename: Optional filename for error messages
        
    Returns:
        Extracted and normalized text
        
    Raises:
        PDFParserError: If PDF cannot be parsed
    """
    parser = PDFParser()
    return parser.parse_pdf(pdf_bytes, filename=filename)
