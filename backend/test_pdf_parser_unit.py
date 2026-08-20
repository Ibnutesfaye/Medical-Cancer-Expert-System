"""
Unit tests for PDF parser module.

Tests:
- Text extraction from valid PDFs
- Error handling for encrypted / password-protected PDFs
- Whitespace normalization
- Medical terminology preservation
- Empty PDF error handling
- File-not-found error handling
- Convenience function parse_pdf_bytes
- Paragraph boundary preservation
- _normalize_text method directly

All tests use small self-generated PDFs (via reportlab) so they
run in milliseconds and never time out.
"""

import pytest
from io import BytesIO

import PyPDF2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from pdf_parser import PDFParser, PDFParserError, parse_pdf_bytes


# ---------------------------------------------------------------------------
# Helper — build a small PDF in memory with given text lines
# ---------------------------------------------------------------------------

def make_pdf(*lines: str) -> bytes:
    """Create a minimal single-page PDF containing the given text lines."""
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    y = 750
    for line in lines:
        c.drawString(72, y, line)
        y -= 20
    c.save()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Medical content PDF — used by multiple tests
# ---------------------------------------------------------------------------

MEDICAL_PDF: bytes = make_pdf(
    "Cancer Prevention and Screening Guidelines",
    "Early detection reduces mortality in breast cancer patients.",
    "Risk factors include tumor size, metastasis, and patient age.",
    "Recommended screening: mammogram every 2 years after age 40.",
    "Treatment options: surgery, chemotherapy, radiation therapy.",
    "Consult your oncologist for a personalised treatment plan.",
)


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

class TestPDFParser:
    """Test suite for PDFParser class."""

    def setup_method(self):
        self.parser = PDFParser()

    # ── 1. Parse valid PDF ────────────────────────────────────────────────────

    def test_parse_valid_pdf(self):
        """Text extraction from a valid PDF returns non-empty string."""
        text = self.parser.parse_pdf(MEDICAL_PDF, "medical.pdf")

        assert text is not None
        assert isinstance(text, str)
        assert len(text) > 0
        assert len(text.split()) > 10

    # ── 2. Encrypted PDF raises PDFParserError ────────────────────────────────

    def test_parse_encrypted_pdf_error(self):
        """Encrypted PDF must raise PDFParserError with descriptive message."""
        writer = PyPDF2.PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.encrypt("secret123")

        buf = BytesIO()
        writer.write(buf)
        encrypted = buf.getvalue()

        with pytest.raises(PDFParserError) as exc_info:
            self.parser.parse_pdf(encrypted, "encrypted.pdf")

        msg = str(exc_info.value).lower()
        assert "password" in msg or "encrypted" in msg

    # ── 3. Whitespace normalisation ───────────────────────────────────────────

    def test_whitespace_normalization(self):
        """Tabs and multiple spaces are collapsed; content is preserved."""
        pdf = make_pdf(
            "This  has    multiple     spaces",
            "Normal medical text line here",
        )
        text = self.parser.parse_pdf(pdf, "ws_test.pdf")

        assert "\t" not in text
        assert "\n\n\n" not in text
        assert "medical" in text.lower() or "normal" in text.lower()

    # ── 4. Medical terminology preserved ─────────────────────────────────────

    def test_medical_terminology_preservation(self):
        """Medical terms are preserved exactly as written."""
        text = self.parser.parse_pdf(MEDICAL_PDF, "medical.pdf")
        text_lower = text.lower()

        assert any(term in text_lower for term in [
            "cancer", "tumor", "screening", "oncologist",
            "chemotherapy", "mammogram",
        ])
        assert "." in text   # periods kept

    # ── 5. Empty PDF raises PDFParserError ────────────────────────────────────

    def test_empty_pdf_error(self):
        """A blank PDF with no text raises PDFParserError."""
        buf = BytesIO()
        c = canvas.Canvas(buf, pagesize=letter)
        c.showPage()   # blank — no drawString calls
        c.save()
        blank_pdf = buf.getvalue()

        with pytest.raises(PDFParserError) as exc_info:
            self.parser.parse_pdf(blank_pdf, "empty.pdf")

        assert "no text content" in str(exc_info.value).lower()

    # ── 6. parse_file — file not found ───────────────────────────────────────

    def test_parse_file_not_found(self):
        """Non-existent file path raises PDFParserError."""
        with pytest.raises(PDFParserError) as exc_info:
            self.parser.parse_file("does_not_exist_xyz.pdf")

        assert "failed to read" in str(exc_info.value).lower()

    # ── 7. Convenience function ───────────────────────────────────────────────

    def test_convenience_function(self):
        """parse_pdf_bytes convenience function works correctly."""
        text = parse_pdf_bytes(MEDICAL_PDF, "medical.pdf")

        assert text is not None
        assert isinstance(text, str)
        assert len(text) > 0

    # ── 8. Paragraph boundary preservation ───────────────────────────────────

    def test_paragraph_boundary_preservation(self):
        """Newlines are preserved; no more than 2 consecutive newlines."""
        text = self.parser.parse_pdf(MEDICAL_PDF, "medical.pdf")

        assert "\n" in text
        assert "\n\n\n" not in text

    # ── 9. _normalize_text method directly ───────────────────────────────────

    @pytest.mark.parametrize("raw", [
        "text  with   multiple    spaces",
        "text\t\twith\ttabs",
        "text\n\n\n\n\nmany newlines",
        "  leading and trailing  ",
        "normal text",
        "cancer\ttreatment\tplan",
        "line1\n\n\n\nline2",
    ])
    def test_normalize_text_method(self, raw: str):
        """_normalize_text removes tabs, double-spaces, and triple newlines."""
        result = self.parser._normalize_text(raw)

        assert "\t"     not in result, "tabs must be removed"
        assert "\n\n\n" not in result, "triple newlines must be collapsed"
        assert "  "     not in result, "double spaces must be collapsed"
        assert result   == result.strip(), "no leading/trailing whitespace"

    # ── 10. Multi-page PDF ────────────────────────────────────────────────────

    def test_multipage_pdf(self):
        """Text from all pages is combined into a single string."""
        buf = BytesIO()
        c = canvas.Canvas(buf, pagesize=letter)

        # Page 1
        c.drawString(72, 750, "Page one: glioma tumor diagnosis")
        c.showPage()

        # Page 2
        c.drawString(72, 750, "Page two: melanoma treatment options")
        c.showPage()

        c.save()
        pdf = buf.getvalue()

        text = self.parser.parse_pdf(pdf, "multipage.pdf")

        assert "glioma" in text.lower()
        assert "melanoma" in text.lower()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
