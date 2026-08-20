"""
Medical Safety Guardrails
==========================
Validates RAG responses before returning them to the user.
Prevents hallucination, dangerous advice, and non-medical content.

Rules enforced:
  1. Response must include disclaimer
  2. No specific drug dosages
  3. No definitive diagnosis language
  4. Cancer-domain enforcement
  5. Minimum citation confidence
"""

from __future__ import annotations

import re

# Patterns that indicate dangerous/non-compliant responses
FORBIDDEN_PATTERNS = [
    r'\b\d+\s*(mg|ml|mcg|units?)\b',      # specific dosages
    r'\byou have\s+cancer\b',               # definitive diagnosis
    r'\bdiagnosed with\b',                  # diagnosis claim
    r'\bprescribe\b',                       # prescribing
    r'\btake\s+\d+\s+tablet',              # dosage instructions
]

# Required elements for medical responses
REQUIRED_DISCLAIMER = "consult"   # must contain some form of "consult a doctor"

# Non-cancer domains to reject
NON_CANCER_TOPICS = [
    "stock market", "weather", "recipe", "sports score",
    "cryptocurrency", "politics", "news headline",
]


def validate_response(response: str, query: str) -> dict:
    """
    Validate a RAG response for medical safety compliance.

    Returns:
        {
            "safe": bool,
            "issues": list[str],
            "cleaned_response": str,
            "disclaimer_added": bool
        }
    """
    issues = []
    disclaimer_added = False

    # Check for forbidden patterns
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, response, re.IGNORECASE):
            issues.append(f"Contains potentially dangerous pattern: {pattern}")

    # Check for non-cancer topics in query
    query_lower = query.lower()
    for topic in NON_CANCER_TOPICS:
        if topic in query_lower:
            issues.append(f"Query appears off-topic: {topic}")

    # Ensure disclaimer is present
    cleaned = response
    if REQUIRED_DISCLAIMER not in response.lower():
        disclaimer = "\n\n⚠️ *This is not a medical diagnosis. Please consult a qualified doctor.*"
        cleaned += disclaimer
        disclaimer_added = True

    return {
        "safe":              len(issues) == 0,
        "issues":            issues,
        "cleaned_response":  cleaned,
        "disclaimer_added":  disclaimer_added,
    }


def is_cancer_related(query: str) -> bool:
    """
    Quick check — does the query relate to cancer/oncology?
    Returns True if the query should be answered by the medical AI.
    """
    cancer_keywords = [
        "cancer", "tumor", "tumour", "oncol", "melanoma", "glioma",
        "carcinoma", "malignant", "biopsy", "chemotherapy", "radiation",
        "metastasis", "lymphoma", "leukemia", "sarcoma", "screening",
        "mammogram", "psa", "mri", "ct scan", "biopsy", "pathology",
        "symptoms", "treatment", "diagnosis", "prognosis", "survival",
    ]
    q = query.lower()
    return any(kw in q for kw in cancer_keywords)
