"""
Lightweight PII detector — addresses OWASP LLM06 (Sensitive Information
Disclosure).

We use simple regex over high-precision patterns rather than a heavy NER
model. Intentional trade-off: fewer false positives, slightly lower
recall. Every rule has a comment explaining why it's there.

Returns BOTH the redacted text and a list of detection events so the
audit log can record what was scrubbed without persisting the raw value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
# Each tuple: (label, regex). Order matters — most specific first to avoid
# wider patterns swallowing narrower ones.
# Order matters: most specific patterns first so they consume the substring
# before a wider regex can match part of it.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # API-key shaped tokens (long alphanumeric, e.g. sk-..., AKIA..., gh_pat_)
    # Must run before PHONE so it doesn't get sliced by the digit-only patterns.
    ("API_KEY", re.compile(r"\b(?:sk|pk|api|gh|aws|AKIA)[_\-A-Za-z0-9]{16,}\b")),
    # Emails — RFC 5322 simplified
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    # Argentine CUIT/CUIL XX-XXXXXXXX-X — must run BEFORE DNI because the
    # 7–8-digit middle group would otherwise be eaten by DNI_AR.
    ("CUIT_AR", re.compile(r"\b\d{2}-\d{7,8}-\d\b")),
    # Argentine DNI (7-8 digits, optional dot separators)
    ("DNI_AR", re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}\b")),
    # Credit card-like (13-19 digits with optional dashes/spaces).
    # Luhn check is intentionally omitted to keep this dependency-free.
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ \-]?){13,19}\b")),
    # IPv4
    ("IPV4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    # Phone numbers — require a leading "+" so we don't catch random digit
    # runs already redacted by the patterns above.
    ("PHONE", re.compile(r"\+\d{1,3}[ \-.]?\(?\d{2,4}\)?[ \-.]?\d{3,4}[ \-.]?\d{3,4}")),
]


@dataclass
class RedactionResult:
    text: str
    detections: list[dict]   # [{"type": "EMAIL", "count": 2}, ...]

    @property
    def has_pii(self) -> bool:
        return bool(self.detections)


def redact(text: str) -> RedactionResult:
    """
    Replace PII matches with `[REDACTED:<TYPE>]` and return aggregate counts.

    Aggregate counts (not the raw values) are what should be logged.
    """
    if not text:
        return RedactionResult(text=text, detections=[])

    redacted = text
    counts: dict[str, int] = {}
    for label, pattern in _PATTERNS:
        def _sub(match: re.Match[str], lbl=label) -> str:
            counts[lbl] = counts.get(lbl, 0) + 1
            return f"[REDACTED:{lbl}]"

        redacted = pattern.sub(_sub, redacted)

    detections = [{"type": k, "count": v} for k, v in counts.items()]
    return RedactionResult(text=redacted, detections=detections)
