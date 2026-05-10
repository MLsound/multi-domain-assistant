"""
Heuristic prompt-injection scorer — own implementation.

Why heuristic and not a classifier model: the assignment forbids
plug-and-play frameworks; a small set of well-chosen rules is easier to
defend, cheaper to run, and produces interpretable explanations
("matched pattern X with weight Y").

Score is in [0, 1]:
  0.00–0.40  → benign
  0.40–0.70  → suspicious (allow but flag)
  0.70–1.00  → block

Each pattern carries an evidence weight; the final score is
1 - exp(-Σ weights), saturating asymptotically at 1.

This implements two slides from class 2:
  - "Detección de intentos de inyección" (slide 26)
  - "Hardening del sistema / Context anchoring" (slide 25)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Each rule: (regex, weight, label).
# Weights are calibrated so that any single high-confidence match (~1.2)
# already crosses the block threshold; ambiguous matches stay below it.
_RULES: list[tuple[re.Pattern[str], float, str]] = [
    # Direct override attempts — slide 20 ("Sobrescribir instrucciones")
    # Weight ≥ 1.5 so a SINGLE match crosses the 0.7 block threshold:
    # s = 1 - exp(-1.5) ≈ 0.78 → block.
    (re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts?|rules?)", re.I),
     1.6, "OVERRIDE_INSTRUCTIONS"),
    (re.compile(r"olvida(?:te|t[ée])?\s+(?:de\s+)?(?:todas?\s+)?(?:las\s+)?(?:reglas?|instrucciones)", re.I),
     1.6, "OVERRIDE_INSTRUCTIONS_ES"),
    (re.compile(r"ignora\s+(?:todas?\s+)?(?:las\s+)?(?:reglas?|instrucciones)", re.I),
     1.6, "OVERRIDE_INSTRUCTIONS_ES2"),
    (re.compile(r"disregard\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions)", re.I),
     1.6, "DISREGARD_PROMPT"),

    # Role / persona swap — slide 22 ("Doble personaje", "Virtualización")
    (re.compile(r"\b(?:dan|do anything now)\b", re.I), 1.6, "DAN_JAILBREAK"),
    (re.compile(r"you\s+are\s+now\s+(?:a\s+)?(?:dan|jailbreak|unrestricted|developer\s+mode)", re.I),
     1.6, "ROLE_SWAP"),
    (re.compile(r"modo\s+(?:desarrollador|administrador|sin\s+restricciones)", re.I),
     1.5, "ROLE_SWAP_ES"),

    # System prompt extraction — slide 22 ("Manipulación de instrucciones")
    (re.compile(r"reveal\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions)", re.I),
     1.5, "PROMPT_LEAK"),
    (re.compile(r"(?:repeat|print|output|show)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions)", re.I),
     1.5, "PROMPT_LEAK_2"),
    (re.compile(r"muestra(?:me)?\s+(?:tus?\s+)?instrucciones?", re.I),
     1.5, "PROMPT_LEAK_ES"),

    # Secret extraction — overlaps LLM06
    (re.compile(r"output\s+your\s+(?:api[_\s-]?key|secret|token|password)", re.I),
     1.6, "SECRET_EXFIL"),

    # Encoding-based obfuscation — slide 22 ("Ofuscación")
    (re.compile(r"\bbase64\b.*(?:decode|run|execute)", re.I), 0.6, "BASE64_OBFUSCATION"),

    # Suffix-style adversarial markers — slide 33
    (re.compile(r"[!@#$%^&*()_+={}\[\]|\\:;\"'<>,.?/~`]{8,}"), 0.4, "ADVERSARIAL_NOISE"),

    # Indirect-injection markers commonly found in poisoned documents
    (re.compile(r"\bSYSTEM\s*[:>]\s*", re.I), 0.5, "INLINE_SYSTEM_TAG"),
    (re.compile(r"<\s*\|?\s*system\s*\|?\s*>", re.I), 0.6, "SYSTEM_TAG"),
]


@dataclass
class InjectionVerdict:
    score: float                 # in [0, 1]
    decision: str                # "allow" | "flag" | "block"
    matched_rules: list[str]     # labels of matched rules

    @property
    def safe(self) -> bool:
        return self.decision != "block"


def score(text: str, *, block_threshold: float = 0.7, flag_threshold: float = 0.4) -> InjectionVerdict:
    if not text:
        return InjectionVerdict(score=0.0, decision="allow", matched_rules=[])

    matched: list[str] = []
    weight_sum = 0.0
    for pattern, weight, label in _RULES:
        if pattern.search(text):
            matched.append(label)
            weight_sum += weight

    # Saturating sum → score in [0, 1)
    s = 1.0 - math.exp(-weight_sum) if weight_sum > 0 else 0.0

    if s >= block_threshold:
        decision = "block"
    elif s >= flag_threshold:
        decision = "flag"
    else:
        decision = "allow"

    return InjectionVerdict(score=round(s, 4), decision=decision, matched_rules=matched)
