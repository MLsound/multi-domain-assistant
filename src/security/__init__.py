"""
Security utilities — own implementation of guardrails the project needs:
PII detection, rate limiting, canary tokens, prompt-injection scoring.

Deliberately NOT using LLM Guard / Rebuff / NeMo Guardrails as black boxes.
Each module is small, readable and defensible in the oral defense.
"""
