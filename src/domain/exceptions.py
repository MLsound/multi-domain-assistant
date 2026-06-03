"""
Domain-level exceptions.

These are transport-agnostic: the domain raises them, and the API adapter
(``src.api.main``) is solely responsible for translating each one into an
HTTP status code. Nothing here imports FastAPI.
"""

from __future__ import annotations


class DomainError(Exception):
    """Base class for all errors raised by the domain/service layer."""


class RateLimitExceededError(DomainError):
    """Burst rate limit (OWASP LLM04) tripped for the current user."""

    def __init__(self, retry_after: int) -> None:
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry in {retry_after}s.")


class QuotaExceededError(DomainError):
    """The user has consumed their daily query quota."""

    def __init__(self, message: str = "Daily query quota exceeded") -> None:
        super().__init__(message)


class QueryTimeoutError(DomainError):
    """The RAG pipeline did not finish within the allotted time budget."""

    def __init__(self, timeout_sec: float) -> None:
        self.timeout_sec = timeout_sec
        super().__init__(f"Request timed out after {int(timeout_sec)} seconds")


class QueryProcessingError(DomainError):
    """An unexpected failure occurred while processing the query."""
