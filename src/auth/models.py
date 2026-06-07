"""
ORM models for authentication and per-user state.

User: registered account with hashed password.
QueryRecord: persistent per-user query log (complements the JSONL audit log).
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.auth.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), default="")
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), default="user")  # user | researcher | admin
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    quota_queries_per_day: Mapped[int] = mapped_column(Integer, default=200)
    queries_today: Mapped[int] = mapped_column(Integer, default=0)
    quota_reset_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    queries: Mapped[list["QueryRecord"]] = relationship(
        "QueryRecord", back_populates="user", cascade="all, delete-orphan"
    )


class QueryRecord(Base):
    __tablename__ = "query_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    session_id: Mapped[str] = mapped_column(String(64), default="")
    query: Mapped[str] = mapped_column(Text, nullable=False)
    response_preview: Mapped[str] = mapped_column(Text, default="")
    dominant_category: Mapped[str] = mapped_column(String(64), default="")
    confidence: Mapped[float] = mapped_column(default=0.0)
    latency_ms: Mapped[float] = mapped_column(default=0.0)
    blocked_by_guard: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    user: Mapped["User"] = relationship("User", back_populates="queries")
