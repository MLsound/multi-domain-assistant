"""
SQLAlchemy engine + session factory.

SQLite is used for development to keep the project self-contained.
The same code works against PostgreSQL by changing AUTH_DATABASE_URL.
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DEFAULT_DB_PATH = Path("data") / "auth.db"
DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("AUTH_DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

# check_same_thread=False is required for SQLite to work with FastAPI's
# threadpool execution model.
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    """Common declarative base for all auth ORM models."""


def get_db() -> Session:
    """FastAPI dependency that yields a request-scoped DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables. Called at app startup."""
    # Import models so they register with Base.metadata before create_all
    from src.auth import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
