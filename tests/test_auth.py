"""
Tests for the auth module — pure unit tests against an in-memory SQLite.

Coverage:
  - password hashing round-trip
  - JWT issuance / validation / expiry rejection
  - register endpoint (happy + duplicate)
  - login endpoint (happy + wrong password)
  - /auth/me requires a valid token
  - role hierarchy in require_role
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

# Force an isolated DB BEFORE importing the auth modules so they pick it up.
os.environ["AUTH_DATABASE_URL"] = "sqlite:///:memory:"
os.environ["AUTH_JWT_SECRET"] = "test-secret-do-not-use-in-prod-please"

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from jose import jwt
from sqlalchemy.pool import StaticPool

# --- Wire up an in-memory DB shared across the test session ---
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.auth import database as db_mod
from src.auth.database import Base
from src.auth.deps import get_current_user, require_role
from src.auth.router import router as auth_router
from src.auth.security import (
    JWT_ALGORITHM,
    JWT_SECRET,
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)

_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
_SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
db_mod.engine = _engine
db_mod.SessionLocal = _SessionLocal


def _override_get_db():
    s = _SessionLocal()
    try:
        yield s
    finally:
        s.close()


db_mod.get_db = _override_get_db  # so models that import get_db see the test session


from src.auth.models import User  # noqa: E402  (after override)

Base.metadata.create_all(bind=_engine)


@pytest.fixture
def app() -> FastAPI:
    a = FastAPI()
    a.include_router(auth_router)
    a.dependency_overrides[db_mod.get_db] = _override_get_db
    return a


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# password hashing
# ---------------------------------------------------------------------------

def test_password_hash_roundtrip():
    h = hash_password("CorrectHorseBatteryStaple")
    assert h != "CorrectHorseBatteryStaple"
    assert verify_password("CorrectHorseBatteryStaple", h)
    assert not verify_password("wrong", h)


# ---------------------------------------------------------------------------
# JWT
# ---------------------------------------------------------------------------

def test_jwt_roundtrip():
    tok = create_access_token("alice@example.com", "user")
    decoded = decode_token(tok)
    assert decoded is not None
    assert decoded["sub"] == "alice@example.com"
    assert decoded["role"] == "user"


def test_jwt_invalid_returns_none():
    assert decode_token("not.a.token") is None


def test_jwt_expired_rejected():
    payload = {
        "sub": "x@y.z",
        "role": "user",
        "iat": int(datetime.now(timezone.utc).timestamp()) - 7200,
        "exp": int((datetime.now(timezone.utc) - timedelta(hours=1)).timestamp()),
    }
    tok = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    assert decode_token(tok) is None


# ---------------------------------------------------------------------------
# /auth/register and /auth/login
# ---------------------------------------------------------------------------

def test_register_login_me_flow(client: TestClient):
    # Register
    r = client.post(
        "/auth/register",
        json={"email": "jorge@fiuba.ar", "password": "Strong-Pass-1", "full_name": "Jorge Cuenca"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["email"] == "jorge@fiuba.ar"
    assert body["role"] == "user"

    # Duplicate registration is rejected
    r2 = client.post(
        "/auth/register",
        json={"email": "jorge@fiuba.ar", "password": "Strong-Pass-1"},
    )
    assert r2.status_code == 400

    # Login OK
    r3 = client.post(
        "/auth/login",
        json={"email": "jorge@fiuba.ar", "password": "Strong-Pass-1"},
    )
    assert r3.status_code == 200, r3.text
    token = r3.json()["access_token"]
    assert token

    # Login with wrong password
    r4 = client.post(
        "/auth/login",
        json={"email": "jorge@fiuba.ar", "password": "wrong"},
    )
    assert r4.status_code == 401

    # /auth/me with a valid bearer
    r5 = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r5.status_code == 200, r5.text
    me = r5.json()
    assert me["email"] == "jorge@fiuba.ar"


def test_me_without_token_is_401(client: TestClient):
    r = client.get("/auth/me")
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# require_role
# ---------------------------------------------------------------------------

def test_require_role_blocks_lower_role():
    fake_user = User(id=1, email="x@y.z", full_name="x", hashed_password="h", role="user")
    checker = require_role("admin")
    with pytest.raises(HTTPException) as exc:
        checker(user=fake_user)
    assert exc.value.status_code == 403


def test_require_role_allows_higher_role():
    fake_admin = User(id=2, email="a@b.c", full_name="a", hashed_password="h", role="admin")
    checker = require_role("researcher")
    assert checker(user=fake_admin) is fake_admin
