"""Pydantic schemas for the auth endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    full_name: str = Field("", max_length=255)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int


class UserPublic(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    role: str
    is_active: bool
    quota_queries_per_day: int
    queries_today: int
    created_at: datetime

    model_config = {"from_attributes": True}
