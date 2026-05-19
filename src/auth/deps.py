"""
FastAPI dependencies for protected endpoints.

`get_current_user` decodes the bearer JWT and loads the active User.
`require_role` returns a dependency that enforces a minimum role.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from src.auth.database import get_db
from src.auth.models import User
from src.auth.security import decode_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def get_current_user(
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.email == payload["sub"]).one_or_none()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")

    # Daily quota window: rolling 24h.
    now = datetime.now(timezone.utc)
    quota_reset = user.quota_reset_at
    if quota_reset.tzinfo is None:
        quota_reset = quota_reset.replace(tzinfo=timezone.utc)
    if (now - quota_reset).total_seconds() > 86400:
        user.queries_today = 0
        user.quota_reset_at = now
        db.commit()

    return user


def require_role(min_role: str):
    """Role hierarchy: user < researcher < admin."""
    order = {"user": 0, "researcher": 1, "admin": 2}

    def _checker(user: User = Depends(get_current_user)) -> User:
        if order.get(user.role, 0) < order.get(min_role, 99):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return user

    return _checker
