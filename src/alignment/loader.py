"""
Loader for the constitutional principles YAML.

Caches the parsed principles so the Critic Agent can read them without
hitting disk on every call.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).parent / "constitution.yaml"


@dataclass(frozen=True)
class Principle:
    id: str
    name: str
    rule: str
    weight: float


@lru_cache(maxsize=1)
def load_principles(path: Path = DEFAULT_PATH) -> tuple[Principle, ...]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return tuple(
        Principle(
            id=p["id"],
            name=p["name"],
            rule=" ".join(p["rule"].split()),
            weight=float(p.get("weight", 1.0)),
        )
        for p in raw.get("principles", [])
    )


def constitution_for_prompt() -> str:
    """Compact form for inclusion in critic / synthesis prompts."""
    parts = [f"- [{p.id}] {p.name}: {p.rule}" for p in load_principles()]
    return "\n".join(parts)
