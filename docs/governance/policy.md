# AI Governance Policy — Knowledge Assistant

**Owner:** Grupo 1 (Sarmiento, Lloveras, Cuenca) · FIUBA, PNL III, 2026
**Status:** v1.0 — operational
**Framework references:** NIST AI RMF 1.0 (GOVERN), EU AI Act (limited risk), OWASP Top-10 LLM 2025

---

## 1. Purpose

This document operationalises the **GOVERN** function of NIST AI RMF for the Knowledge Assistant. It declares roles, accountability, retention, and decision rules for the AI system in production-equivalent operation.

## 2. Roles & accountability

| Role | Person | Responsibility |
|---|---|---|
| Product owner | F. Sarmiento | Scope, acceptance, stakeholder communication |
| Tech lead / safety officer | A. Lloveras | Architecture, security review, incident command |
| ML / data steward | J. Cuenca | Corpus curation, evaluation, model cards |
| Faculty oversight | Mg. O. Bokhonok / Esp. A. Rodriguez | Sign-off on academic deliverable |

Every change that affects safety controls (Guard, Critic, constitution) requires review by the safety officer.

## 3. RBAC

The system enforces three roles, hierarchically ordered:

| Role | Default quota / day | Access |
|---|---:|---|
| `user` | 200 queries | `/query`, `/me/*` |
| `researcher` | 1 000 queries | + read aggregated `/metrics` |
| `admin` | 10 000 queries | + manage users (future) |

JWT tokens carry the role claim and are validated on every request (`src/auth/deps.py`).

## 4. Acceptable-use boundary

The system answers questions inside the **sustainable energy / smart-building** domain only (cf. principle P5_SCOPE in `src/alignment/constitution.yaml`). Out-of-scope queries trigger a polite refusal rather than a fabricated answer.

## 5. Data retention

| Artifact | Location | Retention |
|---|---|---|
| `users` table | `data/auth.db` | Until user requests deletion |
| `query_records` | `data/auth.db` | 180 days, then truncate |
| `logs/audit.jsonl` | local FS | 90 days, then rotate |
| `logs/agent_traces.jsonl` (when enabled) | local FS | 30 days |

PII detected on input/output is **not** stored verbatim; only aggregate counts (`{"type": "EMAIL", "count": 1}`) are persisted.

## 6. Change control

- All code changes go through PR review on `dev-jorge` → `dev` → `main`.
- The pre-commit hook enforces ruff lint and basic tests.
- Any change to `src/agents/guard_agent.py`, `src/security/*`, `src/alignment/constitution.yaml` is treated as **safety-critical** and requires the safety officer's approval before merge.

## 7. Decision rule on unsafe behaviour

If `MEASURE` (see `risk_register.md`) reports an attack-success rate above 10 % on the red-team suite, deployment is paused until mitigations land. This is the explicit "stop the line" authority the framework expects from GOVERN.

## 8. External dependencies

The system relies on third-party LLM providers (Gemini / Claude / Groq / OpenRouter / Kimi / Ollama). Each provider's privacy policy and data-retention guarantee are listed in `docs/governance/data_sheet.md` § "Third-party processors".
