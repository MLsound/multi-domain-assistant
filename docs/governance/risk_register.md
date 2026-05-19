# Risk Register — Knowledge Assistant

NIST AI RMF **MAP** + **MEASURE** functions: identification, analysis and tracking of risks throughout the AI lifecycle.

Likelihood (L) and Impact (I) are scored 1 (low) – 5 (critical). Risk score = L × I.

| ID | Risk | OWASP | L | I | Score | Mitigation in code | Owner |
|---|---|---|---:|---:|---:|---|---|
| R-01 | Direct prompt injection ("ignore previous instructions") | LLM01 | 5 | 4 | 20 | `src/security/injection_scorer.py` heuristic + canary | Lloveras |
| R-02 | Indirect prompt injection via poisoned RAG document | LLM01 | 4 | 5 | 20 | Output PII scrub; future: per-chunk trust score | Lloveras |
| R-03 | Hallucination / unsupported claim | — | 4 | 4 | 16 | Critic agent + Constitutional principle P1_GROUNDED | Cuenca |
| R-04 | PII leakage in response | LLM06 | 3 | 5 | 15 | `src/security/pii_redactor.py` on input AND output | Lloveras |
| R-05 | DoS via expensive prompt loops | LLM04 | 3 | 4 | 12 | Sliding-window limiter + per-user daily quota | Lloveras |
| R-06 | System-prompt extraction | LLM01 | 3 | 4 | 12 | Canary token + leak-marker check | Lloveras |
| R-07 | Out-of-scope answer (e.g. medical advice) | — | 4 | 3 | 12 | Constitutional principle P5_SCOPE | Cuenca |
| R-08 | Toxic / biased output | — | 2 | 4 | 8 | Constitutional principle P6_NO_TOXICITY + critic | Cuenca |
| R-09 | Stolen or replayed JWT | — | 2 | 4 | 8 | Short token lifetime (60 min); password hashing bcrypt | Sarmiento |
| R-10 | Webhook misuse / SSRF | LLM07 | 2 | 3 | 6 | URL allow-list (TODO) + 10 s timeout | Sarmiento |
| R-11 | Model theft via heavy scraping | LLM10 | 2 | 3 | 6 | Daily quota + rate limit + future watermarking | Cuenca |
| R-12 | Cache poisoning (storing fallback answers) | — | 2 | 2 | 4 | `_UNCACHEABLE_RESPONSES` allow-list in graph | Sarmiento |

## Measurement plan (MEASURE)

For each risk we track at least one quantitative metric exposed via `/metrics` or in the offline evaluation suite:

| Metric | Source | Target |
|---|---|---|
| `injection_block_rate` (red-team) | `tests/test_security.py` | ≥ 0.90 |
| `pii_redacted_count / total_requests` | `/metrics` | observed only |
| `blocked_by_guard_rate` | `/metrics` | < 0.05 on benign suite |
| `faithfulness` (Ragas) | `scripts/run_evaluation.py` | ≥ 0.80 |
| `error_rate` | `/metrics` | < 0.02 |
| `avg_latency_ms` | `/metrics` | < 3000 ms (p50) |

## Review cadence

The register is reviewed every two weeks during the project; risks are re-scored after each evaluation run.
