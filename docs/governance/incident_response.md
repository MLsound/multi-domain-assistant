# Incident Response Runbook

NIST AI RMF **MANAGE** function. This runbook is consulted when a metric crosses a red threshold or a user reports unsafe behaviour.

## Severity classification

| Sev | Trigger | Response time |
|---|---|---|
| SEV-1 | Confirmed PII / secrets leaked to a user | < 1 hour |
| SEV-2 | Successful jailbreak in production / canary token leaked | < 4 hours |
| SEV-3 | Faithfulness < 0.6 on evaluation suite, or error rate > 5 % | < 24 hours |
| SEV-4 | Latency p95 > 8 s, or quota-bypass attempts | < 1 week |

## Response steps (SEV-1 / SEV-2)

1. **Triage** — Safety officer (Lloveras) acknowledges the incident in the project channel and opens an incident note.
2. **Contain** — Set `LLM_PROVIDER=ollama` (local model) so no further outbound calls are made; if the issue is in `guard_agent.py`, set the offending route to a deny-all stub.
3. **Eradicate** — Identify the failing rule / pattern / chunk; add a regression test in `tests/test_security.py` that reproduces the failure; fix the bug.
4. **Recover** — Re-deploy. Run the full evaluation suite + red-team set. Verify metrics return to green.
5. **Postmortem** — Within 5 days: root cause, what controls failed, what new control prevents recurrence, update `risk_register.md`.

## Communication

- **Internal:** group chat (Sarmiento, Lloveras, Cuenca) + email to teaching staff if the incident affects deliverables.
- **Affected users:** transparent message in the response itself; reset their quota; offer to delete logs that contain their query (`/me/queries` exposes them; deletion endpoint is future work).

## Rollback

Every release is a git tag. To roll back:

```bash
git tag -l "v*"
git checkout v0.2.0
docker-compose up --build --force-recreate
```

The auth DB is forward-compatible; older builds keep working against newer schemas because of additive migrations only (rule of thumb: never drop columns in a hotfix).

## Lessons-learned cadence

After each incident a 30-minute group review converts every learning into either:

- a regression test, or
- a new entry in `risk_register.md`, or
- a clarification in `policy.md`.

If neither applies, the lesson is not internalised and we re-discuss.
