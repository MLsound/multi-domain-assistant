# Changelog

## Unreleased

- Added `docs/CONTEXT.md` explaining the specific application domain (Sustainable Energy) used to validate the framework.
- Cross-referenced `CONTEXT.md` across README, API docs, system prompts, and test scripts.
- Added API backend and schema support in `src/api/`
- Added authentication, authorization, and security tooling in `src/auth/` and `src/security/`
- Added model registry and runtime configuration management in `src/config/`
- Added evaluation pipeline, metrics, and MLflow integration in `src/evaluation/`
- Added retrieval, caching, memory, and router functionality in `src/retrieval/`, `src/cache/`, `src/memory/`, and `src/router/`
- Added multi-agent orchestration and agent modules in `src/agents/`
- Added governance and compliance documentation under `docs/governance/`
- Added Docker and deployment scripting support
- Added comprehensive new test coverage in `tests/`
- Ignored MLflow runtime artifact `mlflow.db`
