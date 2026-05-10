# Model Card — Knowledge Assistant Pipeline

Following Mitchell et al., 2019 ("Model Cards for Model Reporting").

## Model details

- **Name:** Knowledge Assistant — Multi-Agent Agentic RAG (v0.3.0)
- **Owners:** Grupo 1 — F. Sarmiento, A. Lloveras, J. Cuenca
- **Affiliation:** FIUBA, Maestría en Inteligencia Artificial, PNL III (2026)
- **Date:** May 2026
- **Type:** Composition of (a) custom MLP domain classifier, (b) sentence-transformer embedders, (c) cross-encoder reranker, (d) third-party LLM (Gemini / Claude / Groq / Ollama …) used through a constitutional-AI-style critic loop.
- **Licence (project):** MIT-style for own code; third-party model licences as published by their providers.

## Intended use

| Aspect | Statement |
|---|---|
| Primary use cases | Q&A over a curated corpus on photovoltaic systems, BESS, HEMS, smart-building APIs, and end-user safety. |
| Primary users | Operators, students, researchers in sustainable energy. |
| Out-of-scope | Medical, legal, financial advice; safety-critical control loops; any action without human review. |

## Factors

The system handles English (primary) and short Spanish queries. Performance has not been quantified across dialects, code-switched inputs or non-Latin scripts.

## Metrics

Reported on `data/eval/test_suite.json` (20 questions across 3 domains):

| Metric | Definition | Latest value |
|---|---|---|
| Faithfulness | Ragas LLM judge | _measured per run_ |
| Context recall | Ragas LLM judge | _measured per run_ |
| Precision@5 | Top-5 chunks from expected domain | _measured per run_ |
| Semantic similarity | bge-large-en cosine | _measured per run_ |
| ROUGE-L F1 | Lexical overlap | _measured per run_ |
| Avg. latency p50 | End-to-end pipeline | < 3000 ms target |
| Injection block rate | own red-team set | ≥ 0.90 target |

## Training data

The MLP router is trained on a 35-document corpus split across:

- `data/scientific/` — 13 PV physics & building thermodynamics documents
- `data/software/` — 15 HEMS / EV charger / thermostat API references
- `data/user/` — 7 safety manuals & scheduling guides

See `docs/governance/data_sheet.md` for provenance and licence per file.

## Quantitative analyses

Bias and fairness are not formally quantified yet; the corpus is technical literature with low identity-attribute content. A residual-bias analysis is listed as future work in the technical report.

## Ethical considerations

- **PII**: detection-and-redaction is applied to every request; original PII is never stored.
- **Safety**: refuses requests that would compromise people, property or critical infrastructure (constitution P3).
- **Transparency**: every response cites the document IDs used (constitution P7).

## Caveats and recommendations

- Third-party LLM responses can drift over time; faithfulness is recomputed per deployment.
- The custom MLP router is trained on a small corpus; broadening the corpus is expected to reduce confidence-collapse on out-of-distribution queries.
