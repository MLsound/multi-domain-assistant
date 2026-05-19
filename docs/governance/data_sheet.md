# Datasheet — Knowledge Assistant Corpus

Following Gebru et al., 2021 ("Datasheets for Datasets").

## Motivation

The corpus was assembled for FIUBA PNL III (2026) to train and evaluate a multi-agent RAG pipeline in the **sustainable energy and smart-building** domain. It is intended for academic use only.

## Composition

| Folder | Files | Source category |
|---|---:|---|
| `data/scientific/` | 13 | Photovoltaic physics, building thermodynamics, energy storage chemistry |
| `data/software/` | 15 | HEMS / EV charger / thermostat API references |
| `data/user/` | 7 | End-user safety manuals (e.g. NFPA 855), scheduling and maintenance guides |
| `data/eval/test_suite.json` | 1 | 20-question evaluation set with reference answers and expected domains |

Files are plain text, ~1–10 KB each, English. Total ≈ 250 KB.

## Collection process

Documents were curated manually from public technical sources by the project team. No personal data were collected. Every file was reviewed for licence compatibility (educational use).

## Pre-processing

- Text is split into 512-character overlapping chunks (`CHUNK_SIZE=512`, `CHUNK_OVERLAP=50`) at indexing time.
- Embeddings are computed with `BAAI/bge-large-en-v1.5` and stored in Qdrant (`collection_name=rag_collection`).
- The MLP router is trained for 50 epochs on `all-MiniLM-L6-v2` embeddings of the same chunks.

## Uses

- Training the domain MLP classifier.
- Retrieval ground-truth for the evaluation suite.
- Reference context for the Synthesis agent at inference time.

The corpus must NOT be used to make safety-critical decisions, to provide medical/legal advice, or to train a model deployed for end-users without further validation.

## Distribution

Bundled in the project repository. Subject to the same licence as the project.

## Maintenance

Maintained by Grupo 1. Issues, additions and removals tracked in the repo.

## Third-party processors

When a remote LLM provider is used at inference time, the user query and the retrieved chunks are transmitted to the provider:

| Provider | Data sent | Retention guarantee |
|---|---|---|
| Google Gemini | query + retrieved chunks + system prompt | per Google AI privacy policy |
| Anthropic Claude | idem | Anthropic privacy policy |
| Groq | idem | Groq privacy policy |
| OpenRouter | idem | OpenRouter privacy policy |
| Moonshot Kimi | idem | Moonshot privacy policy |
| Ollama (local) | nothing leaves the host | n/a |

Operators of the system must surface this to end users (transparency requirement, EU AI Act art. 52).
