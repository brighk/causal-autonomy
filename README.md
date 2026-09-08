# Causal Reasoning Verification (CAVAL CAusal VALidation/Verification) (part from Causal Autonomy Framework)

[![CI](https://github.com/brighk/causal_autonomy/actions/workflows/ci.yml/badge.svg)](https://github.com/brighk/causal_autonomy/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/caval.svg)](https://pypi.org/project/caval/)

CAF (Causal Autonomy Framework) verifies an LLM's output against a knowledge base at request time, using an iterative generate → verify → constrain → regenerate loop. Each draft response is parsed into RDF triplets, then each triplet is checked against a SPARQL knowledge base, and if verification fails the failures are turned into constraints that are fed back into the next generation attempt.

## Architecture

- `api/` - FastAPI gateway (`api/main.py`, `POST /v1/infer`), orchestrates the full pipeline over HTTP.
- `modules/` - the four pipeline stages: `inference_engine/` (drafts a response + causal assertions), `semantic_parser/` (assertions → RDF triples, entity-linked directly against Fuseki's `rdfs:label`/`skos:prefLabel` triples - exact match, then fuzzy), `truth_anchor/` (SPARQL verification against Apache Jena Fuseki), `causal_validator/` (cycle/consistency checks on the resulting causal graph).
- `experiments/` - the standalone `CAFLoop` algorithm (`caf_algorithm.py`), a SPARQL-backed verification layer (`knowledge_base_fvl.py`'s `KnowledgeBaseFVL`, using spaCy for triplet parsing), CounterBench evaluation harnesses, and baselines (CoT, RAG).
- `common/llm_integration.py` - backend-agnostic LLM wrapper shared by every entry point (local HuggingFace models or a running Ollama server).

There are three ways to drive this: the `caval` library (`caval/`, a thin wrapper around `api/`+`modules/` - see [Install as a library](#install-as-a-library) below), the FastAPI service directly (`api/main.py`, needs a separate inference-engine server too - see `modules/inference_engine/`), or `CAFLoop` in a script, which only needs an `InferenceLayer` and a `FormalVerificationLayer` - see [Running a query](#running-a-query) below for that minimal path.

## Setup

`caval` itself is lightweight - parsing, entity linking, and SPARQL
verification only, no web framework or ML stack (it talks to Fuseki and to
an already-running inference engine over plain HTTP). The FastAPI gateway,
local model loading, and the `experiments/` benchmark harness are optional
extras so a `caval`-only consumer doesn't pull in a multi-GB ML stack they
don't need:

```bash
uv sync                          # caval only - lean, no torch/fastapi/etc.
uv sync --extra api              # + the FastAPI gateway (api/main.py)
uv sync --extra llm              # + local model loading (modules/inference_engine/server.py)
uv sync --extra experiments      # + the experiments/ benchmark harness
uv sync --all-extras             # everything - what you want for full local dev on this repo
```

`en_core_web_sm` (spaCy's model, needed for triplet parsing) isn't a real
PyPI package, so it's not a published dependency of `caval` - a plain
`pip install caval` can't resolve it. For local dev on *this* repo, `uv
sync` still installs it automatically (it's in the `dev` dependency-group,
mapped via `[tool.uv.sources]`). If you installed `caval` from PyPI
instead, run this once:

```bash
python -m spacy download en_core_web_sm
```

`Caval()`/`SemanticParser()` raise a clear `RuntimeError` with this exact
command if the model isn't found, rather than failing silently.


## Development checks

Install the development tools and enable automatic checks before each commit:

```bash
uv sync --locked
uv run --no-sync prek install
```

For a tooling-only checkout without the ML dependencies, use
`uv sync --locked --only-dev --inexact` instead of `uv sync --locked`.
Run `prek install` once per clone. The hooks use the Ruff version in `uv.lock`.

```bash
uv run --no-sync prek run --all-files  # check and fix the entire repo
uv run --no-sync ruff check .         # lint without changing files
uv run --no-sync ruff format --check .
```

prek runs Ruff linting, import sorting, and formatting, plus checks for merge
conflicts, YAML/TOML syntax, files larger than 1 MiB, trailing whitespace, and
missing final newlines. Normal commits check staged files only. If hooks fix
files, review and stage those changes before retrying the commit. Existing files
may need cleanup the first time they are checked.

## Running Fuseki

```bash
FUSEKI_ADMIN_PASSWORD=<pick-something> \
  docker compose -f deployment/docker-compose.yml up -d
```

Verify it's actually up *and* the dataset is configured (an unconfigured Fuseki silently serves an empty default dataset instead of erroring):

```bash
curl http://localhost:3030/\$/ping
curl -G http://localhost:3030/dataset/query --data-urlencode "query=ASK { ?s ?p ?o }"
```

The second command should return a JSON `ASK` result, not a 404 or an HTML error page. If it 404s, the dataset config didn't load - see the note in `deployment/docker-compose.yml` about the `secoresearch/fuseki` image's actual mount paths (`/fuseki-base/configuration/assembler.ttl`), which differ from that image's own docs and from older image versions.

### Loading knowledge into it

A single fact, direct SPARQL:

```bash
curl -X POST http://localhost:3030/dataset/update \
  -H "Content-Type: application/sparql-update" \
  --data 'INSERT DATA { <http://local.caf/rain> <http://causality.org/causes> <http://local.caf/slippery_road> }'
```

Bulk load an N-Triples file:

```bash
curl -X POST http://localhost:3030/dataset/data \
  -H "Content-Type: application/n-triples" \
  --data-binary @my_facts.nt
```

To build that file from real text instead of hand-writing it, use the companion [causal-discovery](../causal-discovery) repo's `populate_kb_from_text.py` - it extracts a causal graph from a chunk of text and can POST straight to this endpoint.

### Clearing it

```bash
curl -X POST http://localhost:3030/dataset/update \
  -H "Content-Type: application/sparql-update" \
  --data 'DELETE WHERE { ?s ?p ?o }'
```

## Install as a library

`caval` is [published on PyPI](https://pypi.org/project/caval/) and wraps
the `api/`+`modules/` pipeline as a plain importable class - no FastAPI
service to run yourself, but the LLM still runs as its own background
process (for GPU isolation) and Fuseki still runs via docker-compose.
`pip install caval`/`uv add caval` pulls in only what `caval` itself needs
(spaCy, SPARQLWrapper, httpx, pydantic) - no torch, no FastAPI, nothing
GPU-related - since it talks to Fuseki and to the already-running inference
engine below over plain HTTP, not in-process. Three things running, few
lines of code:

```bash
pip install caval   # or: uv add caval
python -m spacy download en_core_web_sm   # one-time - see the note in Setup above
```

```bash
# 1. Fuseki
FUSEKI_ADMIN_PASSWORD=<pick-something> docker compose -f deployment/docker-compose.yml up -d

# 2. The LLM, in a separate terminal (needs the `llm` extra: uv sync --extra llm)
uv run python -m modules.inference_engine.server
```

```python
# 3. Your script
from caval import Caval

caf = Caval()  # zero-config: reads .env, same as the FastAPI gateway's Settings()
result = caf.ask("Does high cpu usage cause increased response time?")
print(result.text, result.verification_status.is_valid)
```

An async `caf.aask(...)` is also available for embedding in an already-async
app (e.g. a FastAPI route) - `caf.ask(...)` can't be called from inside a
running event loop. See `examples/quickstart.py` and
`examples/quickstart_async.py` for runnable versions of both. `caval` is
built on the `api/`+`modules/` implementation specifically, not the
`experiments/`+`common/` script path described below.

## Running a query

The minimal path - no FastAPI service, just `CAFLoop` directly against a real LLM and a real KB:

```python
from experiments.caf_algorithm import CAFLoop, CAFConfig
from experiments.kb_fvl_with_intervention import KnowledgeBaseFVLWithIntervention
from common.llm_integration import HuggingFaceCausalLMLayer, LLMConfig

llm = HuggingFaceCausalLMLayer(LLMConfig(model_name="Qwen/Qwen3-14B", load_in_4bit=True, trust_remote_code=True))
verifier = KnowledgeBaseFVLWithIntervention(sparql_endpoint="http://localhost:3030/dataset/query")

caf_loop = CAFLoop(
    config=CAFConfig(max_iterations=3, verification_threshold=0.8),
    inference_layer=llm,
    verification_layer=verifier,
)

output = caf_loop.execute("Does water pooling cause mold growth?")
print(output.final_response, output.decision, output.final_score)
```

`KnowledgeBaseFVLWithIntervention` (`experiments/kb_fvl_with_intervention.py`) is a strict superset of `KnowledgeBaseFVL`: for a factual question it verifies via SPARQL exactly like the plain class, but if the question looks counterfactual ("Would X occur if we prevented Y?") it instead builds a causal graph by walking causal-predicate edges outward from the mentioned entities in the same live KB, and answers via Pearl's do-calculus (`experiments/intervention_calculus.py`) - no extra setup required when driving it through `CAFLoop` this way. Plain `KnowledgeBaseFVL` is still there for callers that only ever ask factual questions.

Or run the CounterBench benchmark harness (needs `uv sync --extra experiments --extra llm` - `experiments/` pulls in numpy/pandas/matplotlib, `--use-llm` pulls in torch/transformers via `common/llm_integration.py`):

```bash
uv run python -m experiments.run_counterbench_experiment \
  --input <your-dataset>.json \
  --use-llm --llm-model <name> \
  --use-real-sparql \
  --sparql-endpoint http://localhost:3030/dataset/query \
  --output results/caf_run
```

Or as a live service (needs `uv sync --extra api`, and a separate inference-engine server running too):

```bash
uv run python -m api.main
```

## Gotchas

- **`TruthAnchor`/`KnowledgeBaseFVL` always returns "not found"**: check the dataset is actually configured (see the `ASK` query above), and check the entities you're querying have `rdfs:label` triples - `KnowledgeBaseFVL` links mention text to KB URIs by label, and a KB of bare `<uri> causes <uri>` triples with no labels will never resolve anything.
- **`KnowledgeBaseFVL`'s triplet parser is naive**: it does dependency-parse SVO extraction over the LLM's raw answer text. It handles simple declarative sentences ("X causes Y") reliably, but complex phrasing (relative clauses, passive voice, rephrasing) can make it grab the wrong subject/object - this shows up as an unexpected REJECT even when the KB genuinely supports the claim. If you're building an eval prompt, ask for a short declarative answer.
- **Entity linking is substring-tolerant, which trades false negatives for false positives**: `_link_entity`'s fuzzy match uses `max(ratio, partial_ratio)`, so a short clean phrase (e.g. "habitat destruction") can still link to a much longer KB label that contains it verbatim (e.g. "habitat destruction which in turn leads to biodiversity loss") - useful against KBs with non-atomic, multi-clause labels (common output of `causal-discovery`'s extractor on complex sentences). The flip side: a short or generic entity mention can now spuriously match any long label that happens to contain it as a substring, regardless of whether they're actually the same concept. Prefer specific multi-word claims over single generic words when querying a KB built from non-atomic labels.
- **`modules/semantic_parser/parser.py`'s `EntityLinker` queries Fuseki directly, same as `KnowledgeBaseFVL`** - no separate vector index to seed or keep in sync (an earlier version used ChromaDB for this; it was never actually populated, so entity linking silently found nothing). The gotcha above about needing `rdfs:label`/`skos:prefLabel` triples applies here too.
- **`modules/semantic_parser/parser.py` requires spaCy - there is no fallback**: `SemanticParser.__init__` calls `spacy.load(spacy_model)` (default `en_core_web_sm`) and raises `RuntimeError` immediately if spaCy or the model isn't installed, rather than silently parsing with something weaker - a degraded extractor would quietly undermine what `TruthAnchor` is verifying. Install the model per [Setup](#setup). In `api/main.py`, a failed init here surfaces as `services['parser']` being `None` and `/health` reporting `semantic_parser: false`.
- **`KnowledgeBaseFVLWithIntervention` silently falls back to plain SPARQL** if it can't link the question's entities to the KB, or finds no causal-predicate edges within `causal_graph_max_hops` (default 2) hops of them - a real counterfactual question can come back as an ordinary factual FAILED/PARTIAL instead of a do-calculus VERIFIED/CONTRADICTION if the relevant chain is more than 2 hops away. Increase `causal_graph_max_hops` if your KB's causal chains are longer, at the cost of more SPARQL round-trips per verification.

## License

MIT - see `LICENSE`.
