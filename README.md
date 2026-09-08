# CAVAL — Causal Autonomy Verification And vaLidation

[![CI](https://github.com/brighk/causal_autonomy/actions/workflows/ci.yml/badge.svg)](https://github.com/brighk/causal_autonomy/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/caval.svg)](https://pypi.org/project/caval/)

A causal fact-checker for LLMs. CAVAL prevents language models from hallucinating causal claims by grounding every causal assertion against a formal knowledge base — and answers interventional and counterfactual questions using do-calculus on that knowledge base instead of the model's statistical priors.

---

## The problem

Large language models are trained on observational text. They learn statistical associations — what words follow what words — not causal structure. When you ask an LLM "Does X cause Y?" it pattern-matches against its training distribution. When you ask "What would happen if we forced X?", or "Would Y have occurred if X had been prevented?", it does the same thing, dressed in causal-sounding language. It has no mechanism to distinguish those three questions, and it regularly gets them wrong.

This is not a prompting problem. It is a structural one. Judea Pearl's causal hierarchy formalises it precisely:

| Level | Question type | Example | What it requires |
|-------|--------------|---------|-----------------|
| **1 — Observational** | Does X correlate with Y? | Do CPU spikes co-occur with timeouts? | Pattern matching over data |
| **2 — Interventional** | What if we do(X)? | If we enforce auth at node A, what is the breach rate? | A causal model + do-calculus |
| **3 — Counterfactual** | Would Y have occurred if X had been different? | Would this breach have happened without the rogue base station? | A structural causal model + abduction |

An LLM operates natively at Level 1. It can approximate Levels 2 and 3 in easy cases, but it has no guarantee of correctness and no mechanism for abstention when it doesn't know.

CAVAL addresses this by keeping the LLM where it belongs — generating and reformulating text — and handing Levels 2 and 3 to formal causal machinery.

---

## How it works

CAVAL routes every query to the appropriate reasoning mechanism based on its type:

```
Query
  │
  ├─ Level 2/3 pattern detected? ──Yes──► Build causal graph from KB (SPARQL BFS)
  │                                        Apply do-calculus (Pearl's graph surgery)
  │                                        Return formal answer — LLM not called
  │
  └─ No (factual/observational) ──────► LLM drafts a response
                                         Parse response → RDF triplets
                                         SPARQL-verify each triplet against KB
                                         ┌─ All verified? ──► Return response
                                         └─ Contradictions? ──► Inject constraints
                                                                 LLM retries
                                                                 (up to N times)
```

**Level 1 (factual):** the LLM generates a candidate answer. The semantic parser extracts causal assertions as RDF triplets. The truth anchor queries the knowledge base via SPARQL. If a triplet contradicts the KB, the contradiction is fed back as a constraint and the LLM regenerates. This loop continues until the response is fully grounded or the iteration limit is reached.

**Level 2/3 (interventional and counterfactual):** the LLM is bypassed entirely. CAVAL links the query's entities to KB nodes, walks causal-predicate edges outward via SPARQL traversal to build a local causal graph, applies Pearl's do-calculus (graph surgery: remove incoming edges to the intervened node, propagate through descendants), and returns the formal result. The answer comes from the KB structure, not from the model's priors.

The response always carries a `causal_level` field (1, 2, or 3) so callers know which mechanism answered it.

---

## Quick start

```bash
pip install caval
python -m spacy download en_core_web_sm   # one-time — see Setup
```

Two background services are required: Fuseki (the knowledge base) and the inference engine (the LLM server). See [Setup](#setup) for details.

```bash
# 1. Start Fuseki
FUSEKI_ADMIN_PASSWORD=secret docker compose -f deployment/docker-compose.yml up -d

# 2. Start the inference engine (separate terminal)
uv run python -m modules.inference_engine.server
```

```python
from caval import Caval

caf = Caval()

# Level 1 — factual query: LLM answers, SPARQL verifies
result = caf.ask("Does high CPU usage cause increased response time?")
print(result.text)                               # verified answer
print(result.verification_status.is_valid)       # True
print(result.causal_level)                       # 1

# Level 2 — interventional query: do-calculus answers, LLM not called
result = caf.ask("Would response time increase if we prevent CPU spikes?")
print(result.text)                               # "No. Under do(cpu_spikes=False)..."
print(result.verification_status.verification_method)  # "do_calculus"
print(result.causal_level)                       # 2
```

Async variant for embedding in an existing async app:

```python
result = await caf.aask("Does rain cause road slipperiness?")
```

---

## What it is for

CAVAL is useful in any domain where you need causal reasoning you can trust and trace — where "the model said so" is not an acceptable justification.

**6G network security.** A security analyst queries an incident knowledge graph:
- *Level 1:* "Do signal anomalies co-occur with auth failures at node 7?" — SPARQL fact check.
- *Level 2:* "If we enforce beamforming authentication at node 7, what is the downstream breach probability?" — do-calculus on the causal security model.
- *Level 3:* "Would this breach have occurred had quantum-safe encryption been deployed at the time?" — counterfactual forensics against the stored structural model.

**Medical and clinical reasoning.** Feed clinical literature through the companion [causal-discovery](../causal-discovery) extractor to build a causal knowledge graph, then query it. Every accepted answer traces back to the source edge that supports it.

**Policy and economic analysis.** Same architecture. The KB holds causal relationships extracted from policy documents; CAVAL answers intervention questions ("what is the effect of policy X on outcome Y?") without conflating correlation with causation.

**Verified LLM pipelines.** Any application that uses an LLM to reason about a domain where you have structured causal knowledge. CAVAL sits between the LLM and the user, ensuring the model's causal claims are grounded before they are accepted.

---

## Architecture

```
caval/          Public library: Caval class, CAFPipeline
api/            FastAPI gateway (POST /v1/infer) — same pipeline over HTTP
modules/
  inference_engine/   LLM client (HTTP) and server (vLLM / HuggingFace 4-bit)
  semantic_parser/    LLM text → RDF triplets via spaCy + Fuseki entity linking
  truth_anchor/       SPARQL verification against Apache Jena Fuseki
                      causal_graph_builder.py — BFS KB traversal for Level 2/3
  causal_validator/   Cycle and consistency checks on the verified causal graph
experiments/
  caf_algorithm.py              CAFLoop — the original script-level driver
  intervention_calculus.py      CausalGraph, do-calculus, parse_counterfactual_query
  kb_fvl_with_intervention.py   KnowledgeBaseFVLWithIntervention (experiments path)
  knowledge_base_fvl.py         KnowledgeBaseFVL — base SPARQL verifier
  run_counterbench_experiment.py CounterBench evaluation harness
common/
  llm_integration.py  Backend-agnostic LLM wrapper (Ollama, HuggingFace, 4-bit)
```

**Three entry points:**

1. **`caval` library** (`from caval import Caval`) — `pip install caval`, talks to Fuseki and to a running inference-engine server over HTTP. The primary interface.

2. **FastAPI service** (`api/main.py`) — the same pipeline exposed as `POST /v1/infer`. Run with `uv run python -m api.main`.

3. **`CAFLoop` directly** (`experiments/caf_algorithm.py`) — for scripting and benchmarks, no FastAPI layer. Use `KnowledgeBaseFVLWithIntervention` as the verifier to get Level 2/3 routing here too.

---

## Setup

`caval` itself is lightweight — spaCy, SPARQLWrapper, httpx, pydantic — no GPU stack. The inference engine, FastAPI gateway, and experiments harness are optional extras:

```bash
uv sync                          # caval only
uv sync --extra api              # + FastAPI gateway
uv sync --extra llm              # + local model loading
uv sync --extra experiments      # + benchmark harness
uv sync --all-extras             # everything
```

`en_core_web_sm` is not on PyPI. For local dev, `uv sync` installs it automatically. If you installed `caval` from PyPI:

```bash
python -m spacy download en_core_web_sm
```

`SemanticParser` raises a clear `RuntimeError` with this exact command if the model is missing.

### Running Fuseki

```bash
FUSEKI_ADMIN_PASSWORD=<password> docker compose -f deployment/docker-compose.yml up -d
```

Verify the dataset is configured (an unconfigured Fuseki silently serves an empty default dataset):

```bash
curl http://localhost:3030/$/ping
curl -G http://localhost:3030/dataset/query --data-urlencode "query=ASK { ?s ?p ?o }"
```

The second command should return a JSON `ASK` result. If it 404s, the dataset config did not load — see the note in `deployment/docker-compose.yml` about the `secoresearch/fuseki` image's mount paths.

### Loading knowledge

A single fact:

```bash
curl -X POST http://localhost:3030/dataset/update \
  -H "Content-Type: application/sparql-update" \
  --data 'INSERT DATA {
    <http://local.caf/rain> <http://causality.org/causes> <http://local.caf/slippery_road>
  }'
```

Bulk load from an N-Triples file:

```bash
curl -X POST http://localhost:3030/dataset/data \
  -H "Content-Type: application/n-triples" \
  --data-binary @my_facts.nt
```

To build that file from unstructured text, use the companion [causal-discovery](../causal-discovery) repo's `populate_kb_from_text.py`.

Clearing the dataset:

```bash
curl -X POST http://localhost:3030/dataset/update \
  -H "Content-Type: application/sparql-update" \
  --data 'DELETE WHERE { ?s ?p ?o }'
```

### Development checks

```bash
uv sync --locked
uv run --no-sync prek install
uv run --no-sync prek run --all-files   # lint + format the whole repo
```

---

## Running the CounterBench benchmark

```bash
uv run python -m experiments.run_counterbench_experiment \
  --input <dataset>.json \
  --use-llm --llm-model <name> \
  --use-real-sparql \
  --sparql-endpoint http://localhost:3030/dataset/query \
  --output results/caf_run
```

Requires `uv sync --extra experiments --extra llm`.

---

## Gotchas

**Nothing verifies — everything lands in "unverifiable":** the failure is in entity linking, not the SPARQL check. `EntityLinker` links mention text to KB URIs via `rdfs:label` / `skos:prefLabel`. A KB of bare `<uri> causes <uri>` triples with no labels means every entity fails to link, `subject_linked=False` on every triplet, and `TruthAnchor` skips the SPARQL query entirely rather than querying a fabricated URI. Check the dataset is configured (see the `ASK` query above) and that your KB entities have label triples — without them, nothing can be verified.

**Level 2/3 routing silently falls back to Level 1:** `CausalGraphBuilder` returns an empty graph if it can't link the query's entities to KB nodes, or finds no causal-predicate edges within `max_hops` (default 2) hops. The response will have `causal_level=1` and the LLM will be called. Increase `max_hops` if your causal chains are longer, at the cost of more SPARQL round-trips.

**The semantic parser is fragile on complex phrasing:** `SemanticParser` does dependency-parse SVO extraction. It handles simple declarative sentences ("X causes Y") reliably. Complex phrasing (passive voice, relative clauses, subordinate clauses) can produce wrong triplets — the LLM will see an unexpected constraint. Ask for short declarative answers in your prompts.

**Entity linking trades false negatives for false positives:** the fuzzy matcher uses `max(ratio, partial_ratio)`, so a short phrase ("habitat destruction") can match a longer KB label that contains it as a substring. Prefer specific multi-word phrases over single generic words, especially on KBs built from non-atomic labels.

**`SemanticParser` has no fallback:** if spaCy or `en_core_web_sm` is missing, `Caval()` raises `RuntimeError` immediately rather than degrading silently — a degraded extractor would quietly undermine verification.

---

## License

MIT — see `LICENSE`.
