# Causal Reasoning (CAF)

CAF (Causal Autonomy Framework) verifies an LLM's output against a knowledge base at request time, using an iterative generate → verify → constrain → regenerate loop. Each draft response is parsed into RDF triplets, then each triplet is checked against a SPARQL knowledge base, and if verification fails the failures are turned into constraints that are fed back into the next generation attempt.

## Architecture

- `api/` - FastAPI gateway (`api/main.py`, `POST /v1/infer`), orchestrates the full pipeline over HTTP.
- `modules/` - the four pipeline stages: `inference_engine/` (drafts a response + causal assertions), `semantic_parser/` (assertions → RDF triples, entity-linked via ChromaDB), `truth_anchor/` (SPARQL verification against Apache Jena Fuseki), `causal_validator/` (cycle/consistency checks on the resulting causal graph).
- `experiments/` - the standalone `CAFLoop` algorithm (`caf_algorithm.py`), a SPARQL-backed verification layer (`knowledge_base_fvl.py`'s `KnowledgeBaseFVL`, using spaCy for triplet parsing), CounterBench evaluation harnesses, and baselines (CoT, RAG).
- `common/llm_integration.py` - backend-agnostic LLM wrapper shared by every entry point (local HuggingFace models or a running Ollama server).

There are two ways to drive this: the FastAPI service (`api/main.py`, needs a separate inference-engine server too - see `modules/inference_engine/`), or `CAFLoop` directly in a script, which only needs an `InferenceLayer` and a `FormalVerificationLayer` - see [Running a query](#running-a-query) below for the minimal path.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # needed by KnowledgeBaseFVL's triplet parser
```

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

To build that file from real text instead of hand-writing it, use the companion [causal-discovery](../causal-discovery) repo's `populate_kb_from_text.py` - it extracts a causal graph from a passage and can POST straight to this endpoint.

### Clearing it

```bash
curl -X POST http://localhost:3030/dataset/update \
  -H "Content-Type: application/sparql-update" \
  --data 'DELETE WHERE { ?s ?p ?o }'
```

## Running a query

The minimal path - no FastAPI service, just `CAFLoop` directly against a real LLM and a real KB:

```python
from experiments.caf_algorithm import CAFLoop, CAFConfig
from experiments.knowledge_base_fvl import KnowledgeBaseFVL
from common.llm_integration import HuggingFaceCausalLMLayer, LLMConfig

llm = HuggingFaceCausalLMLayer(LLMConfig(model_name="Qwen/Qwen3-14B", load_in_4bit=True, trust_remote_code=True))
verifier = KnowledgeBaseFVL(sparql_endpoint="http://localhost:3030/dataset/query")

caf_loop = CAFLoop(
    config=CAFConfig(max_iterations=3, verification_threshold=0.8),
    inference_layer=llm,
    verification_layer=verifier,
)

output = caf_loop.execute("Does water pooling cause mold growth?")
print(output.final_response, output.decision, output.final_score)
```

Or run the CounterBench benchmark harness:

```bash
python -m experiments.run_counterbench_experiment \
  --input <your-dataset>.json \
  --use-llm --llm-model <name> \
  --use-real-sparql \
  --sparql-endpoint http://localhost:3030/dataset/query \
  --output results/caf_run
```

Or as a live service (needs a separate inference-engine server running too):

```bash
python -m api.main
```

## Gotchas

- **`TruthAnchor`/`KnowledgeBaseFVL` always returns "not found"**: check the dataset is actually configured (see the `ASK` query above), and check the entities you're querying have `rdfs:label` triples - `KnowledgeBaseFVL` links mention text to KB URIs by label, and a KB of bare `<uri> causes <uri>` triples with no labels will never resolve anything.
- **`KnowledgeBaseFVL`'s triplet parser is naive**: it does dependency-parse SVO extraction over the LLM's raw answer text. It handles simple declarative sentences ("X causes Y") reliably, but complex phrasing (relative clauses, passive voice, rephrasing) can make it grab the wrong subject/object - this shows up as an unexpected REJECT even when the KB genuinely supports the claim. If you're building an eval prompt, ask for a short declarative answer.
- **Entity linking finds nothing despite having loaded data earlier**: `EntityLinker` (`modules/semantic_parser/parser.py`, used by the FastAPI service path) falls back to a non-persistent in-memory ChromaDB client if it can't reach a ChromaDB server - confirm the `chromadb` service in `docker-compose.yml` is actually running, don't assume the fallback picked up prior data.
- **Two different spaCy states in this repo**: `modules/semantic_parser/parser.py` (the FastAPI service path) has spaCy NER disabled (`SPACY_AVAILABLE = False`) and runs on dependency-parse pattern matching instead, independent of whatever spaCy install you have - this is a deliberate fallback, not a bug. `experiments/knowledge_base_fvl.py`'s `KnowledgeBaseFVL` (the `CAFLoop`-direct path above) genuinely requires and uses spaCy - install it per [Setup](#setup).

## License

MIT - see `LICENSE`.
