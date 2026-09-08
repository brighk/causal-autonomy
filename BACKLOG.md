# Backlog

Findings from a static read-through of the codebase (2026-09-08), aimed at
getting the project running cleanly on a local RTX 3090 (Qwen3-14B, 4-bit)
and cleaning up architecture drift between `experiments/` (the original
CAF algorithm) and `api/` + `modules/` (a later service rewrite of
overlapping concepts that never fully converged).

Priority: P0 = blocks running the thing, P1 = wrong/silently-broken
behavior, P2 = architecture/duplication cleanup, P3 = minor.

---

## P0 - blocks running it

### 1. `modules/inference_engine/engine.py` has no quantization path
`InferenceEngine._init_huggingface()` ([modules/inference_engine/engine.py:80-89](modules/inference_engine/engine.py#L80-L89))
calls `AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=..., device_map="auto", trust_remote_code=True)`
with no `BitsAndBytesConfig`. `bitsandbytes` is a project dependency but only
`common/llm_integration.py`'s `HuggingFaceCausalLMLayer` actually uses it
([common/llm_integration.py:135-147](common/llm_integration.py#L135-L147)).
`use_vllm` defaults to `True` ([utils/config.py:34](utils/config.py#L34)) but `vllm` isn't in
`pyproject.toml` at all, so `VLLM_AVAILABLE` will be `False` and it silently
falls through to the unquantized HF path. Default model is
`meta-llama/Llama-3-70b-chat-hf` ([utils/config.py:28-31](utils/config.py#L28-L31)), which is both gated
and enormous.

**Impact:** the `api/main.py` service path (Path B) cannot serve Qwen3-14B
4-bit today - either OOMs on a 24GB 3090 trying to load fp16/fp32, or is
pointed at the wrong model entirely.

**Fix direction:** don't maintain a second LLM-loading implementation here.
Have `InferenceEngine` delegate to `common/llm_integration.py`'s
`HuggingFaceCausalLMLayer`/`OllamaLayer`/`create_causal_lm_layer` instead of
its own `_init_huggingface`/`_init_vllm`. This also fixes finding #7 (two
divergent prompt formats) for free.

### 2. Experiment script SPARQL endpoint defaults point at datasets that don't exist
`config/fuseki/assembler.ttl` ([config/fuseki/assembler.ttl:8-17](config/fuseki/assembler.ttl#L8-L17)) provisions exactly
one Fuseki dataset, named `dataset` (`http://localhost:3030/dataset/query`).
But:
- `experiments/run_experiment.py:725` defaults `--sparql-endpoint` to `http://localhost:3030/conceptnet/query`
- `experiments/run_counterbench_experiment.py:463` defaults to `http://localhost:3030/counterbench/query`
- `experiments/run_counterbench_with_intervention.py:196` same, `.../counterbench/query`
- `experiments/kb_fvl_with_intervention.py:52` (docstring example) and `:476` (default param) also say `.../counterbench/query`

Only `README.md`, `utils/config.py`, and `experiments/knowledge_base_fvl.py`'s
class default agree on `/dataset/query`.

**Impact:** running any of these three scripts with `--use-real-sparql` and
no explicit `--sparql-endpoint` 404s against the Fuseki this repo actually
provisions, unless you've manually created extra named datasets via the
Fuseki admin UI (undocumented, and not what `docker-compose.yml` sets up).

**Fix:** pick one dataset name and use it everywhere, or make the assembler
provision multiple named datasets if the intent was really to separate
`counterbench`/`conceptnet` data from general `dataset` data.

### 3. `run_experiment.py --use-real-sparql` crashes: wrong argparse dest
`--entity-th` is declared as `parser.add_argument("--entity-th", ...)`
([experiments/run_experiment.py:718-723](experiments/run_experiment.py#L718-L723)), which argparse exposes as
`args.entity_th`. But the code reads `args.entity_threshold` twice
([experiments/run_experiment.py:778](experiments/run_experiment.py#L778) and `:782`) - an attribute that doesn't
exist. `AttributeError: 'Namespace' object has no attribute 'entity_threshold'`
on every `--use-real-sparql` run.

**Fix:** rename the flag to `--entity-threshold`, or read `args.entity_th`.

---

## P1 - wrong or silently-broken behavior

### 4. All SPARQL calls block the FastAPI event loop
`TruthAnchor.verify()`/`_execute_query()` ([modules/truth_anchor/verifier.py:40](modules/truth_anchor/verifier.py#L40),
`:143`) and `EntityLinker._execute_sparql_query()`
([modules/semantic_parser/parser.py:201](modules/semantic_parser/parser.py#L201)) are declared `async def` but call
`SPARQLWrapper.query().convert()` directly - a synchronous, blocking HTTP
call, never wrapped in `asyncio.to_thread`/`run_in_executor`. Under
`api/main.py`'s single-process FastAPI server this serializes all concurrent
requests behind whatever Fuseki round-trip is currently in flight; the
`async def` signatures give a false impression of concurrency.

**Fix:** wrap the blocking calls in `asyncio.to_thread(...)`, or switch to
an async SPARQL client.

### 5. Object-matching in `TruthAnchor` compares incompatible strings
When `SemanticParser._get_entity_uri()` can't confidently link an entity
(score \<= 0.7), it fabricates a local URI like `http://local.caf/slippery_road`
([modules/semantic_parser/parser.py:376-390](modules/semantic_parser/parser.py#L376-L390)). That fabricated URI then
becomes `triplet.object_`, which `TruthAnchor._verify_object_match()`
Levenshtein-compares against whatever real KB URIs Fuseki returns
([modules/truth_anchor/verifier.py:169-217](modules/truth_anchor/verifier.py#L169-L217)). Comparing a synthesized
`http://local.caf/...` string against a real KB URI string by edit-distance
is close to meaningless - it will almost always read as a contradiction,
even when the KB does contain the fact, unless the fallback URI happens to
share a long substring with the real one.

**Fix:** at minimum, treat "object failed entity linking" as its own
verification outcome (e.g. `PARTIAL`/`FAILED`, not silently compared as if
it were a resolved URI) rather than feeding a fabricated URI into the
similarity comparison. `experiments/knowledge_base_fvl.py`'s FVL (used by
the `experiments/` path) does not have this problem - worth comparing the
two approaches directly (see #7).

### 6. `SemanticParser._generate_sparql` output is dead code
`SemanticParser.parse()` builds `sparql_query` via `_generate_sparql()`
([modules/semantic_parser/parser.py:311](modules/semantic_parser/parser.py#L311)) and returns it on `ParsedResult`, but
`api/main.py` never reads `parsed_result.sparql_query` - it only ever uses
`parsed_result.triplets`, and `TruthAnchor` builds its own per-triplet query
independently in `_build_sparql_query()`. The combined-triplets query
generator is unused in the actual pipeline.

**Fix:** either delete `_generate_sparql`/remove `sparql_query` from
`ParsedResult`, or actually use it and delete `TruthAnchor._build_sparql_query`.

---

## P2 - architecture / duplication

### 7. Two independent, divergent LLM-loading implementations
- `common/llm_integration.py`: `HuggingFaceCausalLMLayer` + `OllamaLayer`,
  supports 4-bit/8-bit quantization, has explicit Llama-2/Llama-3/Qwen chat
  templates ([common/llm_integration.py:202-264](common/llm_integration.py#L202-L264)), strips `<think>` blocks for
  reasoning models.
- `modules/inference_engine/engine.py`: `InferenceEngine`, vLLM-or-plain-HF,
  no quantization, a single generic chat template via
  `tokenizer.apply_chat_template` with no per-model-family handling, no
  `<think>`-stripping, and a totally different response contract (structured
  `ANSWER:`/`CAUSAL_ASSERTIONS:` text block it then regex-parses back apart,
  vs. `common/llm_integration.py`'s plain string return).

These two exist because `api/`+`modules/` is a service-shaped rewrite of
what `experiments/caf_algorithm.py` + `common/llm_integration.py` already
do, and the rewrite never converged with the original. Same root cause as
#1. Recommend collapsing to one implementation (`common/llm_integration.py`,
since it's the one that actually supports your real workload) that both
`experiments/` scripts and the `modules/inference_engine` service use.

### 8. Two independent, divergent KB-verification implementations
`modules/semantic_parser/parser.py`'s `EntityLinker` +
`modules/truth_anchor/verifier.py`'s `TruthAnchor` (used by the API path)
duplicate what `experiments/knowledge_base_fvl.py`'s `KnowledgeBaseFVL` (used
by the experiments path) already does - entity linking against Fuseki labels,
fuzzy matching, SPARQL verification - as a second, separately-maintained
implementation with different bugs (see #5, which `knowledge_base_fvl.py`
doesn't appear to share). Worth deciding which one is canonical and having
the other delegate to it, same shape as #7.

### 9. `docker-compose.yml` / comments reference a `framework1`/`framework2` split that doesn't exist in this repo
[deployment/docker-compose.yml:3-7](deployment/docker-compose.yml#L3-L7) says "Infra for framework1 (CAF)... framework2
needs neither of these services" and references `docs/SETUP.md`, none of
which exist in this repo (no `framework2/`, no `docs/`). Leftover from
whatever repo this was split out of - either restore the referenced docs or
scrub the stale comment so it doesn't send the next reader looking for files
that aren't there.

---

## P3 - minor

### 10. `CausalValidator._is_causal_predicate` keyword list is mostly dead
Checks for `'causedBy'`, `'resultIn'`, `'leadTo'`, `'produce'`, `'trigger'`,
`'influence'` ([modules/causal_validator/validator.py:138-149](modules/causal_validator/validator.py#L138-L149)), but
`SemanticParser.predicate_templates` ([modules/semantic_parser/parser.py:262-271](modules/semantic_parser/parser.py#L262-L271))
only ever maps `'causes'` to a URI containing a matching keyword; anything
else falls through to a generic `http://local.caf/relation/<verb>` URI built
from spaCy's raw verb token, which is unlikely to literally contain strings
like `resultin` or `leadto`. Most of this keyword list can currently never
match what the parser actually produces.

### 11. Unused `causality:` SPARQL prefix
`SemanticParser._generate_sparql` declares `PREFIX causality: <http://causality.org/>`
([modules/semantic_parser/parser.py:432](modules/semantic_parser/parser.py#L432)) but never uses it in the generated
query body (moot anyway per #6, but worth cleaning up if #6 goes the "keep
and fix" direction instead of "delete").
