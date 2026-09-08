# Backlog

Findings from a static read-through of the codebase (2026-09-08), aimed at
getting the project running cleanly on a local RTX 3090 (Qwen3-14B, 4-bit)
and cleaning up architecture drift between `experiments/` (the original
CAF algorithm) and `api/` + `modules/` (a later service rewrite of
overlapping concepts that never fully converged).

Priority: P0 = blocks running the thing, P1 = wrong/silently-broken
behavior, P2 = architecture/duplication cleanup, P3 = minor.

**Status (2026-09-08, second pass):** all P0s fixed and verified - see
"FIXED" notes inline. `SemanticParser` + `TruthAnchor` smoke-tested end to
end against a live Fuseki instance with real data (92 triples, a small
k8s/microservices causal graph): correct claims verify, false claims are
rejected, and 5 concurrent requests were confirmed not to cross-contaminate
each other's SPARQL queries. Fuseki confirmed running and correctly
configured; `docker-compose` plugin installed; `.env` created from
`.env.example` with `LOAD_IN_4BIT=true` for the RTX 3090. One new P0 found
and fixed during verification (see below, "0."). P1 items #4 fixed; #5, #6
left as-is (architecture/behavior calls, see notes) for the next debugging
pass rather than silently redesigned.

**Status (2026-09-08, third pass - live debugging via `caval`):** batch-testing
`caval` against real prompts against the live KB (see #12 for the batch
methodology) surfaced a 0/6 verification rate even on prompts mapping to
real one-hop KB edges - three compounding causes, all found and fixed:
the LLM never gave an atomic answer (#13, prompt fix), a `ccomp` parsing
gap silently dropped triplets from "X causes Y to Z" phrasing plus the
original #12 case (#12, now fully fixed), and a false-positive verification
bug where fabricated and real URIs were compared as full strings instead of
by local name (#14 - the dangerous direction, since it reports something as
VERIFIED when the KB doesn't actually support it). After all three: 7/7 on
a regression suite (5 true one-hop claims verify in 0 iterations, 2 false
claims correctly rejected).

---

## P0 - blocks running it

### 0. `Settings` crashes on any `.env` that sets `FUSEKI_ADMIN_PASSWORD` - FIXED
Not in the original read-through; found while import-smoke-testing `api/main.py`
after the other P0 fixes below. `utils/config.py`'s `Settings` had no
`fuseki_admin_password` field, and pydantic-settings defaults to
`extra="forbid"`. Since `.env.example` (and the README's setup instructions,
and `deployment/docker-compose.yml`) all set `FUSEKI_ADMIN_PASSWORD`, any
`.env` built by following the README crashed `Settings()` - and therefore
`api/main.py` and `modules/inference_engine/server.py` - with a
`pydantic_core.ValidationError: extra_forbidden` before either could even
start. Fixed by declaring `fuseki_admin_password: str | None` on `Settings`
([utils/config.py](utils/config.py)) - it's consumed by docker-compose, not
read by app code, but pydantic-settings validates the full `.env` against
the model regardless of whether a field is used.

### 1. `modules/inference_engine/engine.py` has no quantization path - FIXED
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

### 2. Experiment script SPARQL endpoint defaults point at datasets that don't exist - FIXED
All five stray `conceptnet`/`counterbench` defaults below now default to
`http://localhost:3030/dataset/query`, matching the assembler, README, and
`utils/config.py`. Left `experiments/kb_fvl_with_intervention.py`'s
class-docstring example at `:52` (`counterbench/query`) alone - it's
explicitly illustrating the separate "manual mode" use case (a hypothetical
second Fuseki dataset for CounterBench's fictional-text harness), not a
functional default.

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

### 3. `run_experiment.py --use-real-sparql` crashes: wrong argparse dest - FIXED
Renamed the flag to `--entity-threshold` (matches what the code reads).

`--entity-th` is declared as `parser.add_argument("--entity-th", ...)`
([experiments/run_experiment.py:718-723](experiments/run_experiment.py#L718-L723)), which argparse exposes as
`args.entity_th`. But the code reads `args.entity_threshold` twice
([experiments/run_experiment.py:778](experiments/run_experiment.py#L778) and `:782`) - an attribute that doesn't
exist. `AttributeError: 'Namespace' object has no attribute 'entity_threshold'`
on every `--use-real-sparql` run.

**Fix:** rename the flag to `--entity-threshold`, or read `args.entity_th`.

---

## P1 - wrong or silently-broken behavior

### 4. All SPARQL calls block the FastAPI event loop - FIXED
Wrapped both blocking call sites in `asyncio.to_thread` -
`TruthAnchor._execute_query()` and `EntityLinker`'s call site,
`SemanticParser._get_entity_uri()` (`EntityLinker`'s own methods are
synchronous helpers; the fix is at the async callers that were blocking on
them). While fixing this, also caught and fixed a related bug it would have
introduced: both classes previously reused one `SPARQLWrapper` instance
across calls, mutating it via `setQuery()` immediately before the blocking
call. Once that call runs on a worker thread instead of inline, the event
loop is free to run other coroutines in the gap - so two concurrent
requests sharing one `SPARQLWrapper` could race and execute each other's
query. Fixed by constructing a fresh `SPARQLWrapper` per query in both
classes instead. Verified with 5 concurrent real requests against live
Fuseki data - each got back exactly its own triplet/verification result.
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

### 5. Object-matching in `TruthAnchor` compares incompatible strings - FIXED
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

**FIXED (the dangerous half - false positives):** live-debugging `caval`
against the real KB turned up the other, worse direction this same root
cause produces: `_verify_object_match` was comparing *full URI strings*,
so two *different* entities in the same namespace (`.../pod_restart` vs the
real `.../pod_rotation`) scored 0.88 similarity purely from the ~20
shared prefix characters, cleared the 0.8 threshold, and reported
`VERIFIED` for a claim the KB doesn't actually support - see #14. Fixed by
comparing local names only (`TruthAnchor._local_name()`), not full URIs.
Confirmed the same pair now scores 0.70, correctly below threshold, and a
7-case regression suite still passes.

**FIXED (the conservative half - false negatives):** "object/subject failed
entity linking" is now its own verification outcome instead of silently
comparing a fabricated URI as if it were resolved. `Triplet` gained
`subject_linked`/`object_linked` bool fields (`api/models.py`), set by
`SemanticParser._get_entity_uri()` (`modules/semantic_parser/parser.py`,
now returns `(uri, linked)`); `TruthAnchor.verify()` checks both flags
before ever querying Fuseki and routes unresolved triplets into a new
`VerificationResult.unverifiable: list[str]` field instead of
`contradictions` - skips a SPARQL round trip that couldn't possibly resolve
anyway. `unverifiable` doesn't count for or against `is_valid` (neither
false-positive nor false-negative pressure). `caval/pipeline.py`'s retry-
constraint prompt and final `VerificationFailedError` message combine
`contradictions + unverifiable` so the LLM/caller still gets useful "what
to fix" feedback even when every triplet fell in the unverifiable bucket
(confirmed this mattered - a KB-agreeing case can produce zero
contradictions and would otherwise retry with a blank constraint).
Verified: a triplet with nonsense entities ("Xyzzyplugh causes quantum
flibbergibbet") now reports `contradictions: []`,
`unverifiable: ["...could not be confidently linked..."]`, `is_valid:
False` - not a false contradiction. Real KB regression suite still 7/7.

`experiments/knowledge_base_fvl.py`'s FVL (used by the `experiments/` path)
does not have this problem - not touched, per this session's established
scope (`experiments/`+`common/` stay independent of the `api/`+`modules/`
path `caval` wraps - see #7/#8).

### 6. `SemanticParser._generate_sparql` output is dead code - FIXED
Deleted `_generate_sparql()` and `ParsedResult.sparql_query` entirely
(no other consumer anywhere in the codebase - confirmed via repo-wide
grep) rather than wiring it in; `TruthAnchor` already builds and uses its
own per-triplet query independently, so there was nothing for the
combined-triplets version to actually do. `SemanticParser.parse()`/
`ParsedResult.__init__` no longer take/return a SPARQL query, just
`triplets` and `source_text`. Closes #11 too (the unused `causality:`
prefix lived inside the deleted method).

`SemanticParser.parse()` used to build `sparql_query` via `_generate_sparql()`
([modules/semantic_parser/parser.py:311](modules/semantic_parser/parser.py#L311)) and return it on `ParsedResult`, but
`api/main.py` never read `parsed_result.sparql_query` - it only ever used
`parsed_result.triplets`, and `TruthAnchor` built its own per-triplet query
independently in `_build_sparql_query()`. The combined-triplets query
generator was unused in the actual pipeline.

---

## P2 - architecture / duplication

### 7. Two independent, divergent LLM-loading implementations - FIXED (delegation), otherwise deliberate
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

**Resolved by #1's fix**: `InferenceEngine`'s HF fallback path now delegates
to `common/llm_integration.py`'s `HuggingFaceCausalLMLayer` instead of
reimplementing model loading - quantization, chat templating, and
`<think>`-stripping no longer diverge between the two. The vLLM path in
`engine.py` stays separate, but that's not duplication to collapse - vLLM
and `transformers` are fundamentally different execution engines, so two
call sites are the correct shape, not drift. `common/`'s own dependency on
`experiments/` (via `InferenceLayer`) was separately cut - see the
"decouple common/ from experiments/" commit.

Given `experiments/`+`common/` are an intentionally independent path from
`api/`+`modules/` (the one `caval` wraps) per this session's own scoping
decisions, not touching `experiments/caf_algorithm.py`'s use of
`common/llm_integration.py` further - that's the existing, working,
intended relationship, not a bug.

### 8. Two independent, divergent KB-verification implementations - deliberate, not fixing
`modules/semantic_parser/parser.py`'s `EntityLinker` +
`modules/truth_anchor/verifier.py`'s `TruthAnchor` (used by the API path)
duplicate what `experiments/knowledge_base_fvl.py`'s `KnowledgeBaseFVL` (used
by the experiments path) already does - entity linking against Fuseki labels,
fuzzy matching, SPARQL verification - as a second, separately-maintained
implementation, previously with different bugs (see #5, now fixed on the
`api/`+`modules/` side; `knowledge_base_fvl.py` never had that specific bug).

**Not collapsing these.** This session repeatedly re-confirmed
`experiments/`+`common/` as a deliberately independent, untouched path
(the original `CAFLoop`/benchmark harness), separate from `api/`+`modules/`
(the service-shaped implementation `caval` wraps) - forcing one to delegate
to the other would mean either coupling the benchmark harness's behavior to
`caval`'s (risking silently changing published/reproducible experiment
results) or vice versa. Two implementations of the same idea is real
maintenance cost, but it's the accepted cost of keeping the benchmark path
stable and independent, not an oversight.

### 9. `docker-compose.yml` / comments reference a `framework1`/`framework2` split that doesn't exist in this repo - FIXED
Scrubbed the stale comment and corrected the usage example to the actual
path in this repo (`deployment/docker-compose.yml`, not
`framework1/deployment/docker-compose.yml`).
[deployment/docker-compose.yml:3-7](deployment/docker-compose.yml#L3-L7) says "Infra for framework1 (CAF)... framework2
needs neither of these services" and references `docs/SETUP.md`, none of
which exist in this repo (no `framework2/`, no `docs/`). Leftover from
whatever repo this was split out of - either restore the referenced docs or
scrub the stale comment so it doesn't send the next reader looking for files
that aren't there.

---

## P3 - minor

### 10. `CausalValidator._is_causal_predicate` keyword list is mostly dead - FIXED
Trimmed to `["causes", "produce", "trigger", "influence"]`. Correction to
the original finding: `'produce'`/`'trigger'`/`'influence'` weren't actually
dead - they're reachable via `SemanticParser._get_predicate_uri`'s fallback
`http://local.caf/relation/<verb-lemma>` URI when the lemma is exactly one
of those words (verified by reading `_get_predicate_uri`). Only
`'causedBy'`/`'resultIn'`/`'leadTo'` were truly unreachable: no single verb
lemma the fallback builds a URI from produces those camelCase compound
forms, and `'cause'`/`'lead'`/`'result'` are all intercepted by
`predicate_templates` before ever reaching the fallback - so those three
could never match anything. Dropped only those three, kept the rest.

### 12. Concrete example of the "naive extractor" gotcha already in README - FIXED
Found while smoke-testing #4's fix against real KB data (a small
k8s/microservices causal graph, 92 triples, loaded via the companion
causal-discovery repo per the README). The KB contains a genuinely true
edge `response_time -> causes -> health_check_timeout`, but asking
`SemanticParser` to parse "response time causes health check timeout."
extracts zero triplets - not a contradiction, a silent miss. Cause: spaCy
tags "timeout" as `ccomp` (clausal complement) rather than `dobj` for this
sentence -
```
response   compound -> time
time       nsubj    -> causes
causes     ROOT
health     compound -> check
check      compound -> timeout
timeout    ccomp    -> causes
```
- an artifact of spaCy's general-domain model on compound technical nouns
("health check timeout" reads to it like it could be a reduced clause).

**FIXED**, alongside a second, related `ccomp` case found via live batch-testing
`caval`: "X causes Y to Z" (accusative-with-infinitive, e.g. "causes database
connections to rise") also attaches as a `ccomp` of "causes", but as a
*genuine* infinitival clause with its own subject ("connections" is `nsubj`
of "rise") - a very common causal phrasing an LLM defaults to, and before
this fix it also silently produced zero triplets. `_parse_text`'s
dobj/attr/pobj/prep-pobj child search ([modules/semantic_parser/parser.py](modules/semantic_parser/parser.py))
now also handles `ccomp`: if the ccomp token has its own nsubj, that's the
object (the "to Z" case); if it doesn't (a plain noun mis-tagged as ccomp,
the original "timeout" case above), the ccomp token itself is the object.
Both confirmed fixed individually and via a 7-case regression suite
end-to-end through `caval`.

### 13. LLM never gave an atomic, extractable answer - FIXED (prompt)
Batch-testing `caval` against 6 prompts mapping to real one-hop KB edges got
0/6 `VERIFIED`, not because the claims were false but because the model
(Qwen3-14B, HF/4-bit path) always elaborated into a multi-step explanation
("garbage collection triggers memory reclamation, which increases CPU
utilization...") instead of stating the one-hop fact plainly - the naive SVO
parser (see #12) can't reduce a paragraph back down to the KB's atomic
`(subject, causes, object)` shape.

**Fix:** tightened `InferenceEngine._causal_task_instructions()`
([modules/inference_engine/engine.py](modules/inference_engine/engine.py))
- shared by both the vLLM and HF generation paths - to demand exactly one
short "X causes Y" sentence (2-4 word noun phrases, no subordinate clauses,
no mechanism explanation) plus a single causal-assertion bullet reusing the
identical wording, rather than vaguely asking for "a clear, accurate answer"
plus "precise, verifiable statements". After the fix: 5/5 true one-hop
claims verified immediately (0 refinement iterations) on the same KB.
Requires restarting the running inference-engine server process to take
effect (prompt construction happens per-request, but the module is only
imported once at process start).

Known remaining gap: only fixes convergence for direct one-hop facts: a
question whose true answer requires multiple hops (not present in this KB
yet) will still need the model to name intermediate KB nodes explicitly,
which this prompt doesn't yet ask for.

### 14. False-positive verification: full-URI Levenshtein comparison - FIXED
Sharper, more dangerous variant of #5's root cause, found via live
batch-testing: `TruthAnchor._verify_object_match()` Levenshtein-compared
*full URI strings*. `"health check timeout causes pod restart"` (the model's
paraphrase; the KB only has `pod_rotation`, not `pod_restart`) verified as
`VERIFIED: True`, because `http://local.caf/pod_restart` vs the real
`http://local.caf/pod_rotation` scores 0.877 similarity - not because
"restart" and "rotation" are alike, but because ~20 of the ~30 characters
are the shared `http://local.caf/pod_` namespace prefix, comfortably over
the 0.8 threshold. Confirmed via direct calculation
(`Levenshtein.ratio(a.lower(), b.lower())` on the full strings) before
fixing. This is the dangerous direction for a system whose entire premise
is deterministic truth-grounding: it reports confidence in a claim the KB
does not actually support.

**Fix:** added `TruthAnchor._local_name()` (extracts the part after the
last `/` or `#`) and compare local names only for the fuzzy match, keeping
the full-string comparison for the exact-match fast path. Same pair now
scores 0.696, correctly below threshold. Confirmed via a 7-case regression
suite (5 true claims still verify, 2 false claims still correctly reject)
that this didn't introduce new false negatives.

### 11. Unused `causality:` SPARQL prefix - FIXED
Resolved by #6: `_generate_sparql` (which declared this unused prefix) was
deleted entirely, not kept-and-fixed, so there's nothing left to clean up.
