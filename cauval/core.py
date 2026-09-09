"""The public Cauval client class."""

import asyncio

from api.models import FinalResponse
from modules.causal_validator.validator import CausalValidator
from modules.inference_engine.client import InferenceEngineClient
from modules.semantic_parser.parser import SemanticParser
from modules.truth_anchor.causal_graph_builder import CausalGraphBuilder
from modules.truth_anchor.verifier import TruthAnchor
from utils.config import Settings, get_settings

from .pipeline import CAFPipeline


class Cauval:
    """
    Programmatic entry point to the CAF pipeline, built on the same
    api/+modules/ implementation as the FastAPI gateway.

    Talks to the LLM over HTTP (modules/inference_engine/server.py, run as
    its own background process) and to Fuseki via SPARQL - both need to be
    up before calling .ask()/.aask(). Zero-config by default: reads
    everything from .env/environment variables exactly like the FastAPI
    gateway's Settings() does. Individual values can be overridden per
    instance without touching the process-wide cached settings.

    Example:
        from cauval import Cauval

        caf = Cauval()
        result = caf.ask("Does high cpu usage cause increased response time?")
        print(result.text, result.verification_status.is_valid)

    Failure behavior: spaCy/model-loading failures raise immediately from
    this constructor (SemanticParser has no fallback - a degraded extractor
    would silently undermine what gets verified). Fuseki/inference-engine
    reachability failures surface as normal exceptions on the first
    .ask()/.aask() call, not at construction - probing them eagerly here
    would make every Cauval() block on a network round trip and would be
    wrong for a Cauval() constructed before the LLM server has finished
    starting up.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        inference_engine_url: str | None = None,
        fuseki_endpoint: str | None = None,
        verification_threshold: float | None = None,
        max_refinement_iterations: int | None = None,
        enable_causal_validation: bool = True,
        spacy_model: str = "en_core_web_sm",
    ):
        settings = settings or get_settings()

        self._inference_engine_url = (
            inference_engine_url or settings.inference_engine_url
        )
        self._verification_threshold = (
            settings.verification_threshold
            if verification_threshold is None
            else verification_threshold
        )
        self._max_refinement_iterations = (
            settings.max_refinement_iterations
            if max_refinement_iterations is None
            else max_refinement_iterations
        )
        self._enable_causal_validation = enable_causal_validation

        fuseki_endpoint = fuseki_endpoint or settings.fuseki_endpoint

        # Constructed once and reused for this instance's lifetime: neither
        # holds a persistent, loop-bound connection (TruthAnchor/EntityLinker
        # both build a fresh SPARQLWrapper per query), so unlike
        # InferenceEngineClient below they're safe to share across separate
        # asyncio.run() calls from .ask().
        self._parser = SemanticParser(
            fuseki_endpoint=fuseki_endpoint, spacy_model=spacy_model
        )
        self._truth_anchor = TruthAnchor(fuseki_endpoint=fuseki_endpoint)
        self._causal_validator = CausalValidator()
        # Level 2/3 routing: walk KB causal edges and answer via do-calculus
        # without LLM involvement.
        self._causal_graph_builder = CausalGraphBuilder(fuseki_endpoint=fuseki_endpoint)

    async def aask(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        max_refinement_iterations: int | None = None,
        verification_threshold: float | None = None,
        enable_causal_validation: bool | None = None,
    ) -> FinalResponse:
        """
        Async-native version of .ask(), for embedding in an already-async
        app (e.g. a FastAPI route). Safe to call concurrently on the same
        Cauval instance.
        """
        # A fresh InferenceEngineClient per call, not shared on self: its
        # httpx.AsyncClient binds internal connection-pool state to the
        # event loop active on first use, and .ask() below runs each call
        # in a brand-new loop via asyncio.run() - a client reused across
        # loops would break on the second call. Constructing one is cheap
        # (no eager I/O), negligible next to an LLM generation call that
        # already has a 300s timeout.
        client = InferenceEngineClient(base_url=self._inference_engine_url)
        try:
            pipeline = CAFPipeline(
                inference=client,
                parser=self._parser,
                truth_anchor=self._truth_anchor,
                causal_validator=self._causal_validator,
                causal_graph_builder=self._causal_graph_builder,
            )
            return await pipeline.run(
                prompt,
                session_id=session_id,
                max_refinement_iterations=(
                    self._max_refinement_iterations
                    if max_refinement_iterations is None
                    else max_refinement_iterations
                ),
                verification_threshold=(
                    self._verification_threshold
                    if verification_threshold is None
                    else verification_threshold
                ),
                enable_causal_validation=(
                    self._enable_causal_validation
                    if enable_causal_validation is None
                    else enable_causal_validation
                ),
            )
        finally:
            await client.close()

    def ask(self, prompt: str, **kwargs) -> FinalResponse:
        """
        Sync entry point: blocks and returns the result directly. Wraps
        .aask() in asyncio.run(), so it cannot be called from inside a
        running event loop (use .aask() there instead - e.g. inside a
        FastAPI handler).
        """
        return asyncio.run(self.aask(prompt, **kwargs))

    def __enter__(self) -> "Cauval":
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    async def __aenter__(self) -> "Cauval":
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass
