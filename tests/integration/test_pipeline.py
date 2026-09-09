"""
CAFPipeline end-to-end (generate -> parse -> verify -> constrain ->
regenerate) against the fixture KB, using StubInference instead of a real
LLM - covers the orchestration logic without needing a GPU. Requires live
Fuseki.
"""

import pytest

from cauval.exceptions import VerificationFailedError
from cauval.pipeline import CAFPipeline
from modules.causal_validator.validator import CausalValidator
from modules.semantic_parser.parser import SemanticParser
from modules.truth_anchor.causal_graph_builder import CausalGraphBuilder
from modules.truth_anchor.verifier import TruthAnchor
from tests.helpers import StubInference

FUSEKI_QUERY = "http://localhost:3030/dataset/query"

pytestmark = pytest.mark.fuseki


def make_pipeline(inference) -> CAFPipeline:
    """Level 1 pipeline (no causal_graph_builder → do-calculus routing disabled)."""
    return CAFPipeline(
        inference=inference,
        parser=SemanticParser(fuseki_endpoint=FUSEKI_QUERY),
        truth_anchor=TruthAnchor(fuseki_endpoint=FUSEKI_QUERY),
        causal_validator=CausalValidator(),
    )


def make_pipeline_l23(inference) -> CAFPipeline:
    """Full pipeline with Level 2/3 do-calculus routing enabled."""
    return CAFPipeline(
        inference=inference,
        parser=SemanticParser(fuseki_endpoint=FUSEKI_QUERY),
        truth_anchor=TruthAnchor(fuseki_endpoint=FUSEKI_QUERY),
        causal_validator=CausalValidator(),
        causal_graph_builder=CausalGraphBuilder(fuseki_endpoint=FUSEKI_QUERY),
    )


async def test_pipeline_verifies_true_claim(fuseki_fixture_kb):
    pipeline = make_pipeline(
        StubInference("caftest_cpu_usage causes caftest_response_time.")
    )

    result = await pipeline.run("does caftest_cpu_usage cause caftest_response_time?")

    assert result.verification_status.is_valid is True
    assert result.refinement_iterations == 0


async def test_pipeline_rejects_then_verifies_on_retry(fuseki_fixture_kb):
    """The real fixture edge is health_check_timeout causes pod_restart.
    First answer names the wrong effect (pod_rotation); the retry,
    post-constraint, gets it right."""
    pipeline = make_pipeline(
        StubInference(
            [
                "caftest_health_check_timeout causes caftest_pod_rotation.",
                "caftest_health_check_timeout causes caftest_pod_restart.",
            ]
        )
    )

    result = await pipeline.run("does health check timeout cause pod restart?")

    assert result.verification_status.is_valid is True
    assert result.refinement_iterations == 1


async def test_pipeline_raises_after_max_iterations(fuseki_fixture_kb):
    pipeline = make_pipeline(
        StubInference("caftest_cpu_usage causes an unrelated nonsense concept.")
    )

    with pytest.raises(VerificationFailedError) as exc_info:
        await pipeline.run("bogus prompt", max_refinement_iterations=2)

    assert exc_info.value.refinement_iterations == 2
    assert exc_info.value.contradictions


async def test_pipeline_level2_do_calculus_no_llm(fuseki_fixture_kb):
    """
    Level 2 path: a counterfactual query is answered by do-calculus alone,
    the LLM is never called.

    Fixture edge: caftest_cpu_usage causes caftest_response_time.
    Query: if we prevent caftest_cpu_usage, caftest_response_time would NOT occur
    (it is a direct descendant — cutting the cause breaks the chain).
    """
    stub = StubInference("this text should never be used")
    pipeline = make_pipeline_l23(stub)

    result = await pipeline.run(
        "Would caftest_response_time occur if we prevent caftest_cpu_usage?"
    )

    assert result.causal_level == 2
    assert result.verification_status.verification_method == "do_calculus"
    assert result.verification_status.is_valid is True
    assert "No" in result.text
    # The LLM must not have been called — do-calculus answered it entirely.
    assert stub.call_count == 0


async def test_pipeline_level2_falls_back_to_level1_without_builder(fuseki_fixture_kb):
    """
    Without a CausalGraphBuilder, a counterfactual-phrased prompt falls
    through to the Level 1 path and the LLM IS called.
    """
    stub = StubInference("caftest_cpu_usage causes caftest_response_time.")
    pipeline = make_pipeline(stub)  # no causal_graph_builder

    result = await pipeline.run(
        "Would caftest_response_time occur if we prevent caftest_cpu_usage?"
    )

    # Level 1 path ran (LLM was called at least once)
    assert stub.call_count >= 1
    assert result.causal_level == 1
