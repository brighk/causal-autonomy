"""
CAFPipeline end-to-end (generate -> parse -> verify -> constrain ->
regenerate) against the fixture KB, using StubInference instead of a real
LLM - covers the orchestration logic without needing a GPU. Requires live
Fuseki.
"""

import pytest

from caval.exceptions import VerificationFailedError
from caval.pipeline import CAFPipeline
from modules.causal_validator.validator import CausalValidator
from modules.semantic_parser.parser import SemanticParser
from modules.truth_anchor.verifier import TruthAnchor
from tests.helpers import StubInference

FUSEKI_QUERY = "http://localhost:3030/dataset/query"

pytestmark = pytest.mark.fuseki


def make_pipeline(inference) -> CAFPipeline:
    return CAFPipeline(
        inference=inference,
        parser=SemanticParser(fuseki_endpoint=FUSEKI_QUERY),
        truth_anchor=TruthAnchor(fuseki_endpoint=FUSEKI_QUERY),
        causal_validator=CausalValidator(),
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
