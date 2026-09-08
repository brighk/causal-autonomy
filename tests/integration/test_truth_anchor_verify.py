"""TruthAnchor.verify() against the fixture KB - requires live Fuseki."""

import pytest

from api.models import Triplet
from modules.truth_anchor.verifier import TruthAnchor

FUSEKI_QUERY = "http://localhost:3030/dataset/query"

pytestmark = pytest.mark.fuseki


@pytest.fixture
def anchor(fuseki_fixture_kb) -> TruthAnchor:
    return TruthAnchor(fuseki_endpoint=FUSEKI_QUERY)


async def test_verify_matches_real_edge(anchor: TruthAnchor):
    triplet = Triplet(
        subject="http://local.caf/test/cpu_usage",
        predicate="http://causality.org/causes",
        object="http://local.caf/test/response_time",
    )
    result = await anchor.verify([triplet], threshold=0.8)
    assert result.is_valid is True
    assert len(result.matched_triplets) == 1
    assert result.contradictions == []


async def test_verify_contradicts_similar_entity(anchor: TruthAnchor):
    """
    The real fixture edge is health_check_timeout causes pod_restart.
    Claiming health_check_timeout causes pod_rotation is the exact false-
    positive regression this session found and fixed: pod_restart and
    pod_rotation are different, real, correctly-linked entities whose local
    names happen to be similar - should read as a contradiction (0.696
    local-name similarity, below threshold), not a false VERIFIED (which is
    what full-URI comparison used to produce at 0.877) and not unverifiable
    (both entities did link).
    """
    triplet = Triplet(
        subject="http://local.caf/test/health_check_timeout",
        predicate="http://causality.org/causes",
        object="http://local.caf/test/pod_rotation",
    )
    result = await anchor.verify([triplet], threshold=0.8)
    assert result.is_valid is False
    assert len(result.contradictions) == 1
    assert result.unverifiable == []
