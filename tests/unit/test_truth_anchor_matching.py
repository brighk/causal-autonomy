"""
Pure-logic tests for TruthAnchor's matching behavior - no Fuseki needed.
TruthAnchor's constructor doesn't connect eagerly (SPARQLWrapper doesn't
connect until .query()), so these run with zero network dependency.
"""

import asyncio

import pytest

from api.models import Triplet
from modules.truth_anchor.verifier import TruthAnchor


@pytest.fixture
def anchor() -> TruthAnchor:
    return TruthAnchor(fuseki_endpoint="http://localhost:3030/dataset/query")


def test_local_name_slash(anchor: TruthAnchor):
    assert anchor._local_name("http://local.caf/pod_restart") == "pod_restart"


def test_local_name_fragment(anchor: TruthAnchor):
    assert anchor._local_name("http://example.org/thing#frag") == "frag"


def test_local_name_plain_literal(anchor: TruthAnchor):
    assert anchor._local_name("plain_literal") == "plain_literal"


def test_verify_object_match_pod_restart_vs_rotation(anchor: TruthAnchor):
    """
    The exact regression this session found and fixed: comparing full URI
    strings let two different entities sharing a namespace prefix score
    0.877 similarity (comfortably over the 0.8 threshold) purely from the
    shared prefix, falsely reading as VERIFIED. Comparing local names only,
    the real similarity is 0.696 - correctly below threshold.
    """
    results = [{"o": "http://local.caf/pod_rotation"}]
    match = anchor._verify_object_match(
        "http://local.caf/pod_restart", results, threshold=0.8
    )
    assert match["matched"] is False
    assert match["score"] == pytest.approx(0.6956521739130435)


def test_verify_object_match_exact(anchor: TruthAnchor):
    results = [{"o": "http://local.caf/response_time"}]
    match = anchor._verify_object_match(
        "http://local.caf/response_time", results, threshold=0.8
    )
    assert match["matched"] is True
    assert match["score"] == 1.0


def test_verify_object_match_no_results(anchor: TruthAnchor):
    match = anchor._verify_object_match("http://local.caf/anything", [], threshold=0.8)
    assert match["matched"] is False
    assert match["score"] == 0.0
    assert match["expected"] is None


def test_verify_unverifiable_not_contradiction(anchor: TruthAnchor):
    """
    A triplet where entity linking failed (subject_linked/object_linked
    False) must be routed to `unverifiable`, not `contradictions` - it was
    never actually checked against the KB, so reporting it as a
    contradiction would claim more certainty than the system has. This
    path never queries Fuseki at all (verify() skips the SPARQL call
    entirely for unresolved triplets), so no fixture/marker needed.
    """
    triplet = Triplet(
        subject="http://local.caf/xyzzyplugh",
        predicate="http://causality.org/causes",
        object="http://local.caf/quantum_flibbergibbet",
        subject_linked=False,
        object_linked=False,
    )

    result = asyncio.run(anchor.verify([triplet], threshold=0.8))

    assert result.is_valid is False
    assert result.contradictions == []
    assert len(result.unverifiable) == 1
    assert "xyzzyplugh" in result.unverifiable[0]
