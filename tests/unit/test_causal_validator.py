"""Pure-logic tests for CausalGraph and CausalValidator - no I/O at all."""

from api.models import Triplet
from modules.causal_validator.validator import CausalGraph, CausalValidator


def test_causal_graph_no_cycle():
    graph = CausalGraph()
    graph.add_causal_edge("a", "b")
    graph.add_causal_edge("b", "c")
    assert graph.has_cycle() is False


def test_causal_graph_cycle_detection():
    graph = CausalGraph()
    graph.add_causal_edge("a", "b")
    graph.add_causal_edge("b", "a")
    assert graph.has_cycle() is True


def test_causal_graph_bidirectional_violation():
    graph = CausalGraph()
    graph.add_causal_edge("a", "b")
    graph.add_causal_edge("b", "a")
    violations = graph.get_violations()
    assert any("a" in v and "b" in v for v in violations)


def test_causal_graph_no_violations_on_simple_chain():
    graph = CausalGraph()
    graph.add_causal_edge("a", "b")
    graph.add_causal_edge("b", "c")
    assert graph.get_violations() == []


def test_is_causal_predicate_reachable_keywords():
    validator = CausalValidator()
    for predicate in [
        "causes",
        "produce",
        "trigger",
        "influence",
        "http://causality.org/causes",
    ]:
        assert validator._is_causal_predicate(predicate) is True, predicate


def test_is_causal_predicate_removed_keywords():
    """
    These were in the original keyword list but unreachable given what
    SemanticParser.predicate_templates actually produces - 'cause'/'lead'/
    'result' are all intercepted by predicate_templates before ever
    reaching the fallback that would build a URI containing these forms,
    and no single verb lemma produces these camelCase compounds anyway.
    """
    validator = CausalValidator()
    for predicate in ["causedBy", "resultIn", "leadTo"]:
        assert validator._is_causal_predicate(predicate) is False, predicate


def test_find_contradiction_same_subject_predicate_different_object():
    validator = CausalValidator()
    verified = [
        Triplet(subject="a", predicate="p", object="b"),
    ]
    triplet = Triplet(subject="a", predicate="p", object="c")
    contradiction = validator._find_contradiction(triplet, verified)
    assert contradiction is not None
    assert "a" in contradiction and "b" in contradiction and "c" in contradiction


def test_find_contradiction_identical_triplet_no_contradiction():
    validator = CausalValidator()
    verified = [Triplet(subject="a", predicate="p", object="b")]
    triplet = Triplet(subject="a", predicate="p", object="b")
    assert validator._find_contradiction(triplet, verified) is None


def test_find_contradiction_different_predicate_no_contradiction():
    validator = CausalValidator()
    verified = [Triplet(subject="a", predicate="p1", object="b")]
    triplet = Triplet(subject="a", predicate="p2", object="c")
    assert validator._find_contradiction(triplet, verified) is None


def test_causal_validator_reset():
    validator = CausalValidator()
    validator.causal_graph.add_causal_edge("a", "b")
    assert list(validator.causal_graph.graph.nodes())

    validator.reset()

    assert list(validator.causal_graph.graph.nodes()) == []
