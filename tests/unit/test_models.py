"""Pure-logic tests for api.models - no I/O."""

from api.models import Triplet


def test_triplet_object_alias():
    triplet = Triplet(subject="s", predicate="p", object="o")
    assert triplet.object_ == "o"


def test_triplet_linked_flags_default_true():
    triplet = Triplet(subject="s", predicate="p", object="o")
    assert triplet.subject_linked is True
    assert triplet.object_linked is True


def test_triplet_linked_flags_explicit_false():
    triplet = Triplet(
        subject="s",
        predicate="p",
        object="o",
        subject_linked=False,
        object_linked=False,
    )
    assert triplet.subject_linked is False
    assert triplet.object_linked is False
