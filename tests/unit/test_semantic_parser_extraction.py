"""
Dependency-parse triplet-extraction tests. Uses a real spaCy model (already
a project dependency, no network needed once installed) but monkeypatches
EntityLinker.link_entity to always miss, so _get_entity_uri deterministically
takes the fabricated-URI fallback path - no Fuseki needed. These lock in
three same-day fixes to the object-search loop in
SemanticParser._parse_text.
"""

import asyncio

import pytest

from modules.semantic_parser.parser import SemanticParser


@pytest.fixture
def parser() -> SemanticParser:
    p = SemanticParser(fuseki_endpoint="http://localhost:3030/dataset/query")
    p.entity_linker.link_entity = lambda text, top_k=1: []
    return p


def test_ccomp_causes_to_infinitive(parser: SemanticParser):
    """
    "X causes Y to Z" (accusative-with-infinitive): spaCy attaches "Z" as a
    ccomp of "causes" with its own nsubj "Y" - before the fix, the object
    search (which only checked dobj/attr/pobj/prep->pobj) found nothing
    here and silently produced zero triplets from a perfectly clean
    sentence.
    """
    triplets = asyncio.run(
        parser._parse_text(
            "Missing connection pool limit causes database connections to rise."
        )
    )
    assert len(triplets) == 1
    assert triplets[0].subject.endswith("missing_connection_pool_limit")
    assert triplets[0].predicate == "http://causality.org/causes"
    assert triplets[0].object_.endswith("database_connections")


def test_ccomp_plain_noun_mistagged(parser: SemanticParser):
    """
    A plain noun spaCy mis-tags as ccomp with no nsubj of its own (not a
    genuine infinitival clause) - before the fix, also zero triplets. Falls
    back to treating the ccomp token itself (with its compound modifiers)
    as the object.
    """
    triplets = asyncio.run(
        parser._parse_text("Elevated response time causes health check timeout.")
    )
    assert len(triplets) == 1
    assert triplets[0].object_.endswith("health_check_timeout")


def test_prep_pobj_leads_to(parser: SemanticParser):
    """ "X leads to Y" - Y attaches as the pobj of a prep that is itself the
    verb's direct child, not a direct object of the verb."""
    triplets = asyncio.run(
        parser._parse_text("Garbage collection cycles lead to increased cpu usage.")
    )
    assert len(triplets) == 1
    assert triplets[0].subject.endswith("garbage_collection_cycles")
    assert triplets[0].predicate == "http://causality.org/causes"
    assert triplets[0].object_.endswith("increased_cpu_usage")


def test_fabricated_uri_marks_unlinked(parser: SemanticParser):
    """With entity linking always missing, both ends of the triplet should
    be flagged unlinked so TruthAnchor treats them as unverifiable, not
    silently comparable as resolved KB entities."""
    triplets = asyncio.run(parser._parse_text("Rain causes flooding."))
    assert len(triplets) == 1
    assert triplets[0].subject_linked is False
    assert triplets[0].object_linked is False
