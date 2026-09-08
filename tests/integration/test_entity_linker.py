"""EntityLinker against the fixture KB - requires live Fuseki."""

import pytest

from modules.semantic_parser.parser import EntityLinker

FUSEKI_QUERY = "http://localhost:3030/dataset/query"

pytestmark = pytest.mark.fuseki


@pytest.fixture
def linker(fuseki_fixture_kb) -> EntityLinker:
    return EntityLinker(fuseki_endpoint=FUSEKI_QUERY)


def test_entity_linker_exact_match(linker: EntityLinker):
    results = linker.link_entity("caftest_cpu_usage")
    assert len(results) == 1
    assert results[0]["source"] == "exact"
    assert results[0]["score"] == 1.0
    assert results[0]["uri"].endswith("cpu_usage")


def test_entity_linker_no_match(linker: EntityLinker):
    assert linker.link_entity("completely unrelated gibberish xyz123") == []
