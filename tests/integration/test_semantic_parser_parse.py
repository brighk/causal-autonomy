"""SemanticParser.parse() end-to-end, real entity linking - requires live Fuseki."""

import pytest

from modules.semantic_parser.parser import SemanticParser

FUSEKI_QUERY = "http://localhost:3030/dataset/query"

pytestmark = pytest.mark.fuseki


async def test_semantic_parser_parse_end_to_end(fuseki_fixture_kb):
    parser = SemanticParser(fuseki_endpoint=FUSEKI_QUERY)

    result = await parser.parse("caftest_cpu_usage causes caftest_response_time.")

    assert len(result.triplets) == 1
    triplet = result.triplets[0]
    assert triplet.subject_linked is True
    assert triplet.object_linked is True
    assert triplet.subject.endswith("cpu_usage")
    assert triplet.object_.endswith("response_time")
