"""
Shared fixtures. Fuseki-dependent tests are marked `@pytest.mark.fuseki`
and use `fuseki_fixture_kb`, which skips (not fails) if Fuseki isn't
reachable, so the suite is safe to run on a machine without Docker/Fuseki
up - see tests/unit/ (no Fuseki needed at all) vs tests/integration/.

Fixture entity labels use a "caftest_" marker baked into the label text
itself, not just the URI: EntityLinker's fuzzy search does CONTAINS-based
word matching across every label in the store regardless of URI namespace,
so a fixture using realistic-but-generic labels (e.g. "cpu usage") would
risk matching real KB content on a dev Fuseki that also has the real
causal-discovery dataset loaded (confirmed live: the real KB has
http://local.caf/cpu_usage labeled literally "cpu_usage"). The "caftest_"
marker keeps fixture queries isolated regardless of what else is loaded.
"""

import httpx
import pytest

FUSEKI_QUERY = "http://localhost:3030/dataset/query"
FUSEKI_UPDATE = "http://localhost:3030/dataset/update"

_TRIPLES = """
    t:cpu_usage rdfs:label "caftest_cpu_usage"@en .
    t:cpu_usage a caf:CausalVariable .
    t:response_time rdfs:label "caftest_response_time"@en .
    t:response_time a caf:CausalVariable .
    t:cpu_usage c:causes t:response_time .

    t:pod_restart rdfs:label "caftest_pod_restart"@en .
    t:pod_restart a caf:CausalVariable .
    t:pod_rotation rdfs:label "caftest_pod_rotation"@en .
    t:pod_rotation a caf:CausalVariable .
    t:health_check_timeout rdfs:label "caftest_health_check_timeout"@en .
    t:health_check_timeout a caf:CausalVariable .
    t:health_check_timeout c:causes t:pod_restart .
"""
# Note: fixture URI local names must themselves be as distinct as the real
# regression scenario (pod_restart vs pod_rotation) - using generic
# sequential placeholders (entity_c, entity_d, ...) here would accidentally
# reproduce the exact false-positive-via-similar-local-name bug this
# fixture exists to regression-test, since TruthAnchor compares URI local
# names, not rdfs:label text. Caught by the tests themselves on first run.

_INSERT_DATA = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX c: <http://causality.org/>
PREFIX t: <http://local.caf/test/>
PREFIX caf: <http://caf.ai/ontology/>
INSERT DATA {{
{_TRIPLES}
}}
"""

_DELETE_DATA = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX c: <http://causality.org/>
PREFIX t: <http://local.caf/test/>
PREFIX caf: <http://caf.ai/ontology/>
DELETE DATA {{
{_TRIPLES}
}}
"""


@pytest.fixture(scope="session")
def fuseki_available() -> bool:
    try:
        return httpx.get("http://localhost:3030/$/ping", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.fixture(scope="module")
def fuseki_fixture_kb(fuseki_available: bool):
    """Inserts the fixture KB before the module's tests run, deletes it after."""
    if not fuseki_available:
        pytest.skip(
            "Fuseki not reachable at localhost:3030 - start it with "
            "`docker compose -f deployment/docker-compose.yml up -d`"
        )
    resp = httpx.post(FUSEKI_UPDATE, data={"update": _INSERT_DATA}, timeout=10.0)
    resp.raise_for_status()
    yield
    httpx.post(FUSEKI_UPDATE, data={"update": _DELETE_DATA}, timeout=10.0)
