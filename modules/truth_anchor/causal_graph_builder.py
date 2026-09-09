"""
Builds a CausalGraph from the live Fuseki KB by:
  1. Linking entity text mentions to KB URIs via rdfs:label / skos:prefLabel
  2. Walking causal-predicate edges outward from those URIs (BFS, bounded)
  3. Resolving discovered URIs back to labels for CausalGraph node IDs

Used by CAFPipeline to handle Level 2/3 (interventional/counterfactual)
queries without LLM involvement — the graph is the sole source of truth for
do-calculus, not the model's priors.
"""

import asyncio

from loguru import logger
from SPARQLWrapper import JSON, SPARQLWrapper

from experiments.intervention_calculus import CausalGraph, normalize_node_id

# Predicate substrings treated as causal when walking the KB graph.
# Kept in sync by hand with modules/causal_validator/validator.py and
# experiments/kb_fvl_with_intervention.py's copies of the same list.
_CAUSAL_PREDICATE_KEYWORDS = (
    "causes",
    "causedby",
    "resultin",
    "leadto",
    "produce",
    "trigger",
    "influence",
)


class CausalGraphBuilder:
    """
    Async SPARQL-backed builder for do-calculus CausalGraphs.

    Constructed once per Cauval/API instance (cheap — no eager I/O), then
    called per Level 2/3 query. Results are cached per unique sorted
    seed-text set so repeated calls within one refinement cycle (same
    prompt, same entity mentions) don't repeat the SPARQL traversal.

    Gracefully returns an empty CausalGraph if Fuseki is unreachable or
    the entities can't be linked — callers fall back to the Level 1 path.
    """

    def __init__(
        self,
        fuseki_endpoint: str = "http://localhost:3030/dataset/query",
        max_hops: int = 2,
    ):
        self.endpoint = fuseki_endpoint
        self.max_hops = max_hops
        self._cache: dict[tuple[str, ...], CausalGraph] = {}

    async def build(self, entity_texts: list[str]) -> CausalGraph:
        """
        Build a CausalGraph from the KB starting from the given entity text mentions.

        Returns an empty CausalGraph if no entities can be linked or no
        causal edges are reachable within max_hops — callers should check
        graph.edges before trusting the result.
        """
        cache_key = tuple(sorted(normalize_node_id(t) for t in entity_texts if t))
        if cache_key in self._cache:
            return self._cache[cache_key]

        graph = CausalGraph()
        visited_uris: set[str] = set()
        frontier: list[str] = []

        for text in entity_texts:
            uri = await self._link_entity(text)
            if uri and uri not in visited_uris:
                visited_uris.add(uri)
                frontier.append(uri)

        for _ in range(self.max_hops):
            if not frontier:
                break
            next_frontier: list[str] = []
            for uri in frontier:
                for neighbor_uri, cause_uri, effect_uri in await self._causal_edges(
                    uri
                ):
                    cause_label = normalize_node_id(
                        await self._resolve_label(cause_uri) or cause_uri
                    )
                    effect_label = normalize_node_id(
                        await self._resolve_label(effect_uri) or effect_uri
                    )
                    graph.add_edge(cause_label, effect_label)
                    if neighbor_uri not in visited_uris:
                        visited_uris.add(neighbor_uri)
                        next_frontier.append(neighbor_uri)
            frontier = next_frontier

        self._cache[cache_key] = graph
        logger.info(
            f"CausalGraphBuilder: built graph with {len(graph.edges)} edges "
            f"from seeds {entity_texts}"
        )
        return graph

    async def _link_entity(self, text: str) -> str | None:
        """Resolve a text label to a KB URI via rdfs:label / skos:prefLabel."""
        normalized = normalize_node_id(text)
        query = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?entity WHERE {{
    {{ ?entity rdfs:label ?label . FILTER(LCASE(STR(?label)) = "{normalized}") }}
    UNION
    {{ ?entity skos:prefLabel ?label . FILTER(LCASE(STR(?label)) = "{normalized}") }}
}}
LIMIT 1
"""
        results = await self._execute_query(query)
        if results:
            return results[0].get("entity")
        return None

    async def _resolve_label(self, uri: str) -> str | None:
        """Resolve a KB URI back to its human-readable label."""
        query = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?label WHERE {{
    {{ <{uri}> rdfs:label ?label }}
    UNION
    {{ <{uri}> skos:prefLabel ?label }}
}}
LIMIT 1
"""
        results = await self._execute_query(query)
        if results:
            return results[0].get("label")
        return None

    async def _causal_edges(self, uri: str) -> list[tuple[str, str, str]]:
        """
        One BFS hop: return (neighbor_uri, cause_uri, effect_uri) tuples for
        every causal-predicate edge touching `uri` in either direction.
        """
        keyword_filter = " || ".join(
            f'CONTAINS(LCASE(STR(?p)), "{kw}")' for kw in _CAUSAL_PREDICATE_KEYWORDS
        )
        query = f"""
SELECT ?p ?o ?s WHERE {{
    {{ <{uri}> ?p ?o . FILTER({keyword_filter}) }}
    UNION
    {{ ?s ?p <{uri}> . FILTER({keyword_filter}) }}
}}
"""
        results = await self._execute_query(query)
        edges: list[tuple[str, str, str]] = []
        for row in results:
            obj_uri = row.get("o")
            subj_uri = row.get("s")
            if obj_uri:
                edges.append((obj_uri, uri, obj_uri))
            elif subj_uri:
                edges.append((subj_uri, subj_uri, uri))
        return edges

    async def _execute_query(self, query: str) -> list[dict[str, str]]:
        def run() -> dict:
            wrapper = SPARQLWrapper(self.endpoint)
            wrapper.setReturnFormat(JSON)
            wrapper.setQuery(query)
            return wrapper.query().convert()

        try:
            response = await asyncio.to_thread(run)
            return [
                {var: val.get("value") for var, val in binding.items()}
                for binding in response.get("results", {}).get("bindings", [])
            ]
        except Exception as e:
            logger.warning(f"CausalGraphBuilder SPARQL query failed: {e}")
            return []
