"""
Module C: Truth Anchor (Symbolic)
Database: Apache Jena Fuseki
Protocol: SPARQL 1.1 over HTTP

Provides deterministic verification by grounding LLM outputs in RDF knowledge graphs.
"""

import asyncio
from typing import Any

import Levenshtein
from loguru import logger
from SPARQLWrapper import JSON, SPARQLWrapper

from api.models import Triplet, VerificationResult


class TruthAnchor:
    """
    Symbolic verification engine.

    Queries Apache Jena Fuseki to verify causal assertions against ground truth.
    Uses exact matching and fuzzy matching (Levenshtein distance).
    """

    def __init__(
        self,
        fuseki_endpoint: str = "http://localhost:3030/dataset/query",
        similarity_threshold: float = 0.85,
    ):
        self.endpoint = fuseki_endpoint
        self.similarity_threshold = similarity_threshold

        # Initialize SPARQL wrapper
        self.sparql = SPARQLWrapper(self.endpoint)
        self.sparql.setReturnFormat(JSON)

        logger.info(f"Truth Anchor initialized with endpoint: {fuseki_endpoint}")

    async def verify(
        self, triplets: list[Triplet], threshold: float = 0.8
    ) -> VerificationResult:
        """
        Verify triplets against the knowledge base.

        Logic:
        IF Query($SPARQL$) returns NULL or False:
            THEN Trigger Negative Feedback Loop (System 2 intervention)
        ELSE:
            Commit to Action Layer

        Args:
            triplets: List of RDF triplets to verify
            threshold: Similarity threshold for fuzzy matching

        Returns:
            VerificationResult with validation status
        """
        if not triplets:
            logger.warning("No triplets to verify")
            return VerificationResult(
                is_valid=False,
                matched_triplets=[],
                contradictions=["No triplets extracted from assertion"],
                verification_method="none",
            )

        matched_triplets = []
        contradictions = []
        unverifiable = []
        total_similarity = 0.0

        for triplet in triplets:
            # Entity linking already failed for subject and/or object (see
            # Triplet.subject_linked/object_linked) - the value there is a
            # fabricated http://local.caf/<text> URI, not a real KB entity.
            # Querying it would almost always come back empty and read as
            # "not found in knowledge base", implying we checked and the KB
            # disagrees, when really we never identified what the claim was
            # even referring to. Report that honestly instead, and skip the
            # SPARQL round trip - it can't possibly resolve.
            if not (triplet.subject_linked and triplet.object_linked):
                unresolved = []
                if not triplet.subject_linked:
                    unresolved.append(f"subject '{triplet.subject}'")
                if not triplet.object_linked:
                    unresolved.append(f"object '{triplet.object_}'")
                unverifiable.append(
                    f"Triplet ({triplet.subject}, {triplet.predicate}, "
                    f"{triplet.object_}): {' and '.join(unresolved)} could "
                    f"not be confidently linked to a KB entity"
                )
                continue

            # Execute SPARQL query
            query = self._build_sparql_query(triplet)

            try:
                results = await self._execute_query(query)

                if results:
                    # Verify object matches
                    match_result = self._verify_object_match(
                        triplet.object_, results, threshold
                    )

                    if match_result["matched"]:
                        matched_triplets.append(triplet)
                        total_similarity += match_result["score"]
                    else:
                        contradictions.append(
                            f"Triplet ({triplet.subject}, {triplet.predicate}, "
                            f"{triplet.object_}) contradicts KB: "
                            f"expected {match_result['expected']}"
                        )
                else:
                    # No results found in KB
                    contradictions.append(
                        f"Triplet ({triplet.subject}, {triplet.predicate}, "
                        f"{triplet.object_}) not found in knowledge base"
                    )

            except Exception as e:
                logger.error(f"SPARQL query failed: {e}")
                contradictions.append(f"Query execution error: {e!s}")

        # Calculate overall validity
        is_valid = len(matched_triplets) > 0 and len(contradictions) == 0
        avg_similarity = (
            total_similarity / len(matched_triplets) if matched_triplets else 0.0
        )

        return VerificationResult(
            is_valid=is_valid,
            matched_triplets=matched_triplets,
            contradictions=contradictions,
            unverifiable=unverifiable,
            similarity_score=avg_similarity,
            verification_method="sparql_fuzzy_match",
        )

    def _build_sparql_query(self, triplet: Triplet) -> str:
        """
        Build SPARQL SELECT query for a single triplet.

        Query format:
        SELECT ?o WHERE { <subject> <predicate> ?o }
        """
        query = f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <https://schema.org/>
PREFIX local: <http://local.caf/>

SELECT ?o
WHERE {{
    <{triplet.subject}> <{triplet.predicate}> ?o .
}}
LIMIT 10
        """.strip()

        return query

    async def _execute_query(self, query: str) -> list[dict[str, Any]]:
        """
        Execute SPARQL query against Fuseki.

        Returns:
            List of result bindings
        """
        try:
            # A fresh SPARQLWrapper per call, not self.sparql: setQuery()
            # mutates wrapper state, and asyncio.to_thread frees the event
            # loop while the HTTP call is in flight, so concurrent requests
            # sharing one wrapper could race and execute each other's query.
            def run_query() -> Any:
                wrapper = SPARQLWrapper(self.endpoint)
                wrapper.setReturnFormat(JSON)
                wrapper.setQuery(query)
                return wrapper.query().convert()

            # SPARQLWrapper's query().convert() is a blocking HTTP call;
            # run it off the event loop so concurrent requests don't
            # serialize behind whatever Fuseki round-trip is in flight.
            response = await asyncio.to_thread(run_query)

            results = []
            for binding in response.get("results", {}).get("bindings", []):
                result = {}
                for var, value in binding.items():
                    result[var] = value.get("value")
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"SPARQL execution failed: {e}")
            raise

    @staticmethod
    def _local_name(value: str) -> str:
        """
        Extract the local name/label portion of a URI (the part after the
        last '/' or '#'), for fuzzy comparison. expected_object here is
        frequently a fabricated http://local.caf/<normalized-text> URI (see
        SemanticParser._get_entity_uri's fallback) built in the same
        namespace the KB itself uses - comparing full URI strings lets that
        shared prefix (most of the string, for a short local name) inflate
        the similarity score for two otherwise unrelated entities, e.g.
        "http://local.caf/pod_restart" vs "http://local.caf/pod_rotation"
        scores 0.88 on the full string despite naming different things -
        enough to spuriously clear a 0.8 threshold and report a claim as
        VERIFIED when the KB doesn't actually support it. Plain literals
        (no '/' or '#') pass through unchanged.
        """
        if "#" in value:
            return value.rsplit("#", 1)[-1]
        if "/" in value:
            return value.rsplit("/", 1)[-1]
        return value

    def _verify_object_match(
        self, expected_object: str, results: list[dict[str, Any]], threshold: float
    ) -> dict[str, Any]:
        """
        Verify if the expected object matches any result using:
        1. Exact match (full value)
        2. Fuzzy match (Levenshtein distance, local name only)

        Returns:
            Dict with 'matched' (bool), 'score' (float), 'expected' (str)
        """
        if not results:
            return {"matched": False, "score": 0.0, "expected": None}

        best_match_score = 0.0
        best_match_value = None

        for result in results:
            result_value = result.get("o", "")

            # Exact match - full value, not just the local name: two
            # genuinely different URIs should never count as an exact match
            # just because they share a local name.
            if result_value == expected_object:
                return {"matched": True, "score": 1.0, "expected": result_value}

            # Fuzzy match using Levenshtein ratio, on local names only - see
            # _local_name's docstring for why the full URI isn't compared.
            similarity = Levenshtein.ratio(
                self._local_name(expected_object).lower(),
                self._local_name(result_value).lower(),
            )

            if similarity > best_match_score:
                best_match_score = similarity
                best_match_value = result_value

        # Check if best match meets threshold
        matched = best_match_score >= threshold

        return {
            "matched": matched,
            "score": best_match_score,
            "expected": best_match_value,
        }

    async def load_rdf_data(self, rdf_file_path: str, format: str = "turtle"):
        """
        Load RDF data into Fuseki.

        Args:
            rdf_file_path: Path to RDF file (.ttl, .nt, .rdf)
            format: RDF serialization format (turtle, nt, xml)
        """
        # This would typically use Fuseki's upload API
        # For now, log the intent
        logger.info(f"Loading RDF data from {rdf_file_path} (format: {format})")

        # TODO: Implement actual RDF upload to Fuseki
        # Can use requests library to POST to Fuseki's data endpoint

    def is_healthy(self) -> bool:
        """Check if Fuseki is accessible"""
        try:
            # Simple ASK query to test connection
            test_query = "ASK { ?s ?p ?o }"
            self.sparql.setQuery(test_query)
            self.sparql.query().convert()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
