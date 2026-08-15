"""
Knowledge Base SPARQL FVL Implementation for CAF
=================================================
Production-ready Formal Verification Layer with real SPARQL/RDF verification.

This implementation replaces SimulatedFVL with actual knowledge base integration:
- Entity linking with spaCy NER and fuzzy matching
- SPARQL query construction and execution
- Caching for performance optimization
- Robust error handling and logging

Requirements:
    pip install rdflib SPARQLWrapper spacy fuzzywuzzy python-Levenshtein
    python -m spacy download en_core_web_sm
"""

import re
import time
from typing import List, Optional, Any, Dict, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
import logging

try:
    from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions
except ImportError:
    raise ImportError("Install SPARQLWrapper: pip install SPARQLWrapper")

try:
    import spacy
    from spacy.language import Language
except ImportError:
    raise ImportError("Install spacy: pip install spacy && python -m spacy download en_core_web_sm")

try:
    from fuzzywuzzy import fuzz
except ImportError:
    # Fallback to difflib if fuzzywuzzy not available
    fuzz = None

from experiments.caf_algorithm import (
    FormalVerificationLayer,
    RDFTriplet,
    VerificationResult,
    VerificationStatus
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class EntityMapping:
    """Entity text mapped to KB URI with confidence."""
    text: str
    uri: str
    confidence: float
    method: str  # exact, fuzzy, cached


@dataclass
class SPARQLQueryResult:
    """SPARQL query execution result."""
    success: bool
    result: Any
    latency_ms: float
    error: Optional[str] = None


class KnowledgeBaseFVL(FormalVerificationLayer):
    """
    Production Formal Verification Layer with real SPARQL verification.

    Key Features:
    - Entity linking with NER and fuzzy matching
    - SPARQL query construction and execution
    - URI caching for performance
    - Robust error handling
    - Metrics tracking

    Architecture:
    1. Parse: Extract (subject, predicate, object) triplets via spaCy
    2. Link: Map entity mentions to KB URIs
    3. Query: Construct SPARQL ASK/SELECT queries
    4. Verify: Execute queries and return verification status
    """

    def __init__(
        self,
        sparql_endpoint: str = "http://localhost:3030/dataset/query",
        entity_threshold: float = 0.7,
        enable_fuzzy_match: bool = True,
        fuzzy_match_limit: int = 10,
        cache_size: int = 1000,
        query_timeout: int = 10,
        spacy_model: str = "en_core_web_sm"
    ):
        """
        Initialize Knowledge Base FVL.

        Args:
            sparql_endpoint: SPARQL endpoint URL (Jena/GraphDB)
            entity_threshold: Minimum similarity for fuzzy entity linking (0-1)
            enable_fuzzy_match: Enable fuzzy string matching for entities
            fuzzy_match_limit: Max candidates to consider for fuzzy matching
            cache_size: LRU cache size for entity URI mappings
            query_timeout: SPARQL query timeout in seconds
            spacy_model: spaCy model to use for NER
        """
        self.sparql_endpoint = sparql_endpoint
        self.entity_threshold = entity_threshold
        self.enable_fuzzy_match = enable_fuzzy_match
        self.fuzzy_match_limit = fuzzy_match_limit
        self.query_timeout = query_timeout

        # Initialize SPARQL wrapper
        self.sparql = SPARQLWrapper(sparql_endpoint)
        self.sparql.setReturnFormat(JSON)
        self.sparql.setTimeout(query_timeout)

        # Load spaCy for NLP
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            logger.error(f"spaCy model '{spacy_model}' not found. Run: python -m spacy download {spacy_model}")
            raise

        # Entity URI cache (in-memory)
        self.entity_cache: Dict[str, EntityMapping] = {}
        self.cache_size = cache_size

        # Metrics
        self.stats = {
            "queries_executed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "exact_matches": 0,
            "fuzzy_matches": 0,
            "failed_links": 0,
            "total_query_time_ms": 0.0
        }

        logger.info(f"KnowledgeBaseFVL initialized with endpoint: {sparql_endpoint}")

    def parse(self, response: str) -> List[RDFTriplet]:
        """
        Extract RDF triplets from LLM response using dependency parsing.

        Process:
        1. Tokenize and parse with spaCy
        2. Extract subject-verb-object patterns
        3. Resolve noun chunks for full entity names
        4. Construct RDFTriplet objects

        Args:
            response: LLM-generated text

        Returns:
            List of extracted RDF triplets
        """
        if not response or not response.strip():
            return []

        doc = self.nlp(response)
        triplets = []

        # Extract SVO patterns from dependency parse
        for sent in doc.sents:
            # Find nominal subjects
            subjects = [token for token in sent if token.dep_ in ("nsubj", "nsubjpass")]

            for subj in subjects:
                # Get the verb (predicate)
                verb = subj.head

                # Find objects (direct object, prepositional object, attribute)
                objects = [child for child in verb.children
                          if child.dep_ in ("dobj", "pobj", "attr", "oprd")]

                # Also check for compound predicates
                aux_verbs = [child for child in verb.children if child.dep_ == "aux"]

                for obj in objects:
                    # Extract full entity text (noun chunks)
                    subj_text = self._extract_entity_text(subj, doc)
                    obj_text = self._extract_entity_text(obj, doc)

                    # Normalize predicate (combine aux + main verb if needed)
                    if aux_verbs:
                        predicate = f"{aux_verbs[0].lemma_}_{verb.lemma_}"
                    else:
                        predicate = verb.lemma_

                    # Skip trivial triplets
                    if len(subj_text) < 2 or len(obj_text) < 2:
                        continue

                    triplet = RDFTriplet(
                        subject=subj_text.lower(),
                        predicate=predicate.lower(),
                        obj=obj_text.lower(),
                        confidence=0.8,  # Could use dependency parse confidence
                        source_span=sent.text
                    )
                    triplets.append(triplet)

        logger.debug(f"Parsed {len(triplets)} triplets from response")
        return triplets

    def _extract_entity_text(self, token, doc) -> str:
        """
        Extract full entity text including compounds and modifiers.

        Args:
            token: spaCy token
            doc: spaCy doc for accessing noun chunks

        Returns:
            Full entity text
        """
        # Try to find containing noun chunk
        for chunk in doc.noun_chunks:
            if token in chunk:
                return chunk.text.strip()

        # Fallback: extract with compound words
        entity_tokens = [token]

        # Add left compounds
        for child in token.children:
            if child.dep_ in ("compound", "amod"):
                entity_tokens.insert(0, child)

        return " ".join([t.text for t in entity_tokens]).strip()

    def verify(
        self,
        triplets: List[RDFTriplet],
        knowledge_base: Any = None  # Not used, kept for interface compatibility
    ) -> List[VerificationResult]:
        """
        Verify triplets against knowledge base via SPARQL.

        Process for each triplet:
        1. Link subject and object to KB URIs
        2. Construct SPARQL ASK query
        3. Execute query and check for exact match
        4. Check for contradiction (negation)
        5. Optionally fuzzy match for partial verification

        Args:
            triplets: List of RDF triplets to verify
            knowledge_base: Unused (endpoint configured at init)

        Returns:
            List of verification results
        """
        results = []

        for triplet in triplets:
            result = self._verify_single_triplet(triplet)
            results.append(result)

        return results

    def _verify_single_triplet(self, triplet: RDFTriplet) -> VerificationResult:
        """Verify a single triplet with full error handling."""

        # Step 1: Entity linking
        subj_mapping = self._link_entity(triplet.subject)
        obj_mapping = self._link_entity(triplet.obj)

        if not subj_mapping or not obj_mapping:
            # Entity not in KB
            logger.debug(f"Failed to link entities: {triplet.subject} or {triplet.obj}")
            return VerificationResult(
                triplet=triplet,
                status=VerificationStatus.FAILED,
                kb_support=False,
                contradiction_found=False,
                supporting_facts=[],
                contradicting_facts=[],
                confidence_score=0.0
            )

        # Step 2: Construct and execute SPARQL query
        query = self._build_ask_query(subj_mapping.uri, triplet.predicate, obj_mapping.uri)
        query_result = self._execute_sparql_query(query)

        if not query_result.success:
            logger.warning(f"SPARQL query failed: {query_result.error}")
            return VerificationResult(
                triplet=triplet,
                status=VerificationStatus.FAILED,
                kb_support=False,
                contradiction_found=False,
                confidence_score=0.0
            )

        # Step 3: Interpret results
        is_verified = query_result.result.get("boolean", False)

        if is_verified:
            return VerificationResult(
                triplet=triplet,
                status=VerificationStatus.VERIFIED,
                kb_support=True,
                contradiction_found=False,
                supporting_facts=[f"{subj_mapping.uri} -> {obj_mapping.uri}"],
                confidence_score=min(subj_mapping.confidence, obj_mapping.confidence)
            )

        # Step 4: Check for contradiction
        contradiction_found = self._check_contradiction(
            subj_mapping.uri,
            triplet.predicate,
            obj_mapping.uri
        )

        if contradiction_found:
            return VerificationResult(
                triplet=triplet,
                status=VerificationStatus.CONTRADICTION,
                kb_support=False,
                contradiction_found=True,
                contradicting_facts=[f"Negation of {triplet} found in KB"],
                confidence_score=0.0
            )

        # Step 5: Try fuzzy match for partial verification
        if self.enable_fuzzy_match:
            fuzzy_score = self._fuzzy_verify(subj_mapping.uri, triplet.predicate, obj_mapping.uri)
            if fuzzy_score > 0.5:
                return VerificationResult(
                    triplet=triplet,
                    status=VerificationStatus.PARTIAL,
                    kb_support=True,
                    contradiction_found=False,
                    confidence_score=fuzzy_score
                )

        # Default: Failed verification
        return VerificationResult(
            triplet=triplet,
            status=VerificationStatus.FAILED,
            kb_support=False,
            contradiction_found=False,
            confidence_score=0.0
        )

    def _link_entity(self, entity_text: str) -> Optional[EntityMapping]:
        """
        Link entity mention to KB URI.

        Strategy:
        1. Check cache
        2. Try exact label match
        3. Try fuzzy string matching (if enabled)

        Args:
            entity_text: Entity mention from text

        Returns:
            EntityMapping with URI and confidence, or None
        """
        # Normalize text
        entity_text = entity_text.strip().lower()

        # Check cache
        if entity_text in self.entity_cache:
            self.stats["cache_hits"] += 1
            mapping = self.entity_cache[entity_text]
            logger.debug(f"Cache hit: {entity_text} -> {mapping.uri}")
            return mapping

        self.stats["cache_misses"] += 1

        # Try exact match
        exact_uri = self._exact_entity_match(entity_text)
        if exact_uri:
            self.stats["exact_matches"] += 1
            mapping = EntityMapping(
                text=entity_text,
                uri=exact_uri,
                confidence=1.0,
                method="exact"
            )
            self._cache_entity(entity_text, mapping)
            return mapping

        # Try fuzzy match
        if self.enable_fuzzy_match:
            fuzzy_result = self._fuzzy_entity_search(entity_text)
            if fuzzy_result:
                self.stats["fuzzy_matches"] += 1
                uri, confidence = fuzzy_result
                mapping = EntityMapping(
                    text=entity_text,
                    uri=uri,
                    confidence=confidence,
                    method="fuzzy"
                )
                self._cache_entity(entity_text, mapping)
                return mapping

        self.stats["failed_links"] += 1
        logger.debug(f"Failed to link entity: {entity_text}")
        return None

    def _exact_entity_match(self, entity_text: str) -> Optional[str]:
        """Find exact label match in KB."""
        entity_literal = self._sparql_string_literal(entity_text)
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

        SELECT ?uri WHERE {{
            {{ ?uri rdfs:label {entity_literal}@en . }}
            UNION
            {{ ?uri skos:prefLabel {entity_literal}@en . }}
            UNION
            {{ ?uri rdfs:label {entity_literal} . }}
        }} LIMIT 1
        """

        result = self._execute_sparql_query(query)

        if result.success and result.result.get("results", {}).get("bindings"):
            uri = result.result["results"]["bindings"][0]["uri"]["value"]
            logger.debug(f"Exact match: {entity_text} -> {uri}")
            return uri

        return None

    def _fuzzy_entity_search(self, entity_text: str) -> Optional[Tuple[str, float]]:
        """
        Find similar entities using fuzzy string matching.

        Returns:
            Tuple of (uri, confidence) or None
        """
        # Use first word as filter to reduce search space
        first_word = entity_text.split()[0] if entity_text else entity_text
        if not first_word:
            return None
        first_word_literal = self._sparql_string_literal(first_word.lower())

        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?uri ?label WHERE {{
            ?uri rdfs:label ?label .
            FILTER(LANG(?label) = "en" || LANG(?label) = "")
            FILTER(CONTAINS(LCASE(STR(?label)), {first_word_literal}))
        }} LIMIT {self.fuzzy_match_limit}
        """

        result = self._execute_sparql_query(query)

        if not result.success or not result.result.get("results", {}).get("bindings"):
            return None

        # Find best fuzzy match
        best_uri = None
        best_score = 0.0

        for binding in result.result["results"]["bindings"]:
            label = binding["label"]["value"].lower()
            score = self._similarity(entity_text, label)

            if score > best_score and score >= self.entity_threshold:
                best_score = score
                best_uri = binding["uri"]["value"]

        if best_uri:
            logger.debug(f"Fuzzy match: {entity_text} -> {best_uri} (score: {best_score:.2f})")
            return (best_uri, best_score)

        return None

    def _similarity(self, a: str, b: str) -> float:
        """
        Similarity score that also credits a short, clean phrase for matching
        *part* of a longer label (e.g. an LLM saying "habitat destruction"
        against a KB label like "habitat destruction which in turn leads to
        biodiversity loss" - a full-string ratio penalizes the length gap
        even though the shorter phrase is an exact match against a prefix).
        Plain ratio() alone would reject that kind of match well below
        entity_threshold, so entity linking would silently fail on any KB
        built from naive, multi-clause text extraction.
        """
        if fuzz:
            return max(fuzz.ratio(a, b), fuzz.partial_ratio(a, b)) / 100.0

        ratio = SequenceMatcher(None, a, b).ratio()
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        if not shorter:
            return ratio

        best_partial = 0.0
        for i in range(len(longer) - len(shorter) + 1):
            window = longer[i:i + len(shorter)]
            best_partial = max(best_partial, SequenceMatcher(None, shorter, window).ratio())
        return max(ratio, best_partial)

    def _cache_entity(self, entity_text: str, mapping: EntityMapping):
        """Add entity mapping to cache with LRU eviction."""
        if len(self.entity_cache) >= self.cache_size:
            # Simple eviction: remove first entry (not true LRU but acceptable)
            self.entity_cache.pop(next(iter(self.entity_cache)))

        self.entity_cache[entity_text] = mapping

    def _build_ask_query(self, subject_uri: str, predicate: str, object_uri: str) -> str:
        """
        Construct SPARQL ASK query for triplet verification.

        Query checks if there exists any relation between subject and object
        that matches the predicate semantically.
        """
        # Normalize predicate (map common verbs to KB relations)
        kb_predicate = self._map_predicate_to_kb(predicate)

        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        ASK {{
            <{subject_uri}> ?p <{object_uri}> .
            FILTER(
                CONTAINS(STR(?p), "{kb_predicate}") ||
                CONTAINS(LCASE(STR(?p)), "{predicate}")
            )
        }}
        """

        return query

    def _map_predicate_to_kb(self, predicate: str) -> str:
        """
        Map natural language predicate to KB relation.

        This is a simple mapping; production would use a learned mapping
        or ontology alignment.
        """
        predicate_map = {
            "be": "IsA",
            "have": "HasA",
            "cause": "Causes",
            "lead": "Causes",
            "result": "ResultsIn",
            "create": "Creates",
            "produce": "Produces",
            "relate": "RelatedTo",
            "similar": "SimilarTo",
            "part": "PartOf",
            "use": "UsedFor",
        }

        # Check for known mapping
        for key, value in predicate_map.items():
            if key in predicate.lower():
                return value

        # Default: return normalized predicate
        return predicate.title().replace("_", "")

    def _check_contradiction(self, subject_uri: str, predicate: str, object_uri: str) -> bool:
        """
        Check if KB contains a contradiction to the triplet.

        This is a simplified implementation; production would check for
        explicit negations and ontological inconsistencies.
        """
        # Check for explicit negation relation
        neg_query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        ASK {{
            {{ <{subject_uri}> ?p <{object_uri}> .
               FILTER(CONTAINS(STR(?p), "Not") || CONTAINS(STR(?p), "Opposite")) }}
            UNION
            {{ <{object_uri}> ?p <{subject_uri}> .
               FILTER(CONTAINS(STR(?p), "Not")) }}
        }}
        """

        result = self._execute_sparql_query(neg_query)
        return result.success and result.result.get("boolean", False)

    def _fuzzy_verify(self, subject_uri: str, predicate: str, object_uri: str) -> float:
        """
        Perform fuzzy verification by checking for related paths.

        Returns confidence score [0-1].
        """
        # Check if there's any relation between subject and object
        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        ASK {{
            <{subject_uri}> ?p <{object_uri}> .
        }}
        """

        result = self._execute_sparql_query(query)

        if result.success and result.result.get("boolean", False):
            return 0.6  # Partial match: entities are related but predicate doesn't match exactly

        return 0.0

    def _execute_sparql_query(self, query: str) -> SPARQLQueryResult:
        """
        Execute SPARQL query with error handling and metrics tracking.

        Args:
            query: SPARQL query string

        Returns:
            SPARQLQueryResult with execution details
        """
        start_time = time.time()
        self.stats["queries_executed"] += 1

        try:
            self.sparql.setQuery(query)
            result = self.sparql.query().convert()

            latency_ms = (time.time() - start_time) * 1000
            self.stats["total_query_time_ms"] += latency_ms

            return SPARQLQueryResult(
                success=True,
                result=result,
                latency_ms=latency_ms
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.debug(f"SPARQL query failed: {e}")

            return SPARQLQueryResult(
                success=False,
                result=None,
                latency_ms=latency_ms,
                error=str(e)
            )

    def _sparql_string_literal(self, text: str) -> str:
        """Escape plain text for safe inclusion as a SPARQL string literal."""
        value = str(text or "")
        value = value.replace("\\", "\\\\")
        value = value.replace('"', '\\"')
        value = value.replace("\n", "\\n")
        value = value.replace("\r", "\\r")
        value = value.replace("\t", "\\t")
        return f'"{value}"'

    def get_stats(self) -> Dict[str, Any]:
        """Return verification statistics."""
        avg_query_time = (
            self.stats["total_query_time_ms"] / self.stats["queries_executed"]
            if self.stats["queries_executed"] > 0 else 0
        )

        cache_hit_rate = (
            self.stats["cache_hits"] / (self.stats["cache_hits"] + self.stats["cache_misses"])
            if (self.stats["cache_hits"] + self.stats["cache_misses"]) > 0 else 0
        )

        return {
            **self.stats,
            "avg_query_time_ms": avg_query_time,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.entity_cache)
        }

    def reset_stats(self):
        """Reset statistics counters."""
        for key in self.stats:
            self.stats[key] = 0 if isinstance(self.stats[key], (int, float)) else self.stats[key]


def test_knowledge_base_fvl():
    """Test KnowledgeBaseFVL with example usage."""
    print("=" * 60)
    print("Knowledge Base FVL Test")
    print("=" * 60)

    # Initialize (requires running triplestore)
    fvl = KnowledgeBaseFVL(
        sparql_endpoint="http://localhost:3030/conceptnet/query",
        entity_threshold=0.7,
        enable_fuzzy_match=True
    )

    # Test parsing
    response = """
    Water causes erosion in soil.
    Erosion leads to land degradation.
    Climate change results in extreme weather.
    """

    print(f"\nInput text:\n{response}")
    print("\nParsing triplets...")

    triplets = fvl.parse(response)
    print(f"\nExtracted {len(triplets)} triplets:")
    for i, t in enumerate(triplets, 1):
        print(f"  {i}. ({t.subject}) --[{t.predicate}]--> ({t.obj})")

    # Test verification (requires running triplestore with data)
    print("\nVerifying triplets...")
    try:
        results = fvl.verify(triplets)
        print(f"\nVerification results:")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r.triplet.subject} -> {r.status.value} (confidence: {r.confidence_score:.2f})")
    except Exception as e:
        print(f"  Verification failed (triplestore not running?): {e}")

    # Print stats
    print("\nStatistics:")
    stats = fvl.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    test_knowledge_base_fvl()
