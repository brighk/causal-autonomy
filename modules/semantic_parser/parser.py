"""
Module B: Semantic Parser (Middleware)
Framework: LangChain + spaCy
Task: Text-to-SPARQL mapping via entity extraction and linking

Protocol:
1. Extract entities using spaCy NER
2. Link entities to KB URIs via Fuseki label lookup (exact, then fuzzy)
3. Construct SPARQL query using templates
"""
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional, Tuple
from SPARQLWrapper import SPARQLWrapper, JSON
from loguru import logger
import re

from api.models import Triplet, CausalAssertion

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    SPACY_AVAILABLE = False

try:
    from fuzzywuzzy import fuzz
except ImportError:
    fuzz = None


@dataclass
class EntityMapping:
    """Entity text mapped to KB URI with confidence."""
    text: str
    uri: str
    confidence: float
    method: str  # exact, fuzzy


class EntityLinker:
    """
    Links entity mentions to KB URIs by querying Fuseki's own rdfs:label /
    skos:prefLabel triples directly - no separate index to keep in sync.
    """

    def __init__(
        self,
        fuseki_endpoint: str = "http://localhost:3030/dataset/query",
        entity_threshold: float = 0.7,
        enable_fuzzy_match: bool = True,
        fuzzy_match_limit: int = 10,
        cache_size: int = 1000,
    ):
        self.sparql = SPARQLWrapper(fuseki_endpoint)
        self.sparql.setReturnFormat(JSON)
        self.entity_threshold = entity_threshold
        self.enable_fuzzy_match = enable_fuzzy_match
        self.fuzzy_match_limit = fuzzy_match_limit
        self.cache_size = cache_size
        self.entity_cache: Dict[str, EntityMapping] = {}

        logger.info(f"Entity Linker initialized against {fuseki_endpoint}")

    def link_entity(self, entity_text: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Find the best-matching KB URI for a given text.

        `top_k` is accepted for interface compatibility with callers, but
        only the single best match is ever returned - a Fuseki label lookup
        isn't a ranked index the way a vector search is.

        Returns:
            List of at most one dict with 'uri', 'label', 'score', 'source'.
        """
        mapping = self._link_entity(entity_text)
        if mapping is None:
            return []

        return [{
            'uri': mapping.uri,
            'label': mapping.text,
            'score': mapping.confidence,
            'source': mapping.method,
        }]

    def _link_entity(self, entity_text: str) -> Optional[EntityMapping]:
        """
        Strategy: check cache, then exact label match, then fuzzy match.
        """
        entity_text = entity_text.strip().lower()
        if not entity_text:
            return None

        if entity_text in self.entity_cache:
            return self.entity_cache[entity_text]

        exact_uri = self._exact_entity_match(entity_text)
        if exact_uri:
            mapping = EntityMapping(text=entity_text, uri=exact_uri, confidence=1.0, method="exact")
            self._cache_entity(entity_text, mapping)
            return mapping

        if self.enable_fuzzy_match:
            fuzzy_result = self._fuzzy_entity_search(entity_text)
            if fuzzy_result:
                uri, confidence = fuzzy_result
                mapping = EntityMapping(text=entity_text, uri=uri, confidence=confidence, method="fuzzy")
                self._cache_entity(entity_text, mapping)
                return mapping

        return None

    def _exact_entity_match(self, entity_text: str) -> Optional[str]:
        """Find exact label match in the KB."""
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

        bindings = self._execute_sparql_query(query).get("results", {}).get("bindings", [])
        if bindings:
            return bindings[0]["uri"]["value"]
        return None

    def _fuzzy_entity_search(self, entity_text: str) -> Optional[Tuple[str, float]]:
        """Find similar entities using fuzzy string matching."""
        first_word = entity_text.split()[0] if entity_text else entity_text
        if not first_word:
            return None
        first_word_literal = self._sparql_string_literal(first_word)

        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?uri ?label WHERE {{
            ?uri rdfs:label ?label .
            FILTER(LANG(?label) = "en" || LANG(?label) = "")
            FILTER(CONTAINS(LCASE(STR(?label)), {first_word_literal}))
        }} LIMIT {self.fuzzy_match_limit}
        """

        bindings = self._execute_sparql_query(query).get("results", {}).get("bindings", [])
        if not bindings:
            return None

        best_uri = None
        best_score = 0.0
        for binding in bindings:
            label = binding["label"]["value"].lower()
            score = self._similarity(entity_text, label)
            if score > best_score and score >= self.entity_threshold:
                best_score = score
                best_uri = binding["uri"]["value"]

        if best_uri:
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
        """Add entity mapping to cache with simple eviction."""
        if len(self.entity_cache) >= self.cache_size:
            self.entity_cache.pop(next(iter(self.entity_cache)))
        self.entity_cache[entity_text] = mapping

    def _execute_sparql_query(self, query: str) -> Dict[str, Any]:
        """Execute a SPARQL query against Fuseki; returns {} on failure."""
        self.sparql.setQuery(query)
        try:
            return self.sparql.query().convert()
        except Exception as e:
            logger.warning(f"SPARQL query failed: {e}")
            return {}

    def _sparql_string_literal(self, text: str) -> str:
        """Escape plain text for safe inclusion as a SPARQL string literal."""
        value = str(text or "")
        value = value.replace("\\", "\\\\")
        value = value.replace('"', '\\"')
        value = value.replace("\n", "\\n")
        value = value.replace("\r", "\\r")
        value = value.replace("\t", "\\t")
        return f'"{value}"'


class SemanticParser:
    """
    Extracts triplets from natural language and converts to SPARQL.

    Pipeline:
    1. NER with spaCy
    2. Entity linking against Fuseki (label lookup + fuzzy match)
    3. Relation extraction
    4. SPARQL generation
    """

    def __init__(
        self,
        fuseki_endpoint: str = "http://localhost:3030/dataset/query",
        spacy_model: str = "en_core_web_sm"
    ):
        # spaCy is required for triplet extraction - no regex fallback, since a
        # weaker extractor would silently degrade what TruthAnchor verifies.
        if not SPACY_AVAILABLE:
            raise RuntimeError(
                "spaCy is not installed. Run: uv sync"
            )
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model '{spacy_model}'")
        except OSError:
            raise RuntimeError(
                f"spaCy model '{spacy_model}' is not installed. Run: "
                f"uv run python -m spacy download {spacy_model}"
            )

        # Initialize entity linker
        self.entity_linker = EntityLinker(fuseki_endpoint)

        # Predicate templates (common relations)
        self.predicate_templates = {
            'is': 'rdf:type',
            'has': 'schema:hasProperty',
            'causes': 'causality:causes',
            'located_in': 'schema:location',
            'part_of': 'schema:isPartOf',
            'related_to': 'schema:relatedTo',
            'created_by': 'schema:creator',
            'used_for': 'schema:purpose'
        }

        logger.info("Semantic Parser initialized")

    async def parse(
        self,
        text: str,
        causal_assertions: Optional[List[CausalAssertion]] = None
    ) -> 'ParsedResult':
        """
        Parse text into RDF triplets.

        Args:
            text: Natural language text
            causal_assertions: Pre-identified causal assertions from LLM

        Returns:
            ParsedResult with triplets and SPARQL query
        """
        triplets = []

        # If we have explicit causal assertions, parse those
        if causal_assertions:
            for assertion in causal_assertions:
                assertion_triplets = await self._parse_assertion(
                    assertion.assertion_text
                )
                triplets.extend(assertion_triplets)
                # Update the assertion's triplets
                assertion.triplets = assertion_triplets
        else:
            # Parse the entire text
            triplets = await self._parse_text(text)

        # Generate SPARQL query
        sparql_query = self._generate_sparql(triplets)

        return ParsedResult(
            triplets=triplets,
            sparql_query=sparql_query,
            source_text=text
        )

    async def _parse_text(self, text: str) -> List[Triplet]:
        """Extract triplets from free-form text using spaCy's dependency parse"""
        doc = self.nlp(text)

        triplets = []

        # Simple subject-predicate-object extraction based on dependency parsing
        for sent in doc.sents:
            for token in sent:
                # Look for subject-verb-object patterns
                if token.dep_ in ('nsubj', 'nsubjpass'):
                    subject = token.text
                    predicate = token.head.text
                    object_ = None

                    # Find object
                    for child in token.head.children:
                        if child.dep_ in ('dobj', 'attr', 'pobj'):
                            object_ = child.text
                            break

                    if object_:
                        # Link entities to URIs
                        subject_uri = await self._get_entity_uri(subject)
                        predicate_uri = self._get_predicate_uri(predicate)
                        object_uri = await self._get_entity_uri(object_)

                        triplets.append(Triplet(
                            subject=subject_uri,
                            predicate=predicate_uri,
                            object=object_uri
                        ))

        return triplets

    async def _parse_assertion(self, assertion: str) -> List[Triplet]:
        """Parse a single causal assertion into triplets"""
        # Use the same parsing logic as _parse_text
        return await self._parse_text(assertion)

    async def _get_entity_uri(self, entity_text: str) -> str:
        """
        Get the URI for an entity using entity linking.
        Falls back to a local URI if no match found.
        """
        linked = self.entity_linker.link_entity(entity_text, top_k=1)

        if linked and linked[0]['score'] > 0.7:
            return linked[0]['uri']
        else:
            # Create a local URI
            normalized = re.sub(r'[^a-zA-Z0-9]', '_', entity_text.lower())
            return f"local:{normalized}"

    def _get_predicate_uri(self, predicate_text: str) -> str:
        """
        Map predicate text to a standard URI.
        Uses template matching and falls back to generic relation.
        """
        predicate_lower = predicate_text.lower()

        # Check templates
        for template_key, uri in self.predicate_templates.items():
            if template_key in predicate_lower:
                return uri

        # Default relation
        normalized = re.sub(r'[^a-zA-Z0-9]', '_', predicate_lower)
        return f"relation:{normalized}"

    def _generate_sparql(self, triplets: List[Triplet]) -> str:
        """
        Generate SPARQL SELECT query from triplets.

        Example:
        SELECT ?o WHERE { :subject :predicate ?o }
        """
        if not triplets:
            return ""

        # Build WHERE clause
        where_patterns = []
        for t in triplets:
            # Use variable if it's a query, else use literal
            obj_var = "?o" if t.object_.startswith("local:") else f"<{t.object_}>"
            where_patterns.append(
                f"<{t.subject}> <{t.predicate}> {obj_var} ."
            )

        where_clause = "\n    ".join(where_patterns)

        sparql = f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX schema: <https://schema.org/>
PREFIX causality: <http://causality.org/>
PREFIX local: <http://local.caf/>
PREFIX relation: <http://local.caf/relation/>

SELECT ?o
WHERE {{
    {where_clause}
}}
        """.strip()

        return sparql

    def is_healthy(self) -> bool:
        """Check if parser is operational"""
        try:
            return self.entity_linker is not None
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


class ParsedResult:
    """Result from semantic parsing"""

    def __init__(
        self,
        triplets: List[Triplet],
        sparql_query: str,
        source_text: str
    ):
        self.triplets = triplets
        self.sparql_query = sparql_query
        self.source_text = source_text
