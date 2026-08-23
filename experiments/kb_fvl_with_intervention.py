"""
Knowledge Base FVL with Intervention Calculus
==============================================

Enhanced Formal Verification Layer that combines:
1. Basic SPARQL verification (for factual queries)
2. Intervention calculus (for counterfactual queries)

This enables CAF to properly handle:
- Factual: "Does X cause Y?" → SPARQL ASK query
- Counterfactual: "Would Y occur if NOT X?" → do-calculus
"""

from typing import List, Any, Optional, Set, Tuple, Dict
from experiments.knowledge_base_fvl import KnowledgeBaseFVL
from experiments.intervention_calculus import (
    CausalGraph,
    parse_causal_context,
    parse_counterfactual_query,
    counterfactual_reasoning_with_graph,
    normalize_node_id,
)
from experiments.caf_algorithm import (
    RDFTriplet,
    VerificationResult,
    VerificationStatus
)

# Predicate substrings treated as "causal" when traversing the live KB to
# build a do-calculus graph. Kept in sync by hand with
# modules/causal_validator/validator.py's _is_causal_predicate keyword list -
# there's no shared constants module between api/ and experiments/ to pull
# this from instead.
CAUSAL_PREDICATE_KEYWORDS = (
    "causes", "causedby", "resultin", "leadto", "produce", "trigger", "influence",
)


class KnowledgeBaseFVLWithIntervention(KnowledgeBaseFVL):
    """
    Enhanced FVL with intervention calculus for counterfactual reasoning.

    Extends KnowledgeBaseFVL with:
    - Detection of counterfactual queries
    - do-calculus graph surgery
    - Proper intervention-based verification

    Two ways to supply the causal graph:

    1. Manual (e.g. CounterBench's fictional-text harness), unchanged:
        fvl = KnowledgeBaseFVLWithIntervention(
            sparql_endpoint="http://localhost:3030/counterbench/query"
        )
        fvl.set_causal_context("Ziklo causes Blaf, Blaf causes Trune...")
        fvl.set_current_query(query)
        results = fvl.verify(triplets)

       Calling set_causal_context() puts the instance in "manual mode":
       it owns current_query/causal_graph for every subsequent verify()
       call, and the KB is never auto-traversed.

    2. Automatic (default; for CAFLoop against a real KB), no setup:
        fvl = KnowledgeBaseFVLWithIntervention(sparql_endpoint="http://localhost:3030/dataset/query")
        caf_loop = CAFLoop(verification_layer=fvl, ...)
        caf_loop.execute("Would the road be slippery if it hadn't rained?")

       CAFLoop passes the prompt through to verify(query=...); if it looks
       counterfactual, the causal graph is built by walking causal-predicate
       edges outward from the entities mentioned, live from the configured
       SPARQL endpoint. Falls back to plain SPARQL verification (like
       KnowledgeBaseFVL) for factual queries, or if no causal edges are
       found within `causal_graph_max_hops` hops.
    """

    def __init__(self, *args, causal_graph_max_hops: int = 2, **kwargs):
        """Initialize with intervention calculus support."""
        super().__init__(*args, **kwargs)
        self.causal_graph: Optional[CausalGraph] = None
        self.causal_context: Optional[str] = None
        self.current_query: Optional[str] = None
        self.last_response: Optional[str] = None
        self.causal_graph_max_hops = causal_graph_max_hops
        self._kb_graph_cache: Dict[Tuple[str, ...], CausalGraph] = {}

    def parse(self, response: str) -> List[RDFTriplet]:
        """Parse response and retain raw text for answer-level verification."""
        self.last_response = response
        return super().parse(response)

    def set_causal_context(self, context: str):
        """
        Build causal graph from context text.

        Args:
            context: Text describing causal relationships
                    e.g., "Ziklo causes Blaf, Blaf causes Trune"
        """
        self.causal_context = context
        self.causal_graph = parse_causal_context(context)

    def set_current_query(self, query: str):
        """
        Set the current query being evaluated.

        This allows the FVL to detect counterfactual queries
        and apply intervention calculus.

        Args:
            query: The question being asked
        """
        self.current_query = query

    def _is_counterfactual_query(self) -> bool:
        """
        Detect if current query is counterfactual.

        Counterfactual patterns:
        - "Would X occur if NOT Y instead of Y?"
        - "What if we prevent X?"
        - "If not X, would Y happen?"

        Returns:
            True if query is counterfactual
        """
        if not self.current_query:
            return False

        parsed = parse_counterfactual_query(self.current_query)
        return parsed is not None

    def verify(
        self,
        triplets: List[RDFTriplet],
        knowledge_base: Any = None,
        query: Optional[str] = None
    ) -> List[VerificationResult]:
        """
        Verify triplets using intervention calculus if counterfactual.

        Decision logic:
        1. If no causal graph with edges → use basic SPARQL (fallback)
        2. If not counterfactual → use basic SPARQL
        3. If counterfactual → use intervention calculus

        In "manual mode" (set_causal_context() was called at least once on
        this instance, e.g. the CounterBench harness), current_query/
        causal_graph are only ever touched by explicit setter calls - the
        `query` passed here is ignored. Otherwise ("auto mode", the
        default for a fresh instance used directly with CAFLoop), `query`
        is used to detect a counterfactual question and, if so, the causal
        graph is (re)built live from the SPARQL KB on every call, since
        consecutive verify() calls may be answering different questions.

        Args:
            triplets: RDF triplets to verify
            knowledge_base: Optional KB (unused)
            query: The prompt/question being answered (CAFLoop passes this
                through automatically)

        Returns:
            Verification results
        """
        if self.causal_context is None:
            # Auto mode.
            self.current_query = query
            self.causal_graph = None
            if query:
                self._ensure_causal_graph(triplets, query)

        # Fallback to basic SPARQL if no causal graph with actual edges
        if not self.causal_graph or not self.causal_graph.edges:
            return super().verify(triplets, knowledge_base)

        # Detect counterfactual query
        if not self._is_counterfactual_query():
            # Not counterfactual → use basic SPARQL
            return super().verify(triplets, knowledge_base)

        # Counterfactual query → use intervention calculus
        return self._verify_with_intervention(triplets)

    def _ensure_causal_graph(
        self,
        triplets: List[RDFTriplet],
        query: str
    ) -> None:
        """
        Auto-build the causal graph from the live KB for a counterfactual
        query (auto mode only - see verify()). No-op if `query` doesn't
        match a recognized counterfactual pattern.
        """
        parsed_query = parse_counterfactual_query(query)
        if not parsed_query:
            return

        seed_texts = [parsed_query['target'], parsed_query['intervention_node']]
        seed_texts.extend(t.subject for t in triplets)
        seed_texts.extend(t.obj for t in triplets)

        graph = self._build_causal_graph_from_kb(seed_texts)
        if graph.edges:
            self.causal_graph = graph

    def _build_causal_graph_from_kb(
        self,
        seed_texts: List[str],
        max_hops: Optional[int] = None
    ) -> CausalGraph:
        """
        Build a do-calculus CausalGraph by walking causal-predicate edges
        outward from the given entity mentions in the live SPARQL KB.

        Links each seed text to a KB URI via the inherited entity linker,
        then does a bounded breadth-first walk over edges whose predicate
        matches CAUSAL_PREDICATE_KEYWORDS, resolving each discovered URI
        back to a label to use as the graph node id (normalized the same
        way as parse_counterfactual_query's entity names, so the two sides
        compare equal).

        Results are cached per unique, sorted seed-text set so repeated
        verify() calls within one CAFLoop refinement loop (same prompt,
        same seeds) don't re-run the SPARQL traversal every iteration.
        """
        if max_hops is None:
            max_hops = self.causal_graph_max_hops

        cache_key = tuple(sorted(normalize_node_id(t) for t in seed_texts if t))
        if cache_key in self._kb_graph_cache:
            return self._kb_graph_cache[cache_key]

        graph = CausalGraph()
        visited_uris: Set[str] = set()
        frontier: List[str] = []

        for seed_text in seed_texts:
            mapping = self._link_entity(seed_text)
            if mapping and mapping.uri not in visited_uris:
                visited_uris.add(mapping.uri)
                frontier.append(mapping.uri)

        for _ in range(max_hops):
            if not frontier:
                break
            next_frontier: List[str] = []

            for uri in frontier:
                for neighbor_uri, cause_uri, effect_uri in self._causal_edges(uri):
                    cause_label = normalize_node_id(self._resolve_label(cause_uri) or cause_uri)
                    effect_label = normalize_node_id(self._resolve_label(effect_uri) or effect_uri)
                    graph.add_edge(cause_label, effect_label)

                    if neighbor_uri not in visited_uris:
                        visited_uris.add(neighbor_uri)
                        next_frontier.append(neighbor_uri)

            frontier = next_frontier

        self._kb_graph_cache[cache_key] = graph
        return graph

    def _causal_edges(self, uri: str) -> List[Tuple[str, str, str]]:
        """
        One hop of causal-predicate-filtered SPARQL traversal from `uri`.

        Returns a list of (neighbor_uri, cause_uri, effect_uri) tuples for
        every causal edge touching `uri`, in either direction.
        """
        keyword_filter = " || ".join(
            f'CONTAINS(LCASE(STR(?p)), "{kw}")' for kw in CAUSAL_PREDICATE_KEYWORDS
        )

        query = f"""
        SELECT ?p ?o ?s WHERE {{
            {{ <{uri}> ?p ?o . FILTER({keyword_filter}) }}
            UNION
            {{ ?s ?p <{uri}> . FILTER({keyword_filter}) }}
        }}
        """
        result = self._execute_sparql_query(query)
        if not result.success:
            return []

        edges: List[Tuple[str, str, str]] = []
        for binding in result.result.get("results", {}).get("bindings", []):
            obj_uri = binding.get("o", {}).get("value")
            subj_uri = binding.get("s", {}).get("value")
            if obj_uri:
                edges.append((obj_uri, uri, obj_uri))
            elif subj_uri:
                edges.append((subj_uri, subj_uri, uri))

        return edges

    def _verify_with_intervention(
        self,
        triplets: List[RDFTriplet]
    ) -> List[VerificationResult]:
        """
        Verify using intervention calculus.

        Process:
        1. Parse counterfactual query
        2. Apply do-calculus (graph surgery)
        3. Check if claim holds under intervention
        4. Return verification result

        Args:
            triplets: Claims to verify

        Returns:
            Verification results using do-calculus
        """
        results = []

        # Parse the counterfactual query
        parsed_query = parse_counterfactual_query(self.current_query)

        if not parsed_query:
            # Failed to parse → fallback to SPARQL
            return super().verify(triplets, knowledge_base=None)

        # Use intervention calculus to compute expected answer, against the
        # already-built graph object (avoids re-parsing causal_context text
        # through parse_causal_context's single-word regex, which would
        # mangle multi-word KB-derived labels).
        expected_answer = counterfactual_reasoning_with_graph(self.current_query, self.causal_graph)
        predicted_answer = self._extract_binary_answer(self.last_response or "")

        # Use one canonical claim for scoring to avoid noisy triplet over-penalization.
        claim_triplet = triplets[0] if triplets else RDFTriplet(
            subject=parsed_query['intervention_node'].lower(),
            predicate="counterfactual_effect",
            obj=parsed_query['target'].lower(),
            confidence=1.0,
            source_span=self.current_query
        )

        if expected_answer is None or predicted_answer == "unknown":
            status = VerificationStatus.PARTIAL
            kb_support = False
            score = 0.5
            supporting_facts = []
            contradicting_facts = []
        else:
            expected_label = "yes" if expected_answer else "no"
            if predicted_answer == expected_label:
                status = VerificationStatus.VERIFIED
                kb_support = True
                score = 1.0
                supporting_facts = [
                    f"Intervention calculus expects '{expected_label}'",
                    f"Based on: do({parsed_query['intervention_node']}={parsed_query['intervention_value']})"
                ]
                contradicting_facts = []
            else:
                status = VerificationStatus.CONTRADICTION
                kb_support = False
                score = 0.0
                supporting_facts = []
                contradicting_facts = [
                    f"Intervention calculus expects '{expected_label}' but response implies '{predicted_answer}'",
                    f"Based on: do({parsed_query['intervention_node']}={parsed_query['intervention_value']})"
                ]

        result = VerificationResult(
            triplet=claim_triplet,
            status=status,
            kb_support=kb_support,
            contradiction_found=(status == VerificationStatus.CONTRADICTION),
            supporting_facts=supporting_facts,
            contradicting_facts=contradicting_facts,
            confidence_score=score
        )
        results.append(result)

        return results

    def _extract_binary_answer(self, response: str) -> str:
        """
        Extract yes/no/unknown from generated text.
        Mirrors evaluator logic so verification and scoring are aligned.
        """
        text = response.lower()

        if 'cannot determine' in text or 'uncertain' in text:
            return 'unknown'
        if 'would not occur' in text or 'would not happen' in text:
            return 'no'
        if 'would occur' in text or 'would happen' in text:
            return 'yes'

        yes_idx = text.find('yes')
        no_idx = text.find('no')

        if yes_idx != -1 and no_idx == -1:
            return 'yes'
        if no_idx != -1 and yes_idx == -1:
            return 'no'
        if yes_idx != -1 and no_idx != -1:
            return 'yes' if yes_idx < no_idx else 'no'
        return 'unknown'

    def get_explanation(self) -> str:
        """
        Get human-readable explanation of last verification.

        Returns:
            Explanation of intervention calculus reasoning
        """
        if not self.current_query or not self.causal_graph:
            return "No query or causal graph available"

        parsed = parse_counterfactual_query(self.current_query)
        if not parsed:
            return "Not a counterfactual query"

        answer = counterfactual_reasoning_with_graph(self.current_query, self.causal_graph)

        explanation = f"""
Counterfactual Reasoning via Intervention Calculus:

Query: {self.current_query}
Intervention: do({parsed['intervention_node']}={parsed['intervention_value']})
Target: {parsed['target']}

Causal Graph:
{self._format_graph()}

Intervention Effect:
- After do({parsed['intervention_node']}={parsed['intervention_value']}):
  {self._explain_intervention_effect(parsed)}

Answer: {'Yes' if answer else 'No'}
"""
        return explanation

    def _format_graph(self) -> str:
        """Format causal graph as text."""
        if not self.causal_graph:
            return "No graph"

        lines = []
        for edge in self.causal_graph.edges:
            lines.append(f"  {edge.cause} → {edge.effect}")
        return "\n".join(lines) if lines else "  (empty graph)"

    def _explain_intervention_effect(self, parsed_query: dict) -> str:
        """Explain the effect of the intervention."""
        intervention_node = parsed_query['intervention_node']
        intervention_value = parsed_query['intervention_value']
        target = parsed_query['target']

        if not intervention_value:
            # Preventing the intervention node
            descendants = self.causal_graph.get_descendants(intervention_node)

            if target in descendants:
                return f"{target} is a descendant of {intervention_node} → won't occur"
            elif target == intervention_node:
                return f"{target} is the intervened node → won't occur"
            else:
                return f"{target} is independent of {intervention_node} → may still occur"
        else:
            # Forcing the intervention node
            descendants = self.causal_graph.get_descendants(intervention_node)

            if target in descendants:
                return f"{target} is a descendant of {intervention_node} → will occur"
            elif target == intervention_node:
                return f"{target} is the intervened node → will occur"
            else:
                return f"{target} is independent of {intervention_node} → check other causes"


def create_intervention_fvl(
    sparql_endpoint: str = "http://localhost:3030/counterbench/query",
    **kwargs
) -> KnowledgeBaseFVLWithIntervention:
    """
    Factory function to create FVL with intervention calculus.

    Args:
        sparql_endpoint: SPARQL endpoint URL
        **kwargs: Additional arguments for KnowledgeBaseFVL

    Returns:
        Configured FVL with intervention support
    """
    return KnowledgeBaseFVLWithIntervention(
        sparql_endpoint=sparql_endpoint,
        **kwargs
    )
