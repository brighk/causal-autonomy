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

from typing import List, Any, Optional
from experiments.knowledge_base_fvl import KnowledgeBaseFVL
from experiments.intervention_calculus import (
    CausalGraph,
    parse_causal_context,
    parse_counterfactual_query,
    counterfactual_reasoning
)
from experiments.caf_algorithm import (
    RDFTriplet,
    VerificationResult,
    VerificationStatus
)


class KnowledgeBaseFVLWithIntervention(KnowledgeBaseFVL):
    """
    Enhanced FVL with intervention calculus for counterfactual reasoning.

    Extends KnowledgeBaseFVL with:
    - Detection of counterfactual queries
    - do-calculus graph surgery
    - Proper intervention-based verification

    Usage:
        fvl = KnowledgeBaseFVLWithIntervention(
            sparql_endpoint="http://localhost:3030/counterbench/query"
        )

        # Builds causal graph from context
        fvl.set_causal_context("Ziklo causes Blaf, Blaf causes Trune...")

        # Verifies with intervention calculus if counterfactual
        results = fvl.verify(triplets)
    """

    def __init__(self, *args, **kwargs):
        """Initialize with intervention calculus support."""
        super().__init__(*args, **kwargs)
        self.causal_graph: Optional[CausalGraph] = None
        self.causal_context: Optional[str] = None
        self.current_query: Optional[str] = None
        self.last_response: Optional[str] = None

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
        knowledge_base: Any = None
    ) -> List[VerificationResult]:
        """
        Verify triplets using intervention calculus if counterfactual.

        Decision logic:
        1. If no causal context → use basic SPARQL (fallback)
        2. If not counterfactual → use basic SPARQL
        3. If counterfactual → use intervention calculus

        Args:
            triplets: RDF triplets to verify
            knowledge_base: Optional KB (unused)

        Returns:
            Verification results
        """
        # Fallback to basic SPARQL if no causal graph
        if not self.causal_graph or not self.causal_context:
            return super().verify(triplets, knowledge_base)

        # Detect counterfactual query
        if not self._is_counterfactual_query():
            # Not counterfactual → use basic SPARQL
            return super().verify(triplets, knowledge_base)

        # Counterfactual query → use intervention calculus
        return self._verify_with_intervention(triplets)

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

        # Use intervention calculus to compute expected answer
        expected_answer = counterfactual_reasoning(self.current_query, self.causal_context)
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

        answer = counterfactual_reasoning(self.current_query, self.causal_context)

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
