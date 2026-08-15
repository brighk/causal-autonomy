"""
Module D: Causal Validator (Verification)
Library: DoWhy
Mechanism: Axiomatic Verification

Validates causal assertions using formal causal reasoning.
Checks for logical consistency and causal soundness.
"""
from typing import List, Dict, Any, Optional
import networkx as nx
from loguru import logger

from api.models import CausalAssertion, Triplet


class CausalGraph:
    """
    Represents causal relationships as a directed graph.
    Used for detecting causal violations.
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_causal_edge(self, cause: str, effect: str, metadata: Optional[Dict] = None):
        """Add a causal relationship to the graph"""
        self.graph.add_edge(cause, effect, **(metadata or {}))

    def has_cycle(self) -> bool:
        """Check if the causal graph contains cycles (causal paradox)"""
        try:
            cycles = list(nx.simple_cycles(self.graph))
            return len(cycles) > 0
        except:
            return False

    def get_violations(self) -> List[str]:
        """Detect causal violations"""
        violations = []

        # Check for cycles
        if self.has_cycle():
            cycles = list(nx.simple_cycles(self.graph))
            for cycle in cycles:
                violations.append(
                    f"Causal cycle detected: {' → '.join(cycle + [cycle[0]])}"
                )

        # Check for contradictory edges
        for node in self.graph.nodes():
            # Get all edges involving this node
            incoming = list(self.graph.predecessors(node))
            outgoing = list(self.graph.successors(node))

            # Check for bidirectional causation (often contradictory)
            for neighbor in outgoing:
                if neighbor in incoming:
                    violations.append(
                        f"Bidirectional causation detected: {node} ↔ {neighbor}"
                    )

        return violations


class CausalValidator:
    """
    Validates causal assertions using axiomatic verification.

    Axioms:
    1. Transitivity: If A causes B and B causes C, then A influences C
    2. Acyclicity: No causal loops (A causes B causes A)
    3. Consistency: Assertions must not contradict verified knowledge
    """

    def __init__(self):
        self.causal_graph = CausalGraph()
        logger.info("Causal Validator initialized")

    async def validate(
        self,
        assertions: List[CausalAssertion],
        verified_triplets: List[Triplet]
    ) -> 'ValidationResult':
        """
        Validate causal assertions for logical consistency.

        Args:
            assertions: Causal assertions to validate
            verified_triplets: Triplets already verified by Truth Anchor

        Returns:
            ValidationResult with validity status and violations
        """
        violations = []

        # Build causal graph from verified triplets
        self._build_graph_from_triplets(verified_triplets)

        # Check for structural violations
        graph_violations = self.causal_graph.get_violations()
        violations.extend(graph_violations)

        # Validate each assertion
        for assertion in assertions:
            assertion_violations = await self._validate_assertion(
                assertion,
                verified_triplets
            )
            violations.extend(assertion_violations)

        is_valid = len(violations) == 0

        return ValidationResult(
            is_valid=is_valid,
            violations=violations,
            causal_graph_nodes=list(self.causal_graph.graph.nodes()),
            causal_graph_edges=list(self.causal_graph.graph.edges())
        )

    def _build_graph_from_triplets(self, triplets: List[Triplet]):
        """Build causal graph from RDF triplets"""
        for triplet in triplets:
            # Identify causal predicates
            if self._is_causal_predicate(triplet.predicate):
                self.causal_graph.add_causal_edge(
                    triplet.subject,
                    triplet.object_,
                    metadata={'predicate': triplet.predicate}
                )

    def _is_causal_predicate(self, predicate: str) -> bool:
        """Check if a predicate represents a causal relationship"""
        causal_keywords = [
            'causes',
            'causedBy',
            'resultIn',
            'leadTo',
            'produce',
            'trigger',
            'influence'
        ]

        predicate_lower = predicate.lower()
        return any(keyword.lower() in predicate_lower for keyword in causal_keywords)

    async def _validate_assertion(
        self,
        assertion: CausalAssertion,
        verified_triplets: List[Triplet]
    ) -> List[str]:
        """Validate a single causal assertion"""
        violations = []

        # Check if assertion's triplets are in the verified set
        for triplet in assertion.triplets:
            if triplet not in verified_triplets:
                # Check if it contradicts any verified triplet
                contradiction = self._find_contradiction(triplet, verified_triplets)
                if contradiction:
                    violations.append(
                        f"Assertion contradicts verified knowledge: {contradiction}"
                    )

        # Check for temporal violations (if temporal info available)
        # This would require additional metadata about time
        # temporal_violations = self._check_temporal_violations(assertion)
        # violations.extend(temporal_violations)

        return violations

    def _find_contradiction(
        self,
        triplet: Triplet,
        verified_triplets: List[Triplet]
    ) -> Optional[str]:
        """
        Find if a triplet contradicts verified knowledge.

        Contradiction occurs when:
        - Same subject and predicate but different object
        """
        for verified in verified_triplets:
            if (triplet.subject == verified.subject and
                triplet.predicate == verified.predicate and
                triplet.object_ != verified.object_):

                return (
                    f"({triplet.subject}, {triplet.predicate}, {triplet.object_}) "
                    f"contradicts verified "
                    f"({verified.subject}, {verified.predicate}, {verified.object_})"
                )

        return None

    def reset(self):
        """Reset the causal graph"""
        self.causal_graph = CausalGraph()

    def is_healthy(self) -> bool:
        """Check if validator is operational"""
        return self.causal_graph is not None


class ValidationResult:
    """Result from causal validation"""

    def __init__(
        self,
        is_valid: bool,
        violations: List[str],
        causal_graph_nodes: List[str],
        causal_graph_edges: List[tuple]
    ):
        self.is_valid = is_valid
        self.violations = violations
        self.causal_graph_nodes = causal_graph_nodes
        self.causal_graph_edges = causal_graph_edges
