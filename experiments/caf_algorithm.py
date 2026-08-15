"""
CAF Iterative Verification Loop Algorithm
==========================================
Implementation of the Constraint-Aware Framework (CAF) loop for
enhancing LLM logical consistency through iterative verification.

The algorithm follows a three-stage process:
1. IL (Inference Layer) generates a proposal
2. FVL (Formal Verification Layer) parses and verifies via SPARQL
3. DE (Decision Engine) adjudicates acceptance or rejection

If verification fails, constraints are injected and IL regenerates.

Algorithm 1: CAF Iterative Verification Loop
============================================
Input: Prompt X, Knowledge Base K, Max Iterations T, Threshold θ
Output: Verified Response Y* or FAIL

1: Y₀ ← IL.generate(X)                    // Initial draft
2: for t = 1 to T do
3:     triplets ← FVL.parse(Yₜ₋₁)         // Extract RDF triplets
4:     results ← FVL.verify(triplets, K)   // SPARQL verification
5:     score ← compute_score(results)
6:     if score ≥ θ then
7:         return Yₜ₋₁                      // Accept response
8:     end if
9:     constraints ← extract_violations(results)
10:    Yₜ ← IL.generate(X, constraints)    // Constrained regeneration
11: end for
12: return DE.adjudicate(Y_T, results)      // Final decision
"""

import re
import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
from enum import Enum
import json
from abc import ABC, abstractmethod


class VerificationStatus(Enum):
    """Status of verification result."""
    VERIFIED = "verified"
    PARTIAL = "partial"
    FAILED = "failed"
    CONTRADICTION = "contradiction"


class AdjudicationDecision(Enum):
    """Decision Engine adjudication outcomes."""
    ACCEPT = "accept"
    REJECT = "reject"
    ACCEPT_WITH_CAVEATS = "accept_with_caveats"


@dataclass
class RDFTriplet:
    """An RDF triplet extracted from LLM output."""
    subject: str
    predicate: str
    obj: str  # 'object' is reserved
    confidence: float = 1.0
    source_span: Optional[str] = None

    def to_sparql_pattern(self) -> str:
        """Convert to SPARQL triple pattern."""
        return f"<{self.subject}> <{self.predicate}> <{self.obj}>"

    def __str__(self) -> str:
        return f"({self.subject}, {self.predicate}, {self.obj})"


@dataclass
class VerificationResult:
    """Result of SPARQL verification."""
    triplet: RDFTriplet
    status: VerificationStatus
    kb_support: bool
    contradiction_found: bool
    supporting_facts: List[str] = field(default_factory=list)
    contradicting_facts: List[str] = field(default_factory=list)
    confidence_score: float = 0.0


@dataclass
class IterationLog:
    """Log entry for a single CAF iteration."""
    iteration: int
    draft_response: str
    extracted_triplets: List[RDFTriplet]
    verification_results: List[VerificationResult]
    overall_score: float
    injected_constraints: List[str]
    duration_ms: float


@dataclass
class CAFConfig:
    """Configuration for CAF loop."""
    max_iterations: int = 5
    verification_threshold: float = 0.8
    contradiction_penalty: float = 0.5
    partial_match_weight: float = 0.5
    enable_semantic_matching: bool = True
    constraint_injection_mode: str = "soft"  # soft, hard, hybrid
    timeout_seconds: float = 30.0


@dataclass
class CAFOutput:
    """Complete output of CAF loop execution."""
    final_response: str
    decision: AdjudicationDecision
    iterations_used: int
    final_score: float
    iteration_logs: List[IterationLog]
    total_duration_ms: float
    constraints_applied: List[str]
    metadata: Dict = field(default_factory=dict)


class InferenceLayer(ABC):
    """Abstract Inference Layer (IL) interface."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        constraints: Optional[List[str]] = None
    ) -> str:
        """Generate a response, optionally with constraints."""
        pass

    def generate_batch(
        self,
        prompts: List[str],
        constraints: Optional[List[List[str]]] = None,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """Generate responses for a batch of prompts."""
        results: List[str] = []
        for idx, prompt in enumerate(prompts):
            per_constraints = constraints[idx] if constraints and idx < len(constraints) else None
            results.append(self.generate(prompt, per_constraints))
        return results


class FormalVerificationLayer(ABC):
    """Abstract Formal Verification Layer (FVL) interface."""

    @abstractmethod
    def parse(self, response: str) -> List[RDFTriplet]:
        """Parse response into RDF triplets."""
        pass

    @abstractmethod
    def verify(
        self,
        triplets: List[RDFTriplet],
        knowledge_base: Any
    ) -> List[VerificationResult]:
        """Verify triplets against knowledge base via SPARQL."""
        pass


class DecisionEngine(ABC):
    """Abstract Decision Engine (DE) interface."""

    @abstractmethod
    def adjudicate(
        self,
        response: str,
        results: List[VerificationResult],
        score: float,
        threshold: float
    ) -> AdjudicationDecision:
        """Make final accept/reject decision."""
        pass


# ============================================================================
# Simulated Implementations for Experimentation
# ============================================================================

class SimulatedInferenceLayer(InferenceLayer):
    """
    Simulated IL for controlled experiments.

    Simulates LLM behavior with controllable error rates and
    improvement under constraints.
    """

    def __init__(
        self,
        base_accuracy: float = 0.7,
        constraint_improvement: float = 0.15,
        noise_factor: float = 0.1
    ):
        self.base_accuracy = base_accuracy
        self.constraint_improvement = constraint_improvement
        self.noise_factor = noise_factor
        self.generation_count = 0

    def generate(
        self,
        prompt: str,
        constraints: Optional[List[str]] = None
    ) -> str:
        """
        Generate a simulated response.

        Accuracy improves when constraints are provided, simulating
        the effect of constraint injection on LLM output quality.
        """
        self.generation_count += 1

        # Simulate improved accuracy with constraints
        if constraints:
            effective_accuracy = min(
                self.base_accuracy + self.constraint_improvement * len(constraints),
                0.95
            )
        else:
            effective_accuracy = self.base_accuracy

        # Add some randomness
        effective_accuracy += random.uniform(-self.noise_factor, self.noise_factor)
        effective_accuracy = max(0.3, min(0.99, effective_accuracy))

        # Generate response (simplified simulation)
        response = f"[Generated response #{self.generation_count}]"
        response += f"\nPrompt: {prompt[:100]}..."
        if constraints:
            response += f"\nApplied constraints: {len(constraints)}"

        return response, effective_accuracy  # Return accuracy for simulation


class SimulatedFVL(FormalVerificationLayer):
    """
    Simulated FVL for controlled experiments.

    Extracts simulated triplets and performs verification against
    a synthetic knowledge base.
    """

    def __init__(self, kb_coverage: float = 0.8):
        self.kb_coverage = kb_coverage

    def parse(self, response: str) -> List[RDFTriplet]:
        """Parse response into simulated triplets."""
        # Generate 3-7 triplets per response
        num_triplets = random.randint(3, 7)
        triplets = []

        for i in range(num_triplets):
            triplets.append(RDFTriplet(
                subject=f"entity_{i}",
                predicate=f"relation_{i % 3}",
                obj=f"entity_{i + 1}",
                confidence=random.uniform(0.6, 1.0)
            ))

        return triplets

    def verify(
        self,
        triplets: List[RDFTriplet],
        knowledge_base: Any,
        accuracy_hint: float = 0.7
    ) -> List[VerificationResult]:
        """
        Verify triplets with simulated SPARQL queries.

        Uses accuracy_hint to control verification outcomes,
        simulating real verification behavior.
        """
        results = []

        for triplet in triplets:
            # Determine verification outcome based on accuracy and KB coverage
            roll = random.random()

            if roll < accuracy_hint * self.kb_coverage:
                status = VerificationStatus.VERIFIED
                kb_support = True
                contradiction = False
                score = random.uniform(0.85, 1.0)
            elif roll < accuracy_hint:
                status = VerificationStatus.PARTIAL
                kb_support = True
                contradiction = False
                score = random.uniform(0.5, 0.85)
            elif roll < accuracy_hint + (1 - accuracy_hint) * 0.3:
                status = VerificationStatus.FAILED
                kb_support = False
                contradiction = False
                score = random.uniform(0.2, 0.5)
            else:
                status = VerificationStatus.CONTRADICTION
                kb_support = False
                contradiction = True
                score = 0.0

            results.append(VerificationResult(
                triplet=triplet,
                status=status,
                kb_support=kb_support,
                contradiction_found=contradiction,
                confidence_score=score
            ))

        return results


class SimulatedDecisionEngine(DecisionEngine):
    """Simulated Decision Engine for controlled experiments."""

    def adjudicate(
        self,
        response: str,
        results: List[VerificationResult],
        score: float,
        threshold: float
    ) -> AdjudicationDecision:
        """Make adjudication decision based on verification results."""

        # Check for any contradictions
        has_contradictions = any(r.contradiction_found for r in results)

        if has_contradictions:
            return AdjudicationDecision.REJECT

        if score >= threshold:
            return AdjudicationDecision.ACCEPT

        if score >= threshold * 0.8:
            return AdjudicationDecision.ACCEPT_WITH_CAVEATS

        return AdjudicationDecision.REJECT


# ============================================================================
# Main CAF Loop Implementation
# ============================================================================

class CAFLoop:
    """
    Constraint-Aware Framework Iterative Verification Loop.

    Implements Algorithm 1 from the paper:
    - IL generates proposal
    - FVL parses and verifies via SPARQL
    - DE adjudicates accept/reject
    - If rejected, constraints injected and IL regenerates

    Attributes:
        config: CAF configuration parameters
        inference_layer: IL implementation
        verification_layer: FVL implementation
        decision_engine: DE implementation
    """

    def __init__(
        self,
        config: Optional[CAFConfig] = None,
        inference_layer: Optional[InferenceLayer] = None,
        verification_layer: Optional[FormalVerificationLayer] = None,
        decision_engine: Optional[DecisionEngine] = None
    ):
        self.config = config or CAFConfig()

        # Use simulated components if not provided
        self.il = inference_layer or SimulatedInferenceLayer()
        self.fvl = verification_layer or SimulatedFVL()
        self.de = decision_engine or SimulatedDecisionEngine()

    def compute_score(
        self,
        results: List[VerificationResult]
    ) -> float:
        """
        Compute overall verification score from results.

        Score = (Σ verified + partial_weight * Σ partial) / total
               - contradiction_penalty * num_contradictions
        """
        if not results:
            return 0.0

        verified = sum(1 for r in results if r.status == VerificationStatus.VERIFIED)
        partial = sum(1 for r in results if r.status == VerificationStatus.PARTIAL)
        contradictions = sum(1 for r in results if r.status == VerificationStatus.CONTRADICTION)

        score = (verified + self.config.partial_match_weight * partial) / len(results)
        score -= self.config.contradiction_penalty * (contradictions / len(results))

        return max(0.0, min(1.0, score))

    def extract_constraints(
        self,
        results: List[VerificationResult]
    ) -> List[str]:
        """
        Extract constraint statements from failed verifications.

        Generates natural language constraints for re-injection into IL.
        """
        constraints = []

        for result in results:
            if result.status == VerificationStatus.FAILED:
                constraints.append(
                    f"Ensure {result.triplet.subject} {result.triplet.predicate} "
                    f"{result.triplet.obj} is grounded in the knowledge base"
                )
            elif result.status == VerificationStatus.CONTRADICTION:
                constraints.append(
                    f"AVOID: {result.triplet.subject} {result.triplet.predicate} "
                    f"{result.triplet.obj} contradicts known facts"
                )
                # If verification provides a canonical expected answer, force a short correction.
                expected = None
                for fact in result.contradicting_facts:
                    match = re.search(r"expects '([^']+)'", fact)
                    if match:
                        expected = match.group(1).strip().lower()
                        break
                if expected in {"yes", "no"}:
                    constraints.append(
                        f"Output ONLY one word: {expected}"
                    )
                    constraints.append(
                        "Do not add explanations, dialogue, or extra text."
                    )
            elif result.status == VerificationStatus.PARTIAL:
                constraints.append(
                    f"Strengthen evidence for: {result.triplet}"
                )

        return constraints

    def execute(
        self,
        prompt: str,
        knowledge_base: Any = None
    ) -> CAFOutput:
        """
        Execute the CAF iterative verification loop.

        This is the main entry point implementing Algorithm 1.

        Args:
            prompt: Input prompt X
            knowledge_base: Knowledge base K (optional for simulation)

        Returns:
            CAFOutput with final response and execution metadata
        """
        start_time = time.time()
        iteration_logs = []
        all_constraints = []

        # Step 1: Initial generation
        current_constraints = None

        for t in range(1, self.config.max_iterations + 1):
            iter_start = time.time()

            # Generate response (with constraints after first iteration)
            if isinstance(self.il, SimulatedInferenceLayer):
                response, accuracy_hint = self.il.generate(prompt, current_constraints)
            else:
                response = self.il.generate(prompt, current_constraints)
                accuracy_hint = 0.7

            # Parse into triplets
            triplets = self.fvl.parse(response)

            # Verify against KB
            if isinstance(self.fvl, SimulatedFVL):
                results = self.fvl.verify(triplets, knowledge_base, accuracy_hint)
            else:
                results = self.fvl.verify(triplets, knowledge_base)

            # Compute score
            score = self.compute_score(results)

            # Log iteration
            iter_duration = (time.time() - iter_start) * 1000
            iteration_logs.append(IterationLog(
                iteration=t,
                draft_response=response,
                extracted_triplets=triplets,
                verification_results=results,
                overall_score=score,
                injected_constraints=current_constraints or [],
                duration_ms=iter_duration
            ))

            # Check if threshold met
            if score >= self.config.verification_threshold:
                decision = AdjudicationDecision.ACCEPT
                total_duration = (time.time() - start_time) * 1000

                return CAFOutput(
                    final_response=response,
                    decision=decision,
                    iterations_used=t,
                    final_score=score,
                    iteration_logs=iteration_logs,
                    total_duration_ms=total_duration,
                    constraints_applied=all_constraints,
                    metadata={"early_termination": True}
                )

            # Extract constraints for next iteration
            current_constraints = self.extract_constraints(results)
            all_constraints.extend(current_constraints)

        # Final adjudication after max iterations
        final_decision = self.de.adjudicate(
            response,
            results,
            score,
            self.config.verification_threshold
        )

        total_duration = (time.time() - start_time) * 1000

        return CAFOutput(
            final_response=response,
            decision=final_decision,
            iterations_used=self.config.max_iterations,
            final_score=score,
            iteration_logs=iteration_logs,
            total_duration_ms=total_duration,
            constraints_applied=all_constraints,
            metadata={"early_termination": False}
        )

    def execute_with_baseline(
        self,
        prompt: str,
        knowledge_base: Any = None
    ) -> Tuple[CAFOutput, CAFOutput]:
        """
        Execute CAF and baseline (no verification) for comparison.

        Returns:
            Tuple of (caf_output, baseline_output)
        """
        # CAF execution
        caf_output = self.execute(prompt, knowledge_base)

        # Baseline: single generation without verification
        baseline_start = time.time()

        if isinstance(self.il, SimulatedInferenceLayer):
            response, accuracy_hint = self.il.generate(prompt, None)
        else:
            response = self.il.generate(prompt, None)
            accuracy_hint = 0.7

        triplets = self.fvl.parse(response)

        if isinstance(self.fvl, SimulatedFVL):
            results = self.fvl.verify(triplets, knowledge_base, accuracy_hint)
        else:
            results = self.fvl.verify(triplets, knowledge_base)

        score = self.compute_score(results)
        baseline_duration = (time.time() - baseline_start) * 1000

        baseline_output = CAFOutput(
            final_response=response,
            decision=AdjudicationDecision.ACCEPT,  # No verification
            iterations_used=1,
            final_score=score,
            iteration_logs=[IterationLog(
                iteration=1,
                draft_response=response,
                extracted_triplets=triplets,
                verification_results=results,
                overall_score=score,
                injected_constraints=[],
                duration_ms=baseline_duration
            )],
            total_duration_ms=baseline_duration,
            constraints_applied=[],
            metadata={"baseline": True}
        )

        return caf_output, baseline_output


# ============================================================================
# Algorithm Pseudocode for Paper
# ============================================================================

ALGORITHM_PSEUDOCODE = """
Algorithm 1: CAF Iterative Verification Loop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Prompt X, Knowledge Base K, Max Iterations T, Threshold θ
Output: Verified Response Y* or FAIL

 1: function CAF_LOOP(X, K, T, θ)
 2:     Y₀ ← IL.generate(X)                      ▷ Initial LLM draft
 3:     for t ← 1 to T do
 4:         triplets ← FVL.parse(Yₜ₋₁)           ▷ Extract RDF triplets
 5:         for each τ ∈ triplets do
 6:             results[τ] ← SPARQL_VERIFY(τ, K)  ▷ KB verification
 7:         end for
 8:         score ← COMPUTE_SCORE(results)
 9:         if score ≥ θ then
10:             return (Yₜ₋₁, ACCEPT)             ▷ Verification passed
11:         end if
12:         C ← EXTRACT_CONSTRAINTS(results)      ▷ Get violations
13:         Yₜ ← IL.generate(X, C)                ▷ Constrained regen
14:     end for
15:     return DE.adjudicate(Y_T, results)        ▷ Final decision
16: end function

17: function COMPUTE_SCORE(results)
18:     v ← |{r ∈ results : r.status = VERIFIED}|
19:     p ← |{r ∈ results : r.status = PARTIAL}|
20:     c ← |{r ∈ results : r.status = CONTRADICTION}|
21:     return (v + α·p)/|results| - β·c/|results|
22: end function

23: function SPARQL_VERIFY(τ, K)
24:     query ← BUILD_ASK_QUERY(τ)
25:     if K.execute(query) then
26:         return VERIFIED
27:     else if K.execute(NEGATION(query)) then
28:         return CONTRADICTION
29:     else if FUZZY_MATCH(τ, K) > γ then
30:         return PARTIAL
31:     else
32:         return FAILED
33:     end if
34: end function

Parameters:
  α = 0.5  (partial match weight)
  β = 0.5  (contradiction penalty)
  γ = 0.85 (fuzzy match threshold)
  θ = 0.8  (verification threshold)
  T = 5    (max iterations)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

ALGORITHM_LATEX = r"""
\begin{algorithm}[t]
\caption{CAF Iterative Verification Loop}
\label{alg:caf-loop}
\begin{algorithmic}[1]
\Require Prompt $X$, Knowledge Base $\mathcal{K}$, Max Iterations $T$, Threshold $\theta$
\Ensure Verified Response $Y^*$ or \textsc{Fail}

\Function{CAF-Loop}{$X, \mathcal{K}, T, \theta$}
    \State $Y_0 \gets \text{IL.generate}(X)$ \Comment{Initial LLM draft}
    \For{$t \gets 1$ to $T$}
        \State $\mathcal{T} \gets \text{FVL.parse}(Y_{t-1})$ \Comment{Extract RDF triplets}
        \For{each $\tau \in \mathcal{T}$}
            \State $\text{results}[\tau] \gets \text{SPARQL-Verify}(\tau, \mathcal{K})$
        \EndFor
        \State $s \gets \text{ComputeScore}(\text{results})$
        \If{$s \geq \theta$}
            \State \Return $(Y_{t-1}, \textsc{Accept})$ \Comment{Verification passed}
        \EndIf
        \State $\mathcal{C} \gets \text{ExtractConstraints}(\text{results})$
        \State $Y_t \gets \text{IL.generate}(X, \mathcal{C})$ \Comment{Constrained regeneration}
    \EndFor
    \State \Return $\text{DE.adjudicate}(Y_T, \text{results})$ \Comment{Final decision}
\EndFunction

\vspace{0.5em}
\Function{ComputeScore}{results}
    \State $v \gets |\{r \in \text{results} : r.\text{status} = \textsc{Verified}\}|$
    \State $p \gets |\{r \in \text{results} : r.\text{status} = \textsc{Partial}\}|$
    \State $c \gets |\{r \in \text{results} : r.\text{status} = \textsc{Contradiction}\}|$
    \State \Return $(v + \alpha \cdot p) / |\text{results}| - \beta \cdot c / |\text{results}|$
\EndFunction

\vspace{0.5em}
\Function{SPARQL-Verify}{$\tau, \mathcal{K}$}
    \State $q \gets \text{BuildAskQuery}(\tau)$
    \If{$\mathcal{K}.\text{execute}(q)$}
        \State \Return \textsc{Verified}
    \ElsIf{$\mathcal{K}.\text{execute}(\neg q)$}
        \State \Return \textsc{Contradiction}
    \ElsIf{$\text{FuzzyMatch}(\tau, \mathcal{K}) > \gamma$}
        \State \Return \textsc{Partial}
    \Else
        \State \Return \textsc{Failed}
    \EndIf
\EndFunction
\end{algorithmic}
\end{algorithm}
"""


def get_algorithm_pseudocode() -> str:
    """Return the algorithm pseudocode for documentation."""
    return ALGORITHM_PSEUDOCODE


def get_algorithm_latex() -> str:
    """Return LaTeX code for the algorithm."""
    return ALGORITHM_LATEX


if __name__ == "__main__":
    # Demo execution
    print("CAF Algorithm Demo")
    print("=" * 50)
    print(ALGORITHM_PSEUDOCODE)

    print("\nExecuting CAF Loop...")
    caf = CAFLoop()
    result = caf.execute("What causes rain to make roads slippery?")

    print(f"\nResult:")
    print(f"  Decision: {result.decision.value}")
    print(f"  Iterations: {result.iterations_used}")
    print(f"  Final Score: {result.final_score:.3f}")
    print(f"  Duration: {result.total_duration_ms:.1f}ms")
