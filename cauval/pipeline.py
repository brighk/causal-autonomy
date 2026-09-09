"""
The CAF generate -> parse -> verify -> constrain -> regenerate loop,
extracted from api/main.py's POST /v1/infer handler so both the FastAPI
gateway and caval's own Cauval class run the exact same logic instead of two
copies that can silently drift apart.

Pearl's causal hierarchy routing:
  Level 1 (observational): factual SPARQL lookup + LLM refinement loop.
  Level 2/3 (interventional/counterfactual): do-calculus via CausalGraphBuilder,
      NO LLM involved — the KB is the sole source of truth.
"""

from loguru import logger

from api.models import FinalResponse, InferenceRequest, VerificationResult
from experiments.intervention_calculus import (
    counterfactual_reasoning_with_graph,
    parse_counterfactual_query,
)

from .exceptions import VerificationFailedError


class CAFPipeline:
    """
    Composes the four pipeline stages (inference, semantic parsing, truth
    anchoring, causal validation) into one refinement loop.

    Takes already-constructed service instances rather than building them
    itself - callers (api/main.py's lifespan-managed singletons, or
    cauval.Cauval's per-instance/per-call services) own construction and
    lifecycle.

    If causal_graph_builder is provided, interventional and counterfactual
    queries (Pearl Level 2/3) are answered by do-calculus against the live
    KB without invoking the LLM at all.  Factual queries (Level 1) use the
    existing generate -> verify -> constrain -> regenerate loop.
    """

    def __init__(
        self,
        *,
        inference,
        parser,
        truth_anchor,
        causal_validator,
        causal_graph_builder=None,
    ):
        self.inference = inference
        self.parser = parser
        self.truth_anchor = truth_anchor
        self.causal_validator = causal_validator
        self.causal_graph_builder = causal_graph_builder

    async def run(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        max_refinement_iterations: int = 3,
        verification_threshold: float = 0.8,
        enable_causal_validation: bool = True,
    ) -> FinalResponse:
        """
        Run the full loop for one prompt.

        For Level 2/3 queries (counterfactual patterns detected by
        parse_counterfactual_query), bypasses the LLM entirely and answers
        via do-calculus against the live KB.  Falls through to the Level 1
        LLM+SPARQL path if the KB has no relevant causal edges or the query
        can't be parsed.

        Raises:
            VerificationFailedError: if the Level 1 path can't ground the
                response within max_refinement_iterations.
        """
        # --- Level 2 / Level 3 routing (no LLM) ---
        if self.causal_graph_builder is not None:
            parsed_cf = parse_counterfactual_query(prompt)
            if parsed_cf is not None:
                graph = await self.causal_graph_builder.build(
                    [parsed_cf["target"], parsed_cf["intervention_node"]]
                )
                if graph.edges:
                    outcome = counterfactual_reasoning_with_graph(prompt, graph)
                    if outcome is not None:
                        do_expr = (
                            f"do({parsed_cf['intervention_node']}"
                            f"={parsed_cf['intervention_value']})"
                        )
                        answer = (
                            f"{'Yes' if outcome else 'No'}. "
                            f"Under intervention {do_expr}, "
                            f"{parsed_cf['target']} would "
                            f"{'occur' if outcome else 'not occur'}."
                        )
                        return FinalResponse(
                            text=answer,
                            verification_status=VerificationResult(
                                is_valid=True,
                                matched_triplets=[],
                                contradictions=[],
                                verification_method="do_calculus",
                            ),
                            refinement_iterations=0,
                            causal_grounding=[],
                            confidence=1.0,
                            session_id=session_id,
                            causal_level=2,
                        )
                logger.warning(
                    "Level 2/3 routing: no KB causal edges found or "
                    "unparseable query — falling through to Level 1"
                )

        # --- Level 1: generate -> parse -> verify -> constrain -> regenerate ---
        inference_req = InferenceRequest(prompt=prompt, session_id=session_id)
        response_candidate = await self.inference.generate(inference_req)

        parsed_result = await self.parser.parse(
            response_candidate.text, response_candidate.causal_assertions
        )

        refinement_count = 0
        is_verified = False
        verification_result = None

        while refinement_count < max_refinement_iterations and not is_verified:
            logger.info(f"Verification iteration {refinement_count + 1}")
            verification_result = await self.truth_anchor.verify(
                triplets=parsed_result.triplets,
                threshold=verification_threshold,
            )

            if enable_causal_validation:
                causal_check = await self.causal_validator.validate(
                    assertions=response_candidate.causal_assertions,
                    verified_triplets=verification_result.matched_triplets,
                )

                if not causal_check.is_valid:
                    verification_result.is_valid = False
                    verification_result.contradictions.extend(causal_check.violations)

            if verification_result.is_valid:
                is_verified = True
                break

            refinement_count += 1
            if refinement_count < max_refinement_iterations:
                # Combined for feedback purposes: the LLM needs "what to
                # fix" regardless of whether a triplet was checked and
                # contradicted, or never resolved to a real KB entity in
                # the first place - VerificationResult keeps the two lists
                # separate for callers who want the distinction, but a
                # retry prompt with only contradictions and an empty
                # unverifiable list would go out with a useless blank
                # constraint if every triplet fell in the latter bucket.
                feedback = (
                    verification_result.contradictions
                    + verification_result.unverifiable
                )
                logger.warning(
                    f"Verification failed. Re-running with constraints. "
                    f"Feedback: {feedback}"
                )

                constrained_prompt = (
                    f"{prompt}\n\n"
                    f"CONSTRAINT: Avoid these contradictions: {', '.join(feedback)}"
                )

                inference_req.prompt = constrained_prompt
                response_candidate = await self.inference.generate(inference_req)
                parsed_result = await self.parser.parse(
                    response_candidate.text, response_candidate.causal_assertions
                )

        if not is_verified:
            logger.error(f"Failed to verify after {refinement_count} iterations")
            raise VerificationFailedError(
                contradictions=(
                    verification_result.contradictions
                    + verification_result.unverifiable
                ),
                refinement_iterations=refinement_count,
            )

        return FinalResponse(
            text=response_candidate.text,
            verification_status=verification_result,
            refinement_iterations=refinement_count,
            causal_grounding=verification_result.matched_triplets,
            confidence=verification_result.similarity_score or 1.0,
            session_id=session_id,
        )
