"""
The CAF generate -> parse -> verify -> constrain -> regenerate loop,
extracted from api/main.py's POST /v1/infer handler so both the FastAPI
gateway and caval's own Caval class run the exact same logic instead of two
copies that can silently drift apart.
"""

from loguru import logger

from api.models import FinalResponse, InferenceRequest

from .exceptions import VerificationFailedError


class CAFPipeline:
    """
    Composes the four pipeline stages (inference, semantic parsing, truth
    anchoring, causal validation) into one refinement loop.

    Takes already-constructed service instances rather than building them
    itself - callers (api/main.py's lifespan-managed singletons, or
    caval.Caval's per-instance/per-call services) own construction and
    lifecycle.
    """

    def __init__(self, *, inference, parser, truth_anchor, causal_validator):
        self.inference = inference
        self.parser = parser
        self.truth_anchor = truth_anchor
        self.causal_validator = causal_validator

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

        Raises:
            VerificationFailedError: if the response can't be grounded in
                the knowledge base within max_refinement_iterations.
        """
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
