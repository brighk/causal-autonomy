"""Test helpers shared across tests/."""

from api.models import CausalAssertion, ResponseCandidate


class StubInference:
    """
    Duck-typed inference stub for CAFPipeline/Cauval tests - only needs an
    async .generate(request) -> ResponseCandidate method (CAFPipeline
    doesn't require a specific class, just this shape).

    `texts` as a list lets a test simulate "the LLM fixes itself on retry"
    to exercise CAFPipeline's refinement loop - one entry per call, holds
    on the last entry once exhausted.
    """

    def __init__(
        self,
        texts: str | list[str],
        causal_assertions: list[CausalAssertion] | None = None,
    ):
        self._texts = [texts] if isinstance(texts, str) else list(texts)
        self._causal_assertions = causal_assertions or []
        self.call_count = 0

    async def generate(self, request) -> ResponseCandidate:
        idx = min(self.call_count, len(self._texts) - 1)
        text = self._texts[idx]
        self.call_count += 1
        return ResponseCandidate(text=text, causal_assertions=self._causal_assertions)
