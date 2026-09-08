"""
Abstract Inference Layer (IL) interface.

Split out of experiments/caf_algorithm.py so common/llm_integration.py (and
anything else that just needs the interface, not the rest of the CAFLoop
algorithm) doesn't have to import experiments/ to get it. experiments/
caf_algorithm.py re-imports InferenceLayer from here, so it's still
importable as `experiments.caf_algorithm.InferenceLayer` for existing code.
"""

from abc import ABC, abstractmethod


class InferenceLayer(ABC):
    """Abstract Inference Layer (IL) interface."""

    @abstractmethod
    def generate(self, prompt: str, constraints: list[str] | None = None) -> str:
        """Generate a response, optionally with constraints."""
        pass

    def generate_batch(
        self,
        prompts: list[str],
        constraints: list[list[str]] | None = None,
        batch_size: int | None = None,
    ) -> list[str]:
        """Generate responses for a batch of prompts."""
        results: list[str] = []
        for idx, prompt in enumerate(prompts):
            per_constraints = (
                constraints[idx] if constraints and idx < len(constraints) else None
            )
            results.append(self.generate(prompt, per_constraints))
        return results
