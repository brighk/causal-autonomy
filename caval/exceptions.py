"""Exceptions raised by the caval public API."""


class VerificationFailedError(Exception):
    """
    Raised when a prompt couldn't be grounded in the knowledge base within
    the allotted refinement iterations.

    Mirrors the 422 api/main.py raises for the same condition over HTTP -
    this is the library-facing equivalent, carrying the same information.
    """

    def __init__(self, contradictions: list[str], refinement_iterations: int):
        self.contradictions = contradictions
        self.refinement_iterations = refinement_iterations
        super().__init__(
            f"Unable to ground response in knowledge base after "
            f"{refinement_iterations} refinement iterations. "
            f"Contradictions: {contradictions}"
        )
