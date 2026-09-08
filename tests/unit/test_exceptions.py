"""Pure-logic tests for caval.exceptions."""

from caval.exceptions import VerificationFailedError


def test_verification_failed_error_attrs_and_message():
    error = VerificationFailedError(
        contradictions=["x contradicts y"], refinement_iterations=2
    )

    assert error.contradictions == ["x contradicts y"]
    assert error.refinement_iterations == 2
    message = str(error)
    assert "2" in message
    assert "x contradicts y" in message
