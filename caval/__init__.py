"""
caval - Causal Autonomy Framework, as an importable library.

    from caval import Caval

    caf = Caval()
    result = caf.ask("Does high cpu usage cause increased response time?")
    print(result.text, result.verification_status.is_valid)

Requires two background services to be running (see the top-level README
and examples/quickstart.py):
  1. Fuseki: docker compose -f deployment/docker-compose.yml up -d
  2. The inference engine: uv run python -m modules.inference_engine.server
"""

from api.models import FinalResponse, Triplet, VerificationResult

from .core import Caval
from .exceptions import VerificationFailedError

__version__ = "0.1.1"

__all__ = [
    "Caval",
    "FinalResponse",
    "Triplet",
    "VerificationResult",
    "VerificationFailedError",
]
