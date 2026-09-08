"""
Caval() construction/wiring smoke test. No `fuseki` marker: per Caval's own
docstring, the constructor doesn't touch the network - it only constructs
SemanticParser (loads spaCy, no network) and TruthAnchor/CausalValidator
(no eager I/O either) - so this runs without Fuseki being reachable.
"""

from caval.core import Caval
from modules.causal_validator.validator import CausalValidator
from modules.semantic_parser.parser import SemanticParser
from modules.truth_anchor.verifier import TruthAnchor


def test_caval_construction_smoke():
    caf = Caval(fuseki_endpoint="http://localhost:3030/dataset/query")

    assert isinstance(caf._parser, SemanticParser)
    assert isinstance(caf._truth_anchor, TruthAnchor)
    assert isinstance(caf._causal_validator, CausalValidator)
