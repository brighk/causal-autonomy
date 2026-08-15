"""CAF Modules - Four-component architecture"""
# Module A: Inference Engine (Neural)
from . import inference_engine

# Module B: Semantic Parser (Middleware)
from . import semantic_parser

# Module C: Truth Anchor (Symbolic)
from . import truth_anchor

# Module D: Causal Validator (Verification)
from . import causal_validator

__all__ = [
    'inference_engine',
    'semantic_parser',
    'truth_anchor',
    'causal_validator'
]
