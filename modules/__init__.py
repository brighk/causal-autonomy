"""CAF Modules - Four-component architecture"""

# Module A: Inference Engine (Neural)
# Module B: Semantic Parser (Middleware)
# Module C: Truth Anchor (Symbolic)
# Module D: Causal Validator (Verification)
from . import causal_validator, inference_engine, semantic_parser, truth_anchor

__all__ = ["inference_engine", "semantic_parser", "truth_anchor", "causal_validator"]
