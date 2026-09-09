# Re-exports from modules.causal_reasoning — do-calculus primitives were moved
# there so the production pipeline (cauval/pipeline.py,
# modules/truth_anchor/causal_graph_builder.py) can import them without
# triggering experiments/__init__.py's numpy/pandas-dependent imports.
from modules.causal_reasoning import (  # noqa: F401
    CausalEdge,
    CausalGraph,
    CounterfactualQuery,
    counterfactual_reasoning,
    counterfactual_reasoning_with_graph,
    normalize_node_id,
    parse_causal_context,
    parse_counterfactual_query,
)
