"""
CAF Experiments Package
=======================
Comprehensive experiment suite for evaluating LLM logical consistency
using the Constraint-Aware Framework (CAF).

Modules:
- synthetic_dataset: Generates synthetic causal chain datasets
- caf_algorithm: CAF iterative verification loop implementation
- metrics: Evaluation metrics (inference depth, contradiction rate, entailment accuracy)
- run_experiment: Main experiment runner
"""

from .caf_algorithm import CAFConfig, CAFLoop, VerificationResult
from .metrics import ExperimentMetrics, MetricsCalculator
from .synthetic_dataset import (
    CausalChain,
    PromptPerturbation,
    SyntheticDatasetGenerator,
)

__all__ = [
    "SyntheticDatasetGenerator",
    "CausalChain",
    "PromptPerturbation",
    "CAFLoop",
    "CAFConfig",
    "VerificationResult",
    "MetricsCalculator",
    "ExperimentMetrics",
]
