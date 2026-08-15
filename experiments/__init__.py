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

from .synthetic_dataset import SyntheticDatasetGenerator, CausalChain, PromptPerturbation
from .caf_algorithm import CAFLoop, CAFConfig, VerificationResult
from .metrics import MetricsCalculator, ExperimentMetrics

__all__ = [
    'SyntheticDatasetGenerator',
    'CausalChain',
    'PromptPerturbation',
    'CAFLoop',
    'CAFConfig',
    'VerificationResult',
    'MetricsCalculator',
    'ExperimentMetrics',
]
