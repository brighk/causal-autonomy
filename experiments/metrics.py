"""
Evaluation Metrics for CAF Experiments
======================================
Implements the three primary evaluation metrics from Table 1:

1. Inference Depth (d): Maximum logical steps before contradiction
2. Contradiction Rate (%): Percentage of chains with contradictions
3. Entailment Accuracy: K ∪ {Input} ⊢ Output verification score

Additionally computes:
- Semantic Invariance: Consistency across perturbations
- Per-domain metrics
- Statistical significance measures
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
import json
from enum import Enum

from .synthetic_dataset import CausalChain, PromptPerturbation, PerturbationType
from .caf_algorithm import CAFOutput, VerificationStatus, AdjudicationDecision


class MetricType(Enum):
    """Types of evaluation metrics."""
    INFERENCE_DEPTH = "inference_depth"
    CONTRADICTION_RATE = "contradiction_rate"
    ENTAILMENT_ACCURACY = "entailment_accuracy"
    SEMANTIC_INVARIANCE = "semantic_invariance"


@dataclass
class InferenceDepthResult:
    """Result for inference depth metric."""
    chain_id: str
    max_depth_achieved: int
    contradiction_at_depth: Optional[int]
    ground_truth_depth: int
    depth_ratio: float  # achieved / ground_truth


@dataclass
class ContradictionResult:
    """Result for contradiction detection."""
    chain_id: str
    contradiction_detected: bool
    contradiction_type: Optional[str]
    false_positive: bool
    false_negative: bool


@dataclass
class EntailmentResult:
    """Result for entailment accuracy."""
    chain_id: str
    total_entailments: int
    correct_entailments: int
    accuracy: float
    kb_grounded: bool


@dataclass
class SemanticInvarianceResult:
    """Result for semantic invariance across perturbations."""
    chain_id: str
    original_output: str
    perturbation_outputs: Dict[str, str]
    consistency_scores: Dict[str, float]
    mean_consistency: float
    variance: float


@dataclass
class ExperimentMetrics:
    """
    Aggregate metrics for a complete experiment run.

    Contains all three primary metrics from Table 1 plus
    additional diagnostic metrics.
    """
    # Primary metrics (Table 1)
    mean_inference_depth: float
    std_inference_depth: float
    contradiction_rate_percent: float
    entailment_accuracy: float

    # Secondary metrics
    semantic_invariance_mean: float
    semantic_invariance_std: float

    # Per-domain breakdown
    metrics_by_domain: Dict[str, Dict[str, float]]

    # Statistical measures
    num_chains: int
    num_perturbations: int
    confidence_interval_95: Tuple[float, float]

    # Raw results for analysis
    depth_results: List[InferenceDepthResult] = field(default_factory=list)
    contradiction_results: List[ContradictionResult] = field(default_factory=list)
    entailment_results: List[EntailmentResult] = field(default_factory=list)
    invariance_results: List[SemanticInvarianceResult] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            "primary_metrics": {
                "mean_inference_depth": round(self.mean_inference_depth, 3),
                "std_inference_depth": round(self.std_inference_depth, 3),
                "contradiction_rate_percent": round(self.contradiction_rate_percent, 2),
                "entailment_accuracy": round(self.entailment_accuracy, 4),
            },
            "secondary_metrics": {
                "semantic_invariance_mean": round(self.semantic_invariance_mean, 4),
                "semantic_invariance_std": round(self.semantic_invariance_std, 4),
            },
            "statistics": {
                "num_chains": self.num_chains,
                "num_perturbations": self.num_perturbations,
                "confidence_interval_95": [
                    round(self.confidence_interval_95[0], 4),
                    round(self.confidence_interval_95[1], 4),
                ],
            },
            "by_domain": self.metrics_by_domain,
        }

    def to_latex_table(self) -> str:
        """Generate LaTeX table row for paper."""
        return (
            f"CAF & {self.mean_inference_depth:.2f} $\\pm$ {self.std_inference_depth:.2f} & "
            f"{self.contradiction_rate_percent:.1f}\\% & "
            f"{self.entailment_accuracy:.3f} & "
            f"{self.semantic_invariance_mean:.3f} \\\\"
        )


class MetricsCalculator:
    """
    Calculator for all evaluation metrics.

    Processes CAF outputs against ground truth datasets to compute
    inference depth, contradiction rate, and entailment accuracy.
    """

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize metrics calculator.

        Args:
            similarity_threshold: Threshold for semantic similarity matching
        """
        self.similarity_threshold = similarity_threshold

    def compute_inference_depth(
        self,
        chain: CausalChain,
        caf_output: CAFOutput
    ) -> InferenceDepthResult:
        """
        Compute inference depth for a single chain.

        Inference Depth (d): Maximum logical steps before contradiction.
        Goal: Maximize d.

        Args:
            chain: Ground truth causal chain
            caf_output: CAF execution output

        Returns:
            InferenceDepthResult with depth metrics
        """
        ground_truth_depth = chain.depth

        # Find the depth at which contradiction occurred (if any)
        contradiction_depth = None
        max_depth_achieved = ground_truth_depth

        for log in caf_output.iteration_logs:
            for result in log.verification_results:
                if result.status == VerificationStatus.CONTRADICTION:
                    # Estimate depth based on triplet position
                    triplet_idx = log.extracted_triplets.index(result.triplet)
                    contradiction_depth = triplet_idx + 1
                    max_depth_achieved = min(max_depth_achieved, triplet_idx)
                    break

        # If no contradiction and accepted, achieved full depth
        if caf_output.decision == AdjudicationDecision.ACCEPT and contradiction_depth is None:
            max_depth_achieved = ground_truth_depth

        depth_ratio = max_depth_achieved / ground_truth_depth if ground_truth_depth > 0 else 0

        return InferenceDepthResult(
            chain_id=chain.chain_id,
            max_depth_achieved=max_depth_achieved,
            contradiction_at_depth=contradiction_depth,
            ground_truth_depth=ground_truth_depth,
            depth_ratio=depth_ratio
        )

    def compute_contradiction_rate(
        self,
        chain: CausalChain,
        caf_output: CAFOutput
    ) -> ContradictionResult:
        """
        Compute contradiction detection for a single chain.

        Contradiction Rate: Percentage of chains where contradictions detected.
        Goal: Minimize for clean chains, maximize detection for injected contradictions.

        Args:
            chain: Ground truth causal chain (may have injected contradictions)
            caf_output: CAF execution output

        Returns:
            ContradictionResult with detection metrics
        """
        # Check if contradiction was detected by CAF
        detected = any(
            result.status == VerificationStatus.CONTRADICTION
            for log in caf_output.iteration_logs
            for result in log.verification_results
        )

        # Check ground truth
        has_injected = len(chain.injected_contradictions) > 0

        # Compute FP/FN
        false_positive = detected and not has_injected
        false_negative = not detected and has_injected

        contradiction_type = None
        if has_injected:
            contradiction_type = "injected"
        elif detected:
            contradiction_type = "emergent"

        return ContradictionResult(
            chain_id=chain.chain_id,
            contradiction_detected=detected,
            contradiction_type=contradiction_type,
            false_positive=false_positive,
            false_negative=false_negative
        )

    def compute_entailment_accuracy(
        self,
        chain: CausalChain,
        caf_output: CAFOutput
    ) -> EntailmentResult:
        """
        Compute entailment accuracy for a single chain.

        Entailment Accuracy: K ∪ {Input} ⊢ Output
        Goal: Target 1.0 (all outputs entailed by KB + input)

        Args:
            chain: Ground truth causal chain with expected entailments
            caf_output: CAF execution output

        Returns:
            EntailmentResult with accuracy metrics
        """
        total_entailments = len(chain.ground_truth_entailments)

        if total_entailments == 0:
            return EntailmentResult(
                chain_id=chain.chain_id,
                total_entailments=0,
                correct_entailments=0,
                accuracy=1.0,
                kb_grounded=True
            )

        # Count verified entailments from CAF output
        correct = 0
        for log in caf_output.iteration_logs:
            for result in log.verification_results:
                if result.status in [VerificationStatus.VERIFIED, VerificationStatus.PARTIAL]:
                    correct += 1

        # Normalize by total expected
        # Use final iteration results as the measure
        final_results = caf_output.iteration_logs[-1].verification_results if caf_output.iteration_logs else []
        verified_count = sum(
            1 for r in final_results
            if r.status == VerificationStatus.VERIFIED
        )
        partial_count = sum(
            1 for r in final_results
            if r.status == VerificationStatus.PARTIAL
        )

        total_checked = len(final_results) if final_results else 1
        accuracy = (verified_count + 0.5 * partial_count) / total_checked

        kb_grounded = caf_output.final_score >= 0.8

        return EntailmentResult(
            chain_id=chain.chain_id,
            total_entailments=total_entailments,
            correct_entailments=verified_count,
            accuracy=accuracy,
            kb_grounded=kb_grounded
        )

    def compute_semantic_invariance(
        self,
        chain: CausalChain,
        original_output: CAFOutput,
        perturbation_outputs: Dict[PerturbationType, CAFOutput]
    ) -> SemanticInvarianceResult:
        """
        Compute semantic invariance across prompt perturbations.

        Semantic Invariance: Consistency across P and ¬(¬P).
        Goal: Minimize variance in outputs across perturbations.

        Args:
            chain: Original causal chain
            original_output: CAF output for original prompt
            perturbation_outputs: Outputs for each perturbation type

        Returns:
            SemanticInvarianceResult with consistency metrics
        """
        original_score = original_output.final_score

        consistency_scores = {}
        perturbation_output_texts = {}

        for ptype, output in perturbation_outputs.items():
            # Consistency = 1 - |original_score - perturbed_score|
            perturbed_score = output.final_score
            consistency = 1.0 - abs(original_score - perturbed_score)
            consistency_scores[ptype.value] = consistency
            perturbation_output_texts[ptype.value] = output.final_response

        scores = list(consistency_scores.values())
        mean_consistency = np.mean(scores) if scores else 1.0
        variance = np.var(scores) if scores else 0.0

        return SemanticInvarianceResult(
            chain_id=chain.chain_id,
            original_output=original_output.final_response,
            perturbation_outputs=perturbation_output_texts,
            consistency_scores=consistency_scores,
            mean_consistency=float(mean_consistency),
            variance=float(variance)
        )

    def compute_all_metrics(
        self,
        chains: List[CausalChain],
        caf_outputs: List[CAFOutput],
        perturbation_outputs: Optional[List[Dict[PerturbationType, CAFOutput]]] = None
    ) -> ExperimentMetrics:
        """
        Compute all metrics for a complete experiment.

        Args:
            chains: List of ground truth causal chains
            caf_outputs: CAF outputs for each chain
            perturbation_outputs: Optional outputs for perturbations

        Returns:
            ExperimentMetrics with all computed metrics
        """
        depth_results = []
        contradiction_results = []
        entailment_results = []
        invariance_results = []

        # Per-domain aggregation
        domain_metrics = defaultdict(lambda: {
            "depths": [], "contradictions": [], "accuracies": []
        })

        for i, (chain, output) in enumerate(zip(chains, caf_outputs)):
            # Compute individual metrics
            depth_result = self.compute_inference_depth(chain, output)
            contradiction_result = self.compute_contradiction_rate(chain, output)
            entailment_result = self.compute_entailment_accuracy(chain, output)

            depth_results.append(depth_result)
            contradiction_results.append(contradiction_result)
            entailment_results.append(entailment_result)

            # Aggregate by domain
            domain_metrics[chain.domain]["depths"].append(depth_result.max_depth_achieved)
            domain_metrics[chain.domain]["contradictions"].append(
                1 if contradiction_result.contradiction_detected else 0
            )
            domain_metrics[chain.domain]["accuracies"].append(entailment_result.accuracy)

            # Compute semantic invariance if perturbation outputs provided
            if perturbation_outputs and i < len(perturbation_outputs):
                invariance_result = self.compute_semantic_invariance(
                    chain, output, perturbation_outputs[i]
                )
                invariance_results.append(invariance_result)

        # Aggregate primary metrics
        depths = [r.max_depth_achieved for r in depth_results]
        mean_depth = float(np.mean(depths))
        std_depth = float(np.std(depths))

        contradiction_count = sum(1 for r in contradiction_results if r.contradiction_detected)
        contradiction_rate = (contradiction_count / len(contradiction_results)) * 100

        accuracies = [r.accuracy for r in entailment_results]
        mean_accuracy = float(np.mean(accuracies))

        # Semantic invariance
        if invariance_results:
            invariance_means = [r.mean_consistency for r in invariance_results]
            invariance_mean = float(np.mean(invariance_means))
            invariance_std = float(np.std(invariance_means))
        else:
            invariance_mean = 0.0
            invariance_std = 0.0

        # Per-domain metrics
        metrics_by_domain = {}
        for domain, data in domain_metrics.items():
            metrics_by_domain[domain] = {
                "mean_depth": round(float(np.mean(data["depths"])), 2),
                "contradiction_rate": round(
                    float(np.mean(data["contradictions"])) * 100, 2
                ),
                "entailment_accuracy": round(float(np.mean(data["accuracies"])), 4),
            }

        # 95% confidence interval for accuracy
        n = len(accuracies)
        if n > 1:
            se = np.std(accuracies) / np.sqrt(n)
            ci_lower = mean_accuracy - 1.96 * se
            ci_upper = mean_accuracy + 1.96 * se
        else:
            ci_lower = ci_upper = mean_accuracy

        total_perturbations = sum(len(c.perturbations) for c in chains)

        return ExperimentMetrics(
            mean_inference_depth=mean_depth,
            std_inference_depth=std_depth,
            contradiction_rate_percent=contradiction_rate,
            entailment_accuracy=mean_accuracy,
            semantic_invariance_mean=invariance_mean,
            semantic_invariance_std=invariance_std,
            metrics_by_domain=metrics_by_domain,
            num_chains=len(chains),
            num_perturbations=total_perturbations,
            confidence_interval_95=(float(ci_lower), float(ci_upper)),
            depth_results=depth_results,
            contradiction_results=contradiction_results,
            entailment_results=entailment_results,
            invariance_results=invariance_results,
        )


def compute_baseline_comparison(
    caf_metrics: ExperimentMetrics,
    baseline_metrics: ExperimentMetrics
) -> Dict[str, Any]:
    """
    Compute comparison between CAF and baseline metrics.

    Args:
        caf_metrics: Metrics from CAF experiment
        baseline_metrics: Metrics from baseline (no CAF) experiment

    Returns:
        Dictionary with improvement percentages and deltas
    """
    def pct_improvement(caf_val, baseline_val):
        if baseline_val == 0:
            return float('inf') if caf_val > 0 else 0.0
        return ((caf_val - baseline_val) / baseline_val) * 100

    comparison = {
        "inference_depth": {
            "caf": caf_metrics.mean_inference_depth,
            "baseline": baseline_metrics.mean_inference_depth,
            "improvement_pct": pct_improvement(
                caf_metrics.mean_inference_depth,
                baseline_metrics.mean_inference_depth
            ),
            "delta": caf_metrics.mean_inference_depth - baseline_metrics.mean_inference_depth,
        },
        "contradiction_rate": {
            "caf": caf_metrics.contradiction_rate_percent,
            "baseline": baseline_metrics.contradiction_rate_percent,
            "reduction_pct": pct_improvement(
                baseline_metrics.contradiction_rate_percent,
                caf_metrics.contradiction_rate_percent
            ),
            "delta": baseline_metrics.contradiction_rate_percent - caf_metrics.contradiction_rate_percent,
        },
        "entailment_accuracy": {
            "caf": caf_metrics.entailment_accuracy,
            "baseline": baseline_metrics.entailment_accuracy,
            "improvement_pct": pct_improvement(
                caf_metrics.entailment_accuracy,
                baseline_metrics.entailment_accuracy
            ),
            "delta": caf_metrics.entailment_accuracy - baseline_metrics.entailment_accuracy,
        },
        "semantic_invariance": {
            "caf": caf_metrics.semantic_invariance_mean,
            "baseline": baseline_metrics.semantic_invariance_mean,
            "improvement_pct": pct_improvement(
                caf_metrics.semantic_invariance_mean,
                baseline_metrics.semantic_invariance_mean
            ),
            "delta": caf_metrics.semantic_invariance_mean - baseline_metrics.semantic_invariance_mean,
        },
    }

    return comparison


def generate_latex_results_table(
    caf_metrics: ExperimentMetrics,
    baseline_metrics: ExperimentMetrics
) -> str:
    """
    Generate LaTeX table for paper results section.

    Args:
        caf_metrics: CAF experiment metrics
        baseline_metrics: Baseline experiment metrics

    Returns:
        LaTeX table code
    """
    comparison = compute_baseline_comparison(caf_metrics, baseline_metrics)

    latex = r"""
\begin{table}[t]
\centering
\caption{Experimental Results: CAF vs Baseline}
\label{tab:results}
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{Baseline} & \textbf{CAF} & \textbf{Improvement} \\
\midrule
"""
    latex += f"Inference Depth (d) & {baseline_metrics.mean_inference_depth:.2f} & "
    latex += f"{caf_metrics.mean_inference_depth:.2f} & "
    latex += f"+{comparison['inference_depth']['improvement_pct']:.1f}\\% \\\\\n"

    latex += f"Contradiction Rate & {baseline_metrics.contradiction_rate_percent:.1f}\\% & "
    latex += f"{caf_metrics.contradiction_rate_percent:.1f}\\% & "
    latex += f"-{comparison['contradiction_rate']['delta']:.1f}pp \\\\\n"

    latex += f"Entailment Accuracy & {baseline_metrics.entailment_accuracy:.3f} & "
    latex += f"{caf_metrics.entailment_accuracy:.3f} & "
    latex += f"+{comparison['entailment_accuracy']['improvement_pct']:.1f}\\% \\\\\n"

    latex += f"Semantic Invariance & {baseline_metrics.semantic_invariance_mean:.3f} & "
    latex += f"{caf_metrics.semantic_invariance_mean:.3f} & "
    latex += f"+{comparison['semantic_invariance']['improvement_pct']:.1f}\\% \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    return latex


if __name__ == "__main__":
    # Demo: Generate sample metrics
    from .synthetic_dataset import SyntheticDatasetGenerator
    from .caf_algorithm import CAFLoop

    print("Metrics Calculator Demo")
    print("=" * 50)

    # Generate small test dataset
    generator = SyntheticDatasetGenerator(seed=42)
    dataset = generator.generate_dataset(num_chains=10, perturbations_per_chain=2)

    # Run CAF on each chain
    caf = CAFLoop()
    outputs = [caf.execute(chain.to_prompt()) for chain in dataset]

    # Compute metrics
    calculator = MetricsCalculator()
    metrics = calculator.compute_all_metrics(dataset, outputs)

    print(f"\nResults:")
    print(f"  Mean Inference Depth: {metrics.mean_inference_depth:.2f} ± {metrics.std_inference_depth:.2f}")
    print(f"  Contradiction Rate: {metrics.contradiction_rate_percent:.1f}%")
    print(f"  Entailment Accuracy: {metrics.entailment_accuracy:.4f}")
    print(f"  95% CI: [{metrics.confidence_interval_95[0]:.4f}, {metrics.confidence_interval_95[1]:.4f}]")
