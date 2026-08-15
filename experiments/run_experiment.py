"""
CAF Full Experiment Runner
==========================
Executes the complete experiment pipeline:

1. Generate synthetic causal chain dataset (75 chains, 3 perturbations each)
2. Run CAF verification loop on all chains
3. Compute all metrics (inference depth, contradiction rate, entailment accuracy)
4. Generate paper-ready artifacts (tables, figures, JSON results)

Usage:
    python -m experiments.run_experiment [--output-dir OUTPUT_DIR] [--num-chains N]
"""

import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import asdict
import random

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.synthetic_dataset import (
    SyntheticDatasetGenerator,
    CausalChain,
    PerturbationType,
)
from experiments.caf_algorithm import (
    CAFLoop,
    CAFConfig,
    CAFOutput,
    get_algorithm_pseudocode,
    get_algorithm_latex,
)
from experiments.metrics import (
    MetricsCalculator,
    ExperimentMetrics,
    compute_baseline_comparison,
    generate_latex_results_table,
)
from experiments.baselines import (
    create_vanilla_baseline,
    create_cot_baseline,
    create_rag_baseline,
    create_rag_cot_baseline,
)


class ExperimentRunner:
    """
    Complete experiment runner for CAF evaluation.

    Orchestrates dataset generation, CAF execution, and metrics computation
    to produce paper-ready results.
    """

    def __init__(
        self,
        output_dir: str = "experiments/results",
        seed: int = 42,
        verbose: bool = True,
        inference_layer = None,
        verification_layer = None  # NEW: Allow custom FVL
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.verbose = verbose
        random.seed(seed)

        # Initialize components
        self.generator = SyntheticDatasetGenerator(seed=seed)
        self.caf = CAFLoop(
            config=CAFConfig(
                max_iterations=5,
                verification_threshold=0.8,
            ),
            inference_layer=inference_layer,  # Use provided IL or default to simulated
            verification_layer=verification_layer  # Use provided FVL or default to simulated
        )
        self.metrics_calc = MetricsCalculator()

        # Initialize baselines (if we have a real LLM)
        self.inference_layer = inference_layer
        self.vanilla_baseline = None
        self.cot_baseline = None
        self.rag_baseline = None
        self.rag_cot_baseline = None

        if inference_layer is not None:
            self.vanilla_baseline = create_vanilla_baseline(inference_layer)
            self.cot_baseline = create_cot_baseline(inference_layer, num_steps=3)
            self.rag_baseline = create_rag_baseline(inference_layer, top_k=3)
            self.rag_cot_baseline = create_rag_cot_baseline(inference_layer, top_k=3, num_steps=3)

        # Results storage
        self.dataset: List[CausalChain] = []
        self.caf_outputs: List[CAFOutput] = []
        self.baseline_outputs: List[CAFOutput] = []
        self.cot_outputs: List[CAFOutput] = []
        self.rag_outputs: List[CAFOutput] = []
        self.rag_cot_outputs: List[CAFOutput] = []
        self.perturbation_outputs: List[Dict[PerturbationType, CAFOutput]] = []
        # Baseline outputs on perturbed prompts, needed to measure baseline
        # semantic invariance under the same protocol as CAF.
        self.baseline_pert_outputs: Dict[str, List[Dict[PerturbationType, CAFOutput]]] = {
            "Vanilla": [], "CoT": [], "RAG": [], "RAG+CoT": []
        }

    def log(self, message: str):
        """Log message if verbose mode enabled."""
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    def generate_dataset(
        self,
        num_chains: int = 75,
        perturbations_per_chain: int = 3
    ) -> List[CausalChain]:
        """
        Generate synthetic causal chain dataset.

        Args:
            num_chains: Number of chains (50-100 for paper)
            perturbations_per_chain: Perturbations per chain (2-3 for paper)

        Returns:
            List of generated CausalChain objects
        """
        self.log(f"Generating dataset: {num_chains} chains, {perturbations_per_chain} perturbations each...")

        self.dataset = self.generator.generate_dataset(
            num_chains=num_chains,
            min_depth=2,
            max_depth=10,
            perturbations_per_chain=perturbations_per_chain,
            contradiction_rate=0.2,
            domains=["physics", "biology", "economics", "logic", "causality"]
        )

        self.log(f"Generated {len(self.dataset)} chains with "
                 f"{sum(len(c.perturbations) for c in self.dataset)} total perturbations")

        return self.dataset

    def run_caf_evaluation(self) -> Tuple[List[CAFOutput], List[CAFOutput]]:
        """
        Run CAF evaluation and all baselines on all chains.

        Returns:
            Tuple of (caf_outputs, baseline_outputs)
        """
        self.log("Running CAF and baseline evaluations...")
        self.caf_outputs = []
        self.baseline_outputs = []
        self.cot_outputs = []
        self.rag_outputs = []
        self.rag_cot_outputs = []
        self.perturbation_outputs = []
        self.baseline_pert_outputs = {
            name: [] for name in ("Vanilla", "CoT", "RAG", "RAG+CoT")
        }

        use_baselines = self.inference_layer is not None

        for i, chain in enumerate(self.dataset):
            self.log(f"Chain {i + 1}/{len(self.dataset)} | domain={chain.domain} depth={chain.depth} contradictions={len(chain.injected_contradictions)}")

            # Run CAF on original prompt
            self.log(f"  [CAF] running...")
            caf_output, baseline_output = self.caf.execute_with_baseline(
                chain.to_prompt()
            )
            self.log(f"  [CAF] done: iters={caf_output.iterations_used} score={caf_output.final_score:.3f} decision={caf_output.decision.value}")
            self.caf_outputs.append(caf_output)
            self.baseline_outputs.append(baseline_output)

            # Run additional baselines (CoT, RAG, RAG+CoT)
            if use_baselines:
                # Set up knowledge base for RAG baselines
                if self.rag_baseline:
                    self.rag_baseline.set_knowledge_base(chain)
                if self.rag_cot_baseline:
                    self.rag_cot_baseline.set_knowledge_base(chain)

                # Generate with each baseline
                prompt = chain.to_prompt()

                # Chain of Thought
                self.log(f"  [CoT] running...")
                cot_response = self.cot_baseline.generate(prompt)
                cot_output = self._create_output_from_response(cot_response, knowledge_base=None)
                self.log(f"  [CoT] done: score={cot_output.final_score:.3f}")
                self.cot_outputs.append(cot_output)

                # RAG
                self.log(f"  [RAG] running...")
                rag_response = self.rag_baseline.generate(prompt, domain=chain.domain)
                rag_output = self._create_output_from_response(rag_response, knowledge_base=None)
                self.log(f"  [RAG] done: score={rag_output.final_score:.3f}")
                self.rag_outputs.append(rag_output)

                # RAG + CoT (strongest baseline)
                self.log(f"  [RAG+CoT] running...")
                rag_cot_response = self.rag_cot_baseline.generate(prompt, domain=chain.domain)
                rag_cot_output = self._create_output_from_response(rag_cot_response, knowledge_base=None)
                self.log(f"  [RAG+CoT] done: score={rag_cot_output.final_score:.3f}")
                self.rag_cot_outputs.append(rag_cot_output)

            # Run on perturbations (CAF always; baselines when a real LLM is available,
            # so that semantic invariance can be measured for every method)
            perturbation_results = {}
            baseline_pert_results = {name: {} for name in self.baseline_pert_outputs}
            for perturbation in chain.perturbations:
                ptype = perturbation.perturbation_type
                self.log(f"  [perturbation={ptype.value}] CAF running...")
                pert_output = self.caf.execute(perturbation.perturbed_prompt)
                self.log(f"  [perturbation={ptype.value}] CAF done: score={pert_output.final_score:.3f}")
                perturbation_results[perturbation.perturbation_type] = pert_output

                if use_baselines:
                    pert_prompt = perturbation.perturbed_prompt

                    self.log(f"  [perturbation={ptype.value}] baselines running...")
                    vanilla_response = self.vanilla_baseline.generate(pert_prompt)
                    baseline_pert_results["Vanilla"][ptype] = \
                        self._create_output_from_response(vanilla_response, knowledge_base=None)

                    cot_response = self.cot_baseline.generate(pert_prompt)
                    baseline_pert_results["CoT"][ptype] = \
                        self._create_output_from_response(cot_response, knowledge_base=None)

                    rag_response = self.rag_baseline.generate(pert_prompt, domain=chain.domain)
                    baseline_pert_results["RAG"][ptype] = \
                        self._create_output_from_response(rag_response, knowledge_base=None)

                    rag_cot_response = self.rag_cot_baseline.generate(pert_prompt, domain=chain.domain)
                    baseline_pert_results["RAG+CoT"][ptype] = \
                        self._create_output_from_response(rag_cot_response, knowledge_base=None)

            self.perturbation_outputs.append(perturbation_results)
            if use_baselines:
                for name in self.baseline_pert_outputs:
                    self.baseline_pert_outputs[name].append(baseline_pert_results[name])

        if use_baselines:
            self.log(f"Completed CAF + 4 baselines on {len(self.dataset)} chains")
        else:
            self.log(f"Completed CAF evaluation on {len(self.dataset)} chains (simulation mode)")
        return self.caf_outputs, self.baseline_outputs

    def _create_output_from_response(self, response: str, knowledge_base=None) -> CAFOutput:
        """
        Helper to create a CAFOutput from a baseline response.

        Performs the same triplet extraction and verification as Vanilla baseline
        to enable fair metric comparison between all baselines.
        """
        from experiments.caf_algorithm import (
            AdjudicationDecision,
            IterationLog,
            SimulatedInferenceLayer,
            SimulatedFVL
        )
        import time

        start_time = time.time()

        # Extract triplets from response (same as Vanilla baseline)
        triplets = self.caf.fvl.parse(response)

        # Verify triplets against knowledge base (same as Vanilla baseline)
        if isinstance(self.caf.fvl, SimulatedFVL):
            # For simulated FVL, provide a reasonable accuracy hint
            verification_results = self.caf.fvl.verify(triplets, knowledge_base, accuracy_hint=0.6)
        else:
            verification_results = self.caf.fvl.verify(triplets, knowledge_base)

        # Compute overall score
        score = self.caf.compute_score(verification_results)
        duration_ms = (time.time() - start_time) * 1000

        # Create iteration log with real verification results
        iteration_log = IterationLog(
            iteration=1,
            draft_response=response,
            extracted_triplets=triplets,
            verification_results=verification_results,
            overall_score=score,
            injected_constraints=[],
            duration_ms=duration_ms
        )

        # Create CAFOutput matching the structure of Vanilla baseline
        return CAFOutput(
            final_response=response,
            decision=AdjudicationDecision.ACCEPT,
            iterations_used=1,
            final_score=score,
            iteration_logs=[iteration_log],
            total_duration_ms=duration_ms,
            constraints_applied=[],
            metadata={"baseline": True}
        )

    def compute_metrics(self) -> Dict[str, ExperimentMetrics]:
        """
        Compute all evaluation metrics for CAF and all baselines.

        Returns:
            Dict mapping method name to ExperimentMetrics
        """
        self.log("Computing metrics for all methods...")

        metrics = {}

        # CAF metrics
        metrics["CAF"] = self.metrics_calc.compute_all_metrics(
            self.dataset,
            self.caf_outputs,
            self.perturbation_outputs
        )

        # Vanilla baseline metrics (perturbation outputs enable semantic
        # invariance measurement; empty list falls back to None in simulation mode)
        metrics["Vanilla"] = self.metrics_calc.compute_all_metrics(
            self.dataset,
            self.baseline_outputs,
            self.baseline_pert_outputs["Vanilla"] or None
        )

        # Additional baselines (if using real LLM)
        if self.inference_layer is not None:
            metrics["CoT"] = self.metrics_calc.compute_all_metrics(
                self.dataset,
                self.cot_outputs,
                self.baseline_pert_outputs["CoT"] or None
            )

            metrics["RAG"] = self.metrics_calc.compute_all_metrics(
                self.dataset,
                self.rag_outputs,
                self.baseline_pert_outputs["RAG"] or None
            )

            metrics["RAG+CoT"] = self.metrics_calc.compute_all_metrics(
                self.dataset,
                self.rag_cot_outputs,
                self.baseline_pert_outputs["RAG+CoT"] or None
            )

        self.log(f"Metrics computed for {len(metrics)} methods")
        return metrics

    def export_results(
        self,
        all_metrics: Dict[str, ExperimentMetrics]
    ) -> Dict[str, str]:
        """
        Export all results to files.

        Args:
            all_metrics: Dictionary mapping method name to metrics

        Returns:
            Dictionary mapping result type to file path
        """
        self.log("Exporting results...")
        exported_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Export dataset
        dataset_path = self.output_dir / f"synthetic_dataset_{timestamp}.json"
        self.generator.export_dataset(self.dataset, str(dataset_path))
        exported_files["dataset"] = str(dataset_path)

        # 2. Export metrics JSON for all methods
        metrics_data = {
            "experiment_info": {
                "timestamp": timestamp,
                "seed": self.seed,
                "num_chains": len(self.dataset),
                "total_perturbations": sum(len(c.perturbations) for c in self.dataset),
                "methods": list(all_metrics.keys()),
            },
            "metrics_by_method": {
                method: metrics.to_dict()
                for method, metrics in all_metrics.items()
            },
        }

        # Add comparisons to CAF
        if "CAF" in all_metrics:
            metrics_data["comparisons_to_caf"] = {}
            for method, metrics in all_metrics.items():
                if method != "CAF":
                    metrics_data["comparisons_to_caf"][method] = compute_baseline_comparison(
                        all_metrics["CAF"], metrics
                    )

        metrics_path = self.output_dir / f"experiment_metrics_{timestamp}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)
        exported_files["metrics"] = str(metrics_path)

        # 3. Export LaTeX results table (comparing all methods)
        latex_table = self._generate_latex_table(all_metrics)
        latex_path = self.output_dir / f"results_table_{timestamp}.tex"
        with open(latex_path, "w") as f:
            f.write(latex_table)
        exported_files["latex_table"] = str(latex_path)

        # 4. Export algorithm LaTeX
        algorithm_path = self.output_dir / f"algorithm_{timestamp}.tex"
        with open(algorithm_path, "w") as f:
            f.write(get_algorithm_latex())
        exported_files["algorithm_latex"] = str(algorithm_path)

        # 5. Export summary report
        report = self._generate_report(all_metrics)
        report_path = self.output_dir / f"experiment_report_{timestamp}.txt"
        with open(report_path, "w") as f:
            f.write(report)
        exported_files["report"] = str(report_path)

        self.log(f"Exported {len(exported_files)} files to {self.output_dir}")
        return exported_files

    def _generate_latex_table(self, all_metrics: Dict[str, ExperimentMetrics]) -> str:
        """Generate LaTeX table comparing all methods."""
        if "CAF" not in all_metrics:
            # Fallback if CAF metrics not available
            return "% No CAF metrics available for comparison\n"

        methods = list(all_metrics.keys())
        caf_metrics = all_metrics["CAF"]

        latex = r"""\begin{table}[t]
\centering
\caption{Comparison of CAF with baseline methods}
\label{tab:caf-comparison}
\begin{tabular}{l""" + "c" * len(methods) + r"""}
\toprule
\textbf{Metric} & """ + " & ".join([f"\\textbf{{{m}}}" for m in methods]) + r""" \\
\midrule
"""

        # Inference Depth
        depths = " & ".join([f"{all_metrics[m].mean_inference_depth:.2f}" for m in methods])
        latex += f"Inference Depth ($d$) & {depths} \\\\\n"

        # Contradiction Rate
        rates = " & ".join([f"{all_metrics[m].contradiction_rate_percent:.1f}\\%" for m in methods])
        latex += f"Contradiction Rate & {rates} \\\\\n"

        # Entailment Accuracy
        accs = " & ".join([f"{all_metrics[m].entailment_accuracy:.3f}" for m in methods])
        latex += f"Entailment Accuracy & {accs} \\\\\n"

        # Semantic Invariance
        invs = " & ".join([f"{all_metrics[m].semantic_invariance_mean:.3f}" for m in methods])
        latex += f"Semantic Invariance & {invs} \\\\\n"

        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        return latex

    def _generate_report(
        self,
        all_metrics: Dict[str, ExperimentMetrics]
    ) -> str:
        """Generate human-readable experiment report for all methods."""
        if not all_metrics:
            return "No metrics available for report generation."

        caf_metrics = all_metrics.get("CAF")
        methods = list(all_metrics.keys())

        report = f"""
{'='*70}
CAF MULTI-METHOD COMPARISON REPORT
{'='*70}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Seed: {self.seed}

DATASET SUMMARY
---------------
Total Chains: {len(self.dataset)}
Total Perturbations: {sum(len(c.perturbations) for c in self.dataset)}
Domains: {', '.join(set(c.domain for c in self.dataset))}
Depth Range: {min(c.depth for c in self.dataset)} - {max(c.depth for c in self.dataset)}
Chains with Contradictions: {sum(1 for c in self.dataset if c.injected_contradictions)}

{'='*70}
METHODS EVALUATED
{'='*70}
"""
        for i, method in enumerate(methods, 1):
            report += f"{i}. {method}\n"

        report += f"\n{'='*70}\n"
        report += "PRIMARY METRICS - ALL METHODS\n"
        report += f"{'='*70}\n\n"

        # Create comparison table
        col_width = 12
        header = f"{'Metric':<25}"
        for method in methods:
            header += f"| {method:>{col_width}} "
        report += header + "\n"
        report += "-" * len(header) + "\n"

        # Inference Depth
        row = f"{'Inference Depth':<25}"
        for method in methods:
            row += f"| {all_metrics[method].mean_inference_depth:>{col_width}.2f} "
        report += row + "\n"

        # Contradiction Rate
        row = f"{'Contradiction Rate (%)':<25}"
        for method in methods:
            row += f"| {all_metrics[method].contradiction_rate_percent:>{col_width}.1f} "
        report += row + "\n"

        # Entailment Accuracy
        row = f"{'Entailment Accuracy':<25}"
        for method in methods:
            row += f"| {all_metrics[method].entailment_accuracy:>{col_width}.4f} "
        report += row + "\n"

        # Semantic Invariance
        row = f"{'Semantic Invariance':<25}"
        for method in methods:
            row += f"| {all_metrics[method].semantic_invariance_mean:>{col_width}.4f} "
        report += row + "\n"

        # Improvements over baselines (if CAF exists)
        if caf_metrics and len(methods) > 1:
            report += f"\n{'='*70}\n"
            report += "CAF IMPROVEMENTS OVER BASELINES\n"
            report += f"{'='*70}\n\n"

            for method in methods:
                if method != "CAF":
                    comparison = compute_baseline_comparison(caf_metrics, all_metrics[method])
                    report += f"\nCAF vs {method}:\n"
                    report += f"  Inference Depth: +{comparison['inference_depth']['improvement_pct']:.1f}%\n"
                    report += f"  Contradiction Rate: {comparison['contradiction_rate']['delta']:.1f}pp reduction\n"
                    report += f"  Entailment Accuracy: +{comparison['entailment_accuracy']['improvement_pct']:.1f}%\n"
                    report += f"  Semantic Invariance: +{comparison['semantic_invariance']['improvement_pct']:.1f}%\n"

        # Statistical details for CAF
        if caf_metrics:
            report += f"\n{'='*70}\n"
            report += "CAF STATISTICAL DETAILS\n"
            report += f"{'='*70}\n\n"
            report += f"  - Inference Depth: {caf_metrics.mean_inference_depth:.2f} ± {caf_metrics.std_inference_depth:.2f}\n"
            report += f"  - 95% CI for Entailment: [{caf_metrics.confidence_interval_95[0]:.4f}, {caf_metrics.confidence_interval_95[1]:.4f}]\n"

            # Per-domain breakdown
            report += f"\n{'='*70}\n"
            report += "PER-DOMAIN BREAKDOWN (CAF)\n"
            report += f"{'='*70}\n"

            for domain, metrics in caf_metrics.metrics_by_domain.items():
                report += f"\n{domain.upper()}:\n"
                report += f"  Mean Depth: {metrics['mean_depth']:.2f}\n"
                report += f"  Contradiction Rate: {metrics['contradiction_rate']:.1f}%\n"
                report += f"  Entailment Accuracy: {metrics['entailment_accuracy']:.4f}\n"

        report += f"""
{'='*70}
ALGORITHM PSEUDOCODE
{'='*70}
{get_algorithm_pseudocode()}

{'='*70}
END OF REPORT
{'='*70}
"""
        return report

    def run_full_experiment(
        self,
        num_chains: int = 75,
        perturbations_per_chain: int = 3
    ) -> Dict[str, Any]:
        """
        Run the complete experiment pipeline.

        Args:
            num_chains: Number of chains (50-100)
            perturbations_per_chain: Perturbations per chain (2-3)

        Returns:
            Complete experiment results dictionary
        """
        start_time = time.time()
        self.log("=" * 60)
        self.log("STARTING CAF FULL EXPERIMENT")
        self.log("=" * 60)

        # Step 1: Generate dataset
        self.generate_dataset(num_chains, perturbations_per_chain)

        # Step 2: Run CAF and baseline evaluations
        self.run_caf_evaluation()

        # Step 3: Compute metrics for all methods
        all_metrics = self.compute_metrics()

        # Step 4: Export results
        exported_files = self.export_results(all_metrics)

        total_time = time.time() - start_time
        self.log("=" * 60)
        self.log(f"EXPERIMENT COMPLETE in {total_time:.1f}s")
        self.log("=" * 60)

        # Print summary
        print("\n" + "=" * 60)
        print("FINAL RESULTS SUMMARY")
        print("=" * 60)

        if "CAF" in all_metrics:
            caf_metrics = all_metrics["CAF"]
            print(f"\nCAF Results:")
            print(f"  Inference Depth: {caf_metrics.mean_inference_depth:.2f}")
            print(f"  Contradiction Rate: {caf_metrics.contradiction_rate_percent:.1f}%")
            print(f"  Entailment Accuracy: {caf_metrics.entailment_accuracy:.4f}")
            print(f"  Semantic Invariance: {caf_metrics.semantic_invariance_mean:.4f}")

            # Show improvements over each baseline
            print(f"\nCAF Improvements:")
            for method in all_metrics:
                if method != "CAF":
                    comparison = compute_baseline_comparison(caf_metrics, all_metrics[method])
                    print(f"  vs {method}:")
                    print(f"    Inference Depth: +{comparison['inference_depth']['improvement_pct']:.1f}%")
                    print(f"    Contradiction Rate: -{comparison['contradiction_rate']['delta']:.1f}pp")
                    print(f"    Entailment Accuracy: +{comparison['entailment_accuracy']['improvement_pct']:.1f}%")

        print(f"\nResults exported to: {self.output_dir}")
        print("=" * 60)

        return {
            "all_metrics": {method: metrics.to_dict() for method, metrics in all_metrics.items()},
            "exported_files": exported_files,
            "execution_time_seconds": total_time,
        }


def main():
    """Main entry point for experiment execution."""
    parser = argparse.ArgumentParser(
        description="Run CAF full experiment for paper"
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results",
        help="Directory for output files"
    )
    parser.add_argument(
        "--num-chains",
        type=int,
        default=75,
        help="Number of causal chains (50-100)"
    )
    parser.add_argument(
        "--perturbations",
        type=int,
        default=3,
        help="Perturbations per chain (2-3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )

    # LLM Configuration
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use real LLM instead of simulation (requires GPU)"
    )
    parser.add_argument(
        "--llm-model",
        default="7b",
        help="Model alias (7b, 8b, 13b, tiny, phi2, mistral, qwen7b, qwen14b, qwen3-14b) "
             "or a full HuggingFace model id such as Qwen/Qwen2.5-7B-Instruct"
    )
    parser.add_argument(
        "--llm-4bit",
        action="store_true",
        help="Use 4-bit quantization (saves memory)"
    )
    parser.add_argument(
        "--llm-8bit",
        action="store_true",
        help="Use 8-bit quantization"
    )

    # SPARQL/FVL Configuration (NEW)
    parser.add_argument(
        "--use-real-sparql",
        action="store_true",
        help="Use real SPARQL verification instead of simulation (requires running triplestore)"
    )
    parser.add_argument(
        "--sparql-endpoint",
        default="http://localhost:3030/conceptnet/query",
        help="SPARQL endpoint URL for verification"
    )
    parser.add_argument(
        "--entity-th",
        type=float,
        default=0.7,
        help="Fuzzy matching threshold for entity linking (0-1)"
    )
    parser.add_argument(
        "--disable-fuzzy-match",
        action="store_true",
        help="Disable fuzzy entity matching (exact matches only)"
    )

    args = parser.parse_args()

    # Validate parameters
    if not 50 <= args.num_chains <= 100:
        print(f"Warning: num_chains={args.num_chains} is outside recommended range [50, 100]")

    if not 2 <= args.perturbations <= 3:
        print(f"Warning: perturbations={args.perturbations} is outside recommended range [2, 3]")

    # Initialize inference layer
    inference_layer = None
    if args.use_llm:
        print("=" * 60)
        print("LOADING REAL LLM (This may take a few minutes...)")
        print("=" * 60)
        from common.llm_integration import create_causal_lm_layer

        inference_layer = create_causal_lm_layer(
            model_size=args.llm_model,
            use_4bit=args.llm_4bit,
            use_8bit=args.llm_8bit,
            open_source=True
        )
        print("LLM loaded successfully!")
    else:
        print("Using simulated inference layer (no real LLM)")

    # Initialize verification layer (NEW)
    verification_layer = None
    if args.use_real_sparql:
        print("=" * 60)
        print("INITIALIZING REAL SPARQL VERIFICATION")
        print("=" * 60)
        try:
            from experiments.knowledge_base_fvl import KnowledgeBaseFVL

            verification_layer = KnowledgeBaseFVL(
                sparql_endpoint=args.sparql_endpoint,
                entity_threshold=args.entity_threshold,
                enable_fuzzy_match=not args.disable_fuzzy_match
            )
            print(f"Knowledge Base FVL initialized with endpoint: {args.sparql_endpoint}")
            print(f"  Entity threshold: {args.entity_threshold}")
            print(f"  Fuzzy matching: {'disabled' if args.disable_fuzzy_match else 'enabled'}")

            # Test connection
            print("\nTesting SPARQL endpoint connection...")
            test_query = "ASK { ?s ?p ?o }"
            result = verification_layer._execute_sparql_query(test_query)
            if result.success:
                print("✓ SPARQL endpoint connection successful!")
            else:
                print(f"✗ SPARQL endpoint connection failed: {result.error}")
                print("  Make sure triplestore is running and endpoint is correct.")
                print("  See REAL_SPARQL_SETUP.md for setup instructions.")
                sys.exit(1)

        except ImportError as e:
            print(f"Error: Failed to import KnowledgeBaseFVL: {e}")
            print("Install dependencies: pip install SPARQLWrapper spacy fuzzywuzzy")
            print("Download spaCy model: python -m spacy download en_core_web_sm")
            sys.exit(1)
        except Exception as e:
            print(f"Error initializing Real FVL: {e}")
            sys.exit(1)
    else:
        print("Using simulated verification layer (no real SPARQL)")

    # Run experiment
    runner = ExperimentRunner(
        output_dir=args.output_dir,
        seed=args.seed,
        verbose=not args.quiet,
        inference_layer=inference_layer,
        verification_layer=verification_layer  # Pass Real FVL if enabled
    )

    results = runner.run_full_experiment(
        num_chains=args.num_chains,
        perturbations_per_chain=args.perturbations
    )

    return results


if __name__ == "__main__":
    main()
