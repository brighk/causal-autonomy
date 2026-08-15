#!/usr/bin/env python3
"""
Run CAF on CounterBench Dataset
================================

Evaluate CAF's causal reasoning capabilities on the CounterBench benchmark.

CounterBench tests counterfactual reasoning on deterministic SCMs with varying
complexity levels (Basic, Conditional, Joint, Nested). This script adapts CAF
to process CounterBench queries and computes evaluation metrics.

Usage:
    # Load dataset and run evaluation
    python -m experiments.run_counterbench_experiment \\
        --input data/counterbench_caf.json \\
        --use-llm \\
        --llm-4bit \\
        --output results/counterbench

    # Quick test on 10 examples
    python -m experiments.run_counterbench_experiment \\
        --input data/counterbench_caf.json \\
        --limit 10 \\
        --output results/counterbench_test

    # Full evaluation with real SPARQL (if causal KB populated)
    python -m experiments.run_counterbench_experiment \\
        --input data/counterbench_caf.json \\
        --use-llm \\
        --llm-4bit \\
        --use-real-sparql \\
        --output results/counterbench_full
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.caf_algorithm import CAFLoop, CAFConfig, AdjudicationDecision
from common.llm_integration import create_causal_lm_layer
from experiments.caf_algorithm import SimulatedInferenceLayer, SimulatedFVL, IterationLog, VerificationResult, RDFTriplet
from enum import Enum


def make_json_serializable(obj: Any) -> Any:
    """Convert CAF objects to JSON-serializable format."""
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, (IterationLog, VerificationResult, RDFTriplet)):
        result = {}
        for key, value in obj.__dict__.items():
            result[key] = make_json_serializable(value)
        return result
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    else:
        return obj


@dataclass
class CounterBenchResult:
    """Result for single CounterBench example."""

    question_id: str
    query: str
    expected_answer: str
    caf_answer: str
    caf_decision: str  # ACCEPT, REJECT, UNCERTAIN
    caf_score: float
    iterations: int
    correct: bool
    reasoning_type: str

    # Detailed info
    response_text: str
    verification_details: Optional[Dict[str, Any]] = None


class CounterBenchEvaluator:
    """Evaluate CAF on CounterBench dataset."""

    def __init__(
        self,
        caf_loop: CAFLoop,
        use_llm: bool = False,
        use_sparql: bool = False
    ):
        """
        Initialize evaluator.

        Args:
            caf_loop: Configured CAF loop
            use_llm: Whether real LLM is being used
            use_sparql: Whether real SPARQL verification is being used
        """
        self.caf_loop = caf_loop
        self.use_llm = use_llm
        self.use_sparql = use_sparql
        self.results: List[CounterBenchResult] = []

    def extract_answer(self, response: str) -> str:
        """
        Extract yes/no answer from CAF response.

        Args:
            response: CAF's final response text

        Returns:
            'yes', 'no', or 'unknown' (lowercase to match CounterBench format)
        """
        response_lower = response.lower()

        # Check for uncertainty first (highest priority)
        if 'cannot determine' in response_lower or 'uncertain' in response_lower:
            return 'unknown'

        # Check for specific counterfactual patterns (more specific than yes/no alone)
        if 'would not occur' in response_lower or 'would not happen' in response_lower:
            return 'no'
        elif 'would occur' in response_lower or 'would happen' in response_lower:
            return 'yes'

        # Look for explicit yes/no (less specific, so checked last)
        # When both appear, prioritize context by looking at first occurrence
        yes_idx = response_lower.find('yes')
        no_idx = response_lower.find('no')

        if yes_idx != -1 and no_idx == -1:
            return 'yes'
        elif no_idx != -1 and yes_idx == -1:
            return 'no'
        elif yes_idx != -1 and no_idx != -1:
            # Both appear - use whichever comes first
            return 'yes' if yes_idx < no_idx else 'no'
        else:
            return 'unknown'

    def process_example(
        self,
        example: Dict[str, Any],
        verbose: bool = False
    ) -> CounterBenchResult:
        """
        Process single CounterBench example through CAF.

        Args:
            example: CounterBench example in CAF format
            verbose: Print detailed information

        Returns:
            CounterBenchResult
        """
        question_id = example['id']
        query = example['query']
        context = example['context']
        expected_answer = example['expected_answer']
        reasoning_type = example['metadata']['reasoning_type']

        # Combine context and query for CAF
        full_query = f"{context}\n\nQuestion: {query}"

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Processing: {question_id}")
            print(f"Type: {reasoning_type}")
            print(f"{'=' * 60}")
            print(f"Query: {full_query}")
            print(f"Expected: {expected_answer}")

        # Execute CAF loop
        try:
            caf_result = self.caf_loop.execute(full_query)

            # Extract answer from response
            caf_answer = self.extract_answer(caf_result.final_response)

            # Check if correct
            correct = (caf_answer == expected_answer)

            result = CounterBenchResult(
                question_id=question_id,
                query=query,
                expected_answer=expected_answer,
                caf_answer=caf_answer,
                caf_decision=caf_result.decision.value,
                caf_score=caf_result.final_score,
                iterations=caf_result.iterations_used,
                correct=correct,
                reasoning_type=reasoning_type,
                response_text=caf_result.final_response,
                verification_details={
                    'iteration_logs': [make_json_serializable(log) for log in caf_result.iteration_logs],
                    'iteration_count': caf_result.iterations_used
                }
            )

            if verbose:
                print(f"\nCAF Answer: {caf_answer}")
                print(f"CAF Decision: {caf_result.decision.value}")
                print(f"CAF Score: {caf_result.final_score:.2f}")
                print(f"Iterations: {caf_result.iterations_used}")
                print(f"Correct: {'✓' if correct else '✗'}")

        except Exception as e:
            if verbose:
                print(f"\n✗ Error processing example: {e}")

            result = CounterBenchResult(
                question_id=question_id,
                query=query,
                expected_answer=expected_answer,
                caf_answer='Unknown',
                caf_decision='ERROR',
                caf_score=0.0,
                iterations=0,
                correct=False,
                reasoning_type=reasoning_type,
                response_text=f"Error: {str(e)}"
            )

        self.results.append(result)
        return result

    def evaluate(
        self,
        examples: List[Dict[str, Any]],
        limit: Optional[int] = None,
        verbose: bool = False
    ) -> None:
        """
        Evaluate CAF on all examples.

        Args:
            examples: List of CounterBench examples in CAF format
            limit: Maximum number of examples to process
            verbose: Print detailed information
        """
        if limit:
            examples = examples[:limit]

        print(f"\nEvaluating CAF on {len(examples)} CounterBench examples...")
        print(f"Configuration: LLM={'Real' if self.use_llm else 'Simulated'}, "
              f"SPARQL={'Real' if self.use_sparql else 'Simulated'}")

        for i, example in enumerate(examples, 1):
            if not verbose:
                print(f"\rProgress: {i}/{len(examples)} ({i*100//len(examples)}%)", end='')

            self.process_example(example, verbose=verbose)

        if not verbose:
            print()  # New line after progress

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute evaluation metrics.

        Returns:
            Metrics dictionary
        """
        if not self.results:
            return {}

        total = len(self.results)
        correct = sum(1 for r in self.results if r.correct)
        accuracy = correct / total if total > 0 else 0.0

        # Metrics by reasoning type
        by_type = {}
        for result in self.results:
            rtype = result.reasoning_type
            if rtype not in by_type:
                by_type[rtype] = {'total': 0, 'correct': 0}

            by_type[rtype]['total'] += 1
            if result.correct:
                by_type[rtype]['correct'] += 1

        for rtype in by_type:
            total_type = by_type[rtype]['total']
            correct_type = by_type[rtype]['correct']
            by_type[rtype]['accuracy'] = correct_type / total_type if total_type > 0 else 0.0

        # Metrics by CAF decision
        by_decision = {}
        for result in self.results:
            dec = result.caf_decision
            if dec not in by_decision:
                by_decision[dec] = {'total': 0, 'correct': 0}

            by_decision[dec]['total'] += 1
            if result.correct:
                by_decision[dec]['correct'] += 1

        # Answer distribution
        answer_dist = {}
        for result in self.results:
            ans = result.caf_answer
            answer_dist[ans] = answer_dist.get(ans, 0) + 1

        # Average iterations and scores
        avg_iterations = sum(r.iterations for r in self.results) / total
        avg_score = sum(r.caf_score for r in self.results) / total

        return {
            'total_examples': total,
            'correct': correct,
            'accuracy': accuracy,
            'by_type': by_type,
            'by_decision': by_decision,
            'answer_distribution': answer_dist,
            'avg_iterations': avg_iterations,
            'avg_score': avg_score,
            'configuration': {
                'use_llm': self.use_llm,
                'use_sparql': self.use_sparql
            }
        }

    def save_results(self, output_dir: str) -> None:
        """
        Save results and metrics to files.

        Args:
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        results_file = output_path / 'results.json'
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        print(f"\n✓ Saved detailed results to {results_file}")

        # Compute and save metrics
        metrics = self.compute_metrics()
        metrics_file = output_path / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Saved metrics to {metrics_file}")

        # Save summary report
        report_file = output_path / 'report.txt'
        with open(report_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("CAF CounterBench Evaluation Report\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: LLM={'Real' if self.use_llm else 'Simulated'}, "
                   f"SPARQL={'Real' if self.use_sparql else 'Simulated'}\n\n")

            f.write("OVERALL PERFORMANCE\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total examples: {metrics['total_examples']}\n")
            f.write(f"Correct: {metrics['correct']}\n")
            f.write(f"Accuracy: {metrics['accuracy']:.2%}\n")
            f.write(f"Avg iterations: {metrics['avg_iterations']:.1f}\n")
            f.write(f"Avg score: {metrics['avg_score']:.2f}\n\n")

            f.write("PERFORMANCE BY REASONING TYPE\n")
            f.write("-" * 70 + "\n")
            for rtype, stats in sorted(metrics['by_type'].items()):
                f.write(f"{rtype:15} | {stats['correct']:3}/{stats['total']:3} | "
                       f"Accuracy: {stats['accuracy']:.2%}\n")

            f.write("\n")
            f.write("ANSWER DISTRIBUTION\n")
            f.write("-" * 70 + "\n")
            for answer, count in sorted(metrics['answer_distribution'].items()):
                pct = count / metrics['total_examples'] * 100
                f.write(f"{answer:10} | {count:4} ({pct:5.1f}%)\n")

        print(f"✓ Saved summary report to {report_file}")

    def print_summary(self) -> None:
        """Print evaluation summary to console."""
        metrics = self.compute_metrics()

        print("\n" + "=" * 70)
        print("COUNTERBENCH EVALUATION SUMMARY")
        print("=" * 70)

        print(f"\nOverall Accuracy: {metrics['accuracy']:.2%} "
              f"({metrics['correct']}/{metrics['total_examples']})")

        print("\nBy Reasoning Type:")
        for rtype, stats in sorted(metrics['by_type'].items()):
            print(f"  {rtype:15} | {stats['correct']:3}/{stats['total']:3} | "
                  f"{stats['accuracy']:.2%}")

        print(f"\nAvg Iterations: {metrics['avg_iterations']:.1f}")
        print(f"Avg Score: {metrics['avg_score']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run CAF on CounterBench dataset"
    )

    # Input/Output
    parser.add_argument(
        '--input',
        required=True,
        help='Input JSON file (from load_counterbench.py)'
    )
    parser.add_argument(
        '--output',
        default='results/counterbench',
        help='Output directory for results'
    )

    # CAF Configuration
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=5,
        help='Maximum CAF iterations'
    )
    parser.add_argument(
        '--verification-threshold',
        type=float,
        default=0.8,
        help='Verification threshold for ACCEPT decision'
    )

    # LLM Configuration
    parser.add_argument(
        '--use-llm',
        action='store_true',
        help='Use real LLM instead of simulation'
    )
    parser.add_argument(
        '--llm-model',
        default='7b',
        choices=['7b', '8b', '13b', 'tiny', 'phi2', 'mistral'],
        help='LLM model size/type (7b=Llama-2-7B [3.5GB], tiny=TinyLlama-1.1B [0.6GB], phi2=Phi-2-2.7B [1.5GB], mistral=Mistral-7B [3.5GB])'
    )
    parser.add_argument(
        '--llm-4bit',
        action='store_true',
        help='Use 4-bit quantization for LLM (recommended for 4GB GPU)'
    )

    # SPARQL Configuration
    parser.add_argument(
        '--use-real-sparql',
        action='store_true',
        help='Use real SPARQL verification'
    )
    parser.add_argument(
        '--sparql-endpoint',
        default='http://localhost:3030/conceptnet/query',
        help='SPARQL endpoint URL'
    )

    # Evaluation options
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of examples to evaluate'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information for each example'
    )

    # Knowledge base extraction
    parser.add_argument(
        '--extract-kb',
        action='store_true',
        help='Extract causal KB from dataset and load into SPARQL endpoint'
    )
    parser.add_argument(
        '--kb-file',
        default='data/counterbench_kb.nt',
        help='Path to save/load extracted KB (N-Triples format)'
    )
    parser.add_argument(
        '--fuseki-data-endpoint',
        help='Fuseki data endpoint for loading KB (default: derived from sparql-endpoint)'
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.input}...")
    with open(args.input) as f:
        examples = json.load(f)
    print(f"✓ Loaded {len(examples)} examples")

    # Extract and load knowledge base if requested
    if args.extract_kb and args.use_real_sparql:
        print("\n" + "=" * 70)
        print("EXTRACTING CAUSAL KNOWLEDGE BASE")
        print("=" * 70)

        # Import extractor
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from convert_counterbench_to_rdf import CounterBenchKBExtractor

        # Extract relations
        extractor = CounterBenchKBExtractor()
        extractor.extract_from_dataset(examples)

        # Save to file
        kb_file = Path(args.kb_file)
        extractor.save(str(kb_file))

        # Derive data endpoint from query endpoint
        if args.fuseki_data_endpoint:
            data_endpoint = args.fuseki_data_endpoint
        else:
            # Convert query endpoint to data endpoint
            # http://localhost:3030/counterbench/query -> http://localhost:3030/counterbench/data
            data_endpoint = args.sparql_endpoint.replace('/query', '/data')

        print(f"\nLoading KB into Fuseki ({data_endpoint})...")

        # Load into Fuseki
        import subprocess
        try:
            result = subprocess.run(
                [
                    'curl', '-X', 'POST',
                    '-H', 'Content-Type: application/n-triples',
                    '--data-binary', f'@{kb_file}',
                    data_endpoint
                ],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                print("✓ KB loaded successfully into SPARQL endpoint")
            else:
                print(f"⚠ Warning: Failed to load KB into Fuseki")
                print(f"  Error: {result.stderr}")
                print(f"\nManual loading:")
                print(f"  curl -X POST -H 'Content-Type: application/n-triples' \\")
                print(f"       --data-binary @{kb_file} \\")
                print(f"       {data_endpoint}")

        except subprocess.TimeoutExpired:
            print("⚠ Warning: Timeout loading KB into Fuseki")
            print(f"\nManual loading:")
            print(f"  curl -X POST -H 'Content-Type: application/n-triples' \\")
            print(f"       --data-binary @{kb_file} \\")
            print(f"       {data_endpoint}")
        except FileNotFoundError:
            print("⚠ Warning: 'curl' command not found")
            print(f"\nManual loading:")
            print(f"  curl -X POST -H 'Content-Type: application/n-triples' \\")
            print(f"       --data-binary @{kb_file} \\")
            print(f"       {data_endpoint}")

    # Initialize inference layer
    if args.use_llm:
        print(f"\nInitializing real LLM ({args.llm_model}, "
              f"{'4-bit' if args.llm_4bit else '8-bit'})...")
        from common.llm_integration import create_causal_lm_layer
        inference_layer = create_causal_lm_layer(
            model_size=args.llm_model,
            use_4bit=args.llm_4bit
        )
        print("✓ LLM loaded")
    else:
        print("\nUsing simulated inference layer")
        inference_layer = SimulatedInferenceLayer()

    # Initialize verification layer
    if args.use_real_sparql:
        print(f"\nInitializing real SPARQL verification ({args.sparql_endpoint})...")
        from experiments.knowledge_base_fvl import KnowledgeBaseFVL
        verification_layer = KnowledgeBaseFVL(sparql_endpoint=args.sparql_endpoint)
        print("✓ SPARQL endpoint ready")
    else:
        print("\nUsing simulated verification layer")
        # Simulated FVL already imported above
        verification_layer = SimulatedFVL()

    # Create CAF loop
    caf_config = CAFConfig(
        max_iterations=args.max_iterations,
        verification_threshold=args.verification_threshold
    )

    caf_loop = CAFLoop(
        config=caf_config,
        inference_layer=inference_layer,
        verification_layer=verification_layer
    )

    # Create evaluator
    evaluator = CounterBenchEvaluator(
        caf_loop=caf_loop,
        use_llm=args.use_llm,
        use_sparql=args.use_real_sparql
    )

    # Run evaluation
    evaluator.evaluate(examples, limit=args.limit, verbose=args.verbose)

    # Print summary
    evaluator.print_summary()

    # Save results
    evaluator.save_results(args.output)

    print(f"\n✓ Evaluation complete!")
    print(f"\nResults saved to: {args.output}/")

    return 0


if __name__ == '__main__':
    exit(main())
