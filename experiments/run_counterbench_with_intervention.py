#!/usr/bin/env python3
"""
Run CounterBench with Intervention Calculus
============================================

Enhanced CAF evaluation using intervention calculus for counterfactual reasoning.

This script combines:
- LLM generation (TinyLlama, Phi-2, etc.)
- Intervention calculus (do-calculus for counterfactuals)
- SPARQL verification (for factual queries)

Usage:
    python -m experiments.run_counterbench_with_intervention \
        --input data/counterbench.json \
        --limit 10 \
        --use-llm --llm-model tiny --llm-4bit \
        --use-intervention \
        --output results/caf_intervention
"""

import re
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.caf_algorithm import CAFLoop, CAFConfig
from common.llm_integration import create_causal_lm_layer
from experiments.kb_fvl_with_intervention import KnowledgeBaseFVLWithIntervention
from experiments.run_counterbench_experiment import (
    CounterBenchEvaluator,
    CounterBenchResult
)


class InterventionCAFEvaluator(CounterBenchEvaluator):
    """
    CAF evaluator with intervention calculus support.

    Extends CounterBenchEvaluator to:
    1. Set causal context for each example
    2. Set current query for counterfactual detection
    3. Use intervention-enhanced FVL
    """

    def __init__(self, *args, fvl: KnowledgeBaseFVLWithIntervention = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fvl = fvl

    def process_example(
        self,
        example: Dict[str, Any],
        verbose: bool = False
    ) -> CounterBenchResult:
        """
        Process example with intervention calculus.

        Enhanced flow:
        1. Set causal context in FVL
        2. Set current query for counterfactual detection
        3. Run CAF loop (with intervention-aware verification)
        4. Extract answer
        """
        query = example['query']
        context = example.get('context', '')
        expected_answer = example['expected_answer']
        question_id = example.get('id', 'unknown')
        reasoning_type = example.get('metadata', {}).get('type', 'basic')
        if context:
            full_query = (
                f"{context}\n\n"
                f"Question: {query}\n"
                "Answer with exactly one word: yes or no."
            )
        else:
            full_query = f"{query}\nAnswer with exactly one word: yes or no."

        if verbose:
            print(f"\n{'='*70}")
            print(f"Processing: {question_id}")
            print(f"Query: {query}")
            print(f"Context: {context[:100]}...")
            print(f"Expected: {expected_answer}")

        try:
            # Set causal context and query for intervention calculus
            if self.fvl:
                self.fvl.set_causal_context(context)
                self.fvl.set_current_query(query)

            # Run CAF loop
            caf_result = self.caf_loop.execute(full_query, context)

            # Extract answer
            caf_answer = self.extract_answer(caf_result.final_response)
            correct = (caf_answer == expected_answer)

            if verbose and self.fvl:
                print("\nIntervention Calculus Explanation:")
                print(self.fvl.get_explanation())

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
                    'iteration_logs': [str(log) for log in caf_result.iteration_logs],
                    'iteration_count': caf_result.iterations_used,
                    'used_intervention': self.fvl._is_counterfactual_query() if self.fvl else False
                }
            )

            if verbose:
                print(f"\nCAF Answer: {caf_answer}")
                print(f"Correct: {'✓' if correct else '✗'}")

        except Exception as e:
            if verbose:
                print(f"\n✗ Error: {e}")

            result = CounterBenchResult(
                question_id=question_id,
                query=query,
                expected_answer=expected_answer,
                caf_answer='unknown',
                caf_decision='ERROR',
                caf_score=0.0,
                iterations=0,
                correct=False,
                reasoning_type=reasoning_type,
                response_text=f"Error: {str(e)}"
            )

        self.results.append(result)
        return result

    def extract_answer(self, response: str) -> str:
        """
        Strict answer extraction for CounterBench.

        Uses only the first non-empty line and first yes/no token
        to avoid being misled by trailing chat artifacts.
        """
        if not response:
            return "unknown"

        first_line = ""
        for line in response.splitlines():
            line = line.strip()
            if line:
                first_line = line.lower()
                break

        if not first_line:
            return "unknown"

        # Prioritize explicit uncertainty marker on first line.
        if "cannot determine" in first_line or "uncertain" in first_line:
            return "unknown"

        match = re.search(r"\b(yes|no)\b", first_line)
        if match:
            return match.group(1)

        # Fallback to the base extractor for non-strict legacy outputs.
        return super().extract_answer(response)


def main():
    parser = argparse.ArgumentParser(description='CAF with Intervention Calculus')
    parser.add_argument('--input', required=True, help='Input JSON file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--limit', type=int, help='Limit number of examples')

    # LLM options
    parser.add_argument('--use-llm', action='store_true', help='Use real LLM')
    parser.add_argument('--llm-model', default='tiny', choices=['tiny', 'phi2', '7b'])
    parser.add_argument('--llm-4bit', action='store_true', help='Use 4-bit quantization')

    # Intervention calculus
    parser.add_argument('--use-intervention', action='store_true', help='Use intervention calculus')

    # SPARQL (for factual queries fallback)
    parser.add_argument('--sparql-endpoint', default='http://localhost:3030/counterbench/query')

    args = parser.parse_args()

    print("="*70)
    print("CAF with Intervention Calculus - CounterBench Evaluation")
    print("="*70)
    print()

    # Load data
    with open(args.input) as f:
        data = json.load(f)

    if args.limit:
        data = data[:args.limit]

    print(f"✓ Loaded {len(data)} examples")

    # Create LLM
    if args.use_llm:
        from common.llm_integration import create_causal_lm_layer
        llm = create_causal_lm_layer(args.llm_model, use_4bit=args.llm_4bit)
        # Strict benchmark decoding: short, deterministic yes/no outputs.
        if hasattr(llm, "config"):
            llm.config.max_new_tokens = 8
            llm.config.do_sample = False
            llm.config.temperature = 0.0
            llm.config.top_p = 1.0
        print(f"✓ LLM loaded: {args.llm_model}")
    else:
        from experiments.caf_algorithm import SimulatedInferenceLayer
        llm = SimulatedInferenceLayer()
        print("✓ Using simulated LLM")

    # Create FVL with intervention calculus
    if args.use_intervention:
        fvl = KnowledgeBaseFVLWithIntervention(sparql_endpoint=args.sparql_endpoint)
        print(f"✓ Intervention calculus enabled")
    else:
        from experiments.caf_algorithm import SimulatedFVL
        fvl = SimulatedFVL()
        print("✓ Using simulated FVL")

    # Create CAF loop
    config = CAFConfig(max_iterations=3, verification_threshold=0.7)
    caf_loop = CAFLoop(
        config=config,
        inference_layer=llm,
        verification_layer=fvl
    )

    # Create evaluator
    evaluator = InterventionCAFEvaluator(
        caf_loop=caf_loop,
        use_llm=args.use_llm,
        use_sparql=args.use_intervention,
        fvl=fvl if args.use_intervention else None
    )

    # Evaluate
    print()
    print(f"Evaluating on {len(data)} examples...")
    print(f"Configuration: LLM={'Real' if args.use_llm else 'Simulated'}, Intervention={'Yes' if args.use_intervention else 'No'}")
    print()

    evaluator.evaluate(data, limit=args.limit, verbose=False)

    # Save results
    output_dir = Path(args.output)
    evaluator.save_results(output_dir)

    # Print summary
    metrics = evaluator.compute_metrics()

    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    total_examples = metrics.get('total_examples', len(evaluator.results))
    print(f"Accuracy: {metrics['accuracy']*100:.2f}% ({metrics['correct']}/{total_examples})")
    print(f"Avg Iterations: {metrics['avg_iterations']:.1f}")
    print(f"Avg Score: {metrics['avg_score']:.2f}")
    print()
    print(f"✓ Results saved to {output_dir}/")


if __name__ == '__main__':
    main()
