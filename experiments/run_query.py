#!/usr/bin/env python3
"""
Ad-hoc single-question CAF runner.

The other two entry points in this package (run_experiment.py,
run_counterbench_experiment.py) are full benchmark harnesses - dataset
generation, multiple baselines, metrics export. Neither is a fast way to ask
CAF one question and step through it in a debugger. This is that: load one
real local LLM, connect to one real Fuseki endpoint, run CAFLoop.execute()
on one prompt, print what happened at each iteration.

Usage:
    uv run python -m experiments.run_query --prompt "Does rain cause slippery roads?"

    # No LLM/KB needed - fully simulated, useful as a first smoke test:
    uv run python -m experiments.run_query --simulate --prompt "test"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.caf_algorithm import CAFConfig, CAFLoop


def main():
    parser = argparse.ArgumentParser(description="Run CAF on a single ad-hoc prompt")
    parser.add_argument("--prompt", required=True, help="Question to ask CAF")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use SimulatedInferenceLayer/SimulatedFVL - no LLM or Fuseki needed",
    )

    parser.add_argument(
        "--llm-model",
        default="Qwen/Qwen3-14B",
        help="HF model id, alias, or 'ollama:<tag>' (see common.llm_integration)",
    )
    parser.add_argument(
        "--llm-4bit",
        action="store_true",
        default=True,
        help="4-bit quantization (default on - needed for 14B on 24GB VRAM)",
    )
    parser.add_argument("--llm-no-4bit", action="store_false", dest="llm_4bit")

    parser.add_argument(
        "--sparql-endpoint",
        default="http://localhost:3030/dataset/query",
        help="Must match the dataset name in config/fuseki/assembler.ttl "
        "(that file provisions a dataset literally named 'dataset')",
    )
    parser.add_argument(
        "--use-intervention",
        action="store_true",
        default=True,
        help="Use KnowledgeBaseFVLWithIntervention (do-calculus for counterfactuals, "
        "falls back to plain SPARQL for factual questions)",
    )

    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--verification-threshold", type=float, default=0.8)

    args = parser.parse_args()

    if args.simulate:
        print("=== Simulated mode: no LLM, no Fuseki ===")
        inference_layer = None
        verification_layer = None
    else:
        print(f"=== Loading LLM: {args.llm_model} (4bit={args.llm_4bit}) ===")
        from common.llm_integration import create_causal_lm_layer

        inference_layer = create_causal_lm_layer(
            model_size=args.llm_model,
            use_4bit=args.llm_4bit,
        )
        # <- breakpoint here to inspect the loaded model/tokenizer before generation

        print(f"=== Connecting to Fuseki: {args.sparql_endpoint} ===")
        if args.use_intervention:
            from experiments.kb_fvl_with_intervention import (
                KnowledgeBaseFVLWithIntervention,
            )

            verification_layer = KnowledgeBaseFVLWithIntervention(
                sparql_endpoint=args.sparql_endpoint
            )
        else:
            from experiments.knowledge_base_fvl import KnowledgeBaseFVL

            verification_layer = KnowledgeBaseFVL(sparql_endpoint=args.sparql_endpoint)

    caf_loop = CAFLoop(
        config=CAFConfig(
            max_iterations=args.max_iterations,
            verification_threshold=args.verification_threshold,
        ),
        inference_layer=inference_layer,
        verification_layer=verification_layer,
    )

    print(f"\n=== Executing CAF loop on: {args.prompt!r} ===\n")
    output = caf_loop.execute(args.prompt)
    # <- breakpoint here to inspect `output.iteration_logs` iteration-by-iteration

    print(f"\nDecision:        {output.decision.value}")
    print(f"Final score:     {output.final_score:.3f}")
    print(f"Iterations used: {output.iterations_used}")
    print(f"Duration:        {output.total_duration_ms:.1f}ms")
    print(f"\nFinal response:\n{output.final_response}")

    for log in output.iteration_logs:
        print(f"\n--- Iteration {log.iteration} (score={log.overall_score:.3f}) ---")
        for t in log.extracted_triplets:
            print(f"  triplet: {t}")
        for r in log.verification_results:
            print(f"  verify:  {r.status.value} (kb_support={r.kb_support})")
        if log.injected_constraints:
            print(f"  constraints injected: {log.injected_constraints}")


if __name__ == "__main__":
    main()
