"""
Batch-test caval against a list of prompts, catching failures per-prompt so
one bad result doesn't kill the whole run - useful for exploring pipeline
behavior across many claims without a script dying on the first rejection.

Prerequisites: same as examples/quickstart.py (Fuseki + the inference
engine must already be running).

Edit PROMPTS below to add your own cases. Run with:
    uv run python -m examples.batch_test
"""

from caval import Caval, VerificationFailedError

# Matched to the k8s/microservices causal graph currently loaded in Fuseki
# (see the real edges via the SPARQL query in the README/BACKLOG.md).
# Mix of: should-verify (real one-hop edges), should-reject (wrong/unknown
# entities), and known-tricky (multi-hop elaboration, "is a ..." asides).
PROMPTS = [
    "Does high cpu usage cause increased response time?",
    "Does garbage collection cause high cpu usage?",
    "Does response time cause a health check timeout?",
    "Does a health check timeout cause pod rotation?",
    "Does an in process lru cache cause high memory usage?",
    "Does missing connection pool limit cause database connections to rise?",
    # should reject - not a real edge in this KB
    "Does high cpu usage cause thermal throttling?",
    # unrelated entity, not in the KB at all
    "Does rain cause slippery roads?",
]


def main() -> None:
    with Caval() as caf:
        for prompt in PROMPTS:
            print("=" * 70)
            print("PROMPT:", prompt)
            try:
                result = caf.ask(prompt)
                status = (
                    "VERIFIED" if result.verification_status.is_valid else "REJECTED"
                )
                print(
                    f"RESULT: {status} (returned, {result.refinement_iterations} iterations)"
                )
                print("  answer:", result.text)
                print("  grounding:", result.causal_grounding)
            except VerificationFailedError as e:
                print(f"RESULT: FAILED after {e.refinement_iterations} iterations")
                for c in e.contradictions:
                    print("  -", c)
            except Exception as e:  # noqa: BLE001 - batch probe, log and continue
                print("RESULT: ERROR:", repr(e))
            print()


if __name__ == "__main__":
    main()
