"""
Quickstart: minimal end-to-end `caval` usage (sync).

Before running, start the two background services this depends on:

  1. Fuseki (knowledge base):
       docker compose -f deployment/docker-compose.yml up -d

  2. The inference engine (LLM), in a separate terminal:
       uv run python -m modules.inference_engine.server

Both must be reachable at the URLs in your .env (or the defaults: Fuseki at
http://localhost:3030/dataset/query, inference engine at
http://localhost:8001).

Run with:
    uv run python -m examples.quickstart
"""

from caval import Caval, VerificationFailedError


def main() -> None:
    with Caval() as caf:
        prompt = "Does high cpu usage cause increased response time?"
        try:
            result = caf.ask(prompt)
        except VerificationFailedError as e:
            print(f"Could not verify: {e}")
            return

        print("Answer:", result.text)
        print("Verified:", result.verification_status.is_valid)
        print("Refinement iterations:", result.refinement_iterations)
        print("Causal grounding:", result.causal_grounding)


if __name__ == "__main__":
    main()
