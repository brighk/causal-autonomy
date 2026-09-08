"""
Quickstart: minimal end-to-end `caval` usage (async).

Same prerequisites as examples/quickstart.py (Fuseki + the inference engine
must already be running). Use .aask() instead of .ask() when embedding
caval in an already-async app - e.g. a FastAPI route - since .ask() can't
be called from inside a running event loop.

Run with:
    uv run python -m examples.quickstart_async
"""

import asyncio

from caval import Caval, VerificationFailedError


async def main() -> None:
    async with Caval() as caf:
        prompt = "Does high cpu usage cause increased response time?"
        try:
            result = await caf.aask(prompt)
        except VerificationFailedError as e:
            print(f"Could not verify: {e}")
            return

        print("Answer:", result.text)
        print("Verified:", result.verification_status.is_valid)


if __name__ == "__main__":
    asyncio.run(main())
