"""
Client for communicating with the Inference Engine service.
Uses gRPC for high-throughput, low-latency communication.
"""
from typing import Dict, Any, Optional
import httpx
from loguru import logger

from api.models import InferenceRequest, ResponseCandidate, CausalAssertion


class InferenceEngineClient:
    """Client for Inference Engine Module A"""

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 min timeout for large models

    async def generate(self, request: InferenceRequest) -> ResponseCandidate:
        """
        Send inference request to the engine.

        Args:
            request: InferenceRequest with prompt and generation params

        Returns:
            ResponseCandidate with generated text and causal assertions
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/generate",
                json=request.model_dump()
            )
            response.raise_for_status()

            data = response.json()

            # Convert raw assertions to CausalAssertion objects
            causal_assertions = [
                CausalAssertion(
                    assertion_text=assertion,
                    triplets=[],  # Will be populated by semantic parser
                    confidence=0.9  # Default confidence
                )
                for assertion in data.get('causal_assertions_raw', [])
            ]

            return ResponseCandidate(
                text=data['text'],
                causal_assertions=causal_assertions,
                generation_metadata=data.get('metadata', {})
            )

        except httpx.HTTPError as e:
            logger.error(f"Inference engine request failed: {e}")
            raise

    async def is_healthy(self) -> bool:
        """Check if inference engine is healthy"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
