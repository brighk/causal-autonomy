"""
FastAPI Gateway for the Causal Autonomy Framework (CAF).
Orchestrates the complete pipeline: Inference → Parsing → Verification → Validation.
"""

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

from caval.exceptions import VerificationFailedError
from caval.pipeline import CAFPipeline
from modules.causal_validator.validator import CausalValidator
from modules.inference_engine.client import InferenceEngineClient
from modules.semantic_parser.parser import SemanticParser
from modules.truth_anchor.verifier import TruthAnchor
from utils.config import get_settings

from .models import CAFRequest, CAFResponse, HealthStatus

# Global service instances
services: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup services"""
    logger.info("Initializing CAF services...")

    settings = get_settings()

    # Initialize modules with error handling
    try:
        services["inference"] = InferenceEngineClient(
            base_url=settings.inference_engine_url
        )
        logger.info("✓ Inference client initialized")
    except Exception as e:
        logger.warning(f"⚠ Inference client initialization failed: {e}")
        services["inference"] = None

    try:
        services["parser"] = SemanticParser(fuseki_endpoint=settings.fuseki_endpoint)
        logger.info("✓ Semantic parser initialized")
    except Exception as e:
        logger.warning(f"⚠ Semantic parser initialization failed: {e}")
        services["parser"] = None

    try:
        services["truth_anchor"] = TruthAnchor(fuseki_endpoint=settings.fuseki_endpoint)
        logger.info("✓ Truth anchor initialized")
    except Exception as e:
        logger.warning(f"⚠ Truth anchor initialization failed: {e}")
        services["truth_anchor"] = None

    try:
        services["causal_validator"] = CausalValidator()
        logger.info("✓ Causal validator initialized")
    except Exception as e:
        logger.warning(f"⚠ Causal validator initialization failed: {e}")
        services["causal_validator"] = None

    logger.info("Service initialization complete")
    yield

    # Cleanup
    logger.info("Shutting down CAF services...")
    services.clear()


app = FastAPI(
    title="Causal Autonomy Framework API",
    description="Sovereign Agent with Deterministic Output via Causal Grounding",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus instrumentation
Instrumentator().instrument(app).expose(app)


@app.get("/health", response_model=HealthStatus)
async def health_check():
    """System health check endpoint"""
    components = {
        "inference_engine": await services["inference"].is_healthy()
        if services.get("inference")
        else False,
        "semantic_parser": services["parser"].is_healthy()
        if services.get("parser")
        else False,
        "truth_anchor": services["truth_anchor"].is_healthy()
        if services.get("truth_anchor")
        else False,
        "causal_validator": services["causal_validator"].is_healthy()
        if services.get("causal_validator")
        else False,
    }

    all_healthy = all(components.values())

    return HealthStatus(
        status="healthy" if all_healthy else "degraded", components=components
    )


@app.post("/v1/infer", response_model=CAFResponse)
async def causal_inference(request: CAFRequest):
    """
    Main CAF pipeline endpoint.

    Pipeline stages:
    1. Ingestion: Receive natural language prompt P
    2. Drafting: LLM generates response candidate R_c and causal assertion A_c
    3. Extraction: Middleware extracts triplets (s, p, o) from A_c
    4. Verification: SPARQL query to Apache Jena against ground truth
    5. Reification: Output R_f if valid, else re-run with constraints
    """
    start_time = time.time()

    logger.info(f"Processing inference request: {request.prompt[:50]}...")

    pipeline = CAFPipeline(
        inference=services["inference"],
        parser=services["parser"],
        truth_anchor=services["truth_anchor"],
        causal_validator=services["causal_validator"],
    )

    try:
        final_response = await pipeline.run(
            request.prompt,
            session_id=request.session_id,
            max_refinement_iterations=request.max_refinement_iterations,
            verification_threshold=request.verification_threshold,
            enable_causal_validation=request.enable_causal_validation,
        )
    except VerificationFailedError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception(f"Pipeline error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    processing_time = (time.time() - start_time) * 1000

    return CAFResponse(
        final_response=final_response,
        processing_time_ms=processing_time,
        pipeline_metadata={
            "verification_method": final_response.verification_status.verification_method,
            "refinement_iterations": final_response.refinement_iterations,
            "total_triplets_verified": len(
                final_response.verification_status.matched_triplets
            ),
        },
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Causal Autonomy Framework",
        "version": "1.0.0",
        "status": "operational",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
