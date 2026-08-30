"""
FastAPI Gateway for the Causal Autonomy Framework (CAF).
Orchestrates the complete pipeline: Inference → Parsing → Verification → Validation.
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from contextlib import asynccontextmanager
import sys
import time
from pathlib import Path
from typing import Dict, Any
from loguru import logger

# framework1/ (for modules.* and utils.*, both now live under here)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from .models import (
    CAFRequest, CAFResponse, HealthStatus,
    InferenceRequest, FinalResponse, VerificationResult
)
from modules.inference_engine.client import InferenceEngineClient
from modules.semantic_parser.parser import SemanticParser
from modules.truth_anchor.verifier import TruthAnchor
from modules.causal_validator.validator import CausalValidator
from utils.config import get_settings


# Global service instances
services: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup services"""
    logger.info("Initializing CAF services...")

    settings = get_settings()

    # Initialize modules with error handling
    try:
        services['inference'] = InferenceEngineClient(
            base_url=settings.inference_engine_url
        )
        logger.info("✓ Inference client initialized")
    except Exception as e:
        logger.warning(f"⚠ Inference client initialization failed: {e}")
        services['inference'] = None

    try:
        services['parser'] = SemanticParser(
            fuseki_endpoint=settings.fuseki_endpoint
        )
        logger.info("✓ Semantic parser initialized")
    except Exception as e:
        logger.warning(f"⚠ Semantic parser initialization failed: {e}")
        services['parser'] = None

    try:
        services['truth_anchor'] = TruthAnchor(
            fuseki_endpoint=settings.fuseki_endpoint
        )
        logger.info("✓ Truth anchor initialized")
    except Exception as e:
        logger.warning(f"⚠ Truth anchor initialization failed: {e}")
        services['truth_anchor'] = None

    try:
        services['causal_validator'] = CausalValidator()
        logger.info("✓ Causal validator initialized")
    except Exception as e:
        logger.warning(f"⚠ Causal validator initialization failed: {e}")
        services['causal_validator'] = None

    logger.info("Service initialization complete")
    yield

    # Cleanup
    logger.info("Shutting down CAF services...")
    services.clear()


app = FastAPI(
    title="Causal Autonomy Framework API",
    description="Sovereign Agent with Deterministic Output via Causal Grounding",
    version="1.0.0",
    lifespan=lifespan
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
        "inference_engine": await services['inference'].is_healthy() if services.get('inference') else False,
        "semantic_parser": services['parser'].is_healthy() if services.get('parser') else False,
        "truth_anchor": services['truth_anchor'].is_healthy() if services.get('truth_anchor') else False,
        "causal_validator": services['causal_validator'].is_healthy() if services.get('causal_validator') else False
    }

    all_healthy = all(components.values())

    return HealthStatus(
        status="healthy" if all_healthy else "degraded",
        components=components
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

    try:
        # Stage 1 & 2: Ingestion + Drafting
        logger.info(f"Processing inference request: {request.prompt[:50]}...")

        inference_req = InferenceRequest(
            prompt=request.prompt,
            session_id=request.session_id
        )
        response_candidate = await services['inference'].generate(inference_req)

        # Stage 3: Extraction
        logger.info("Extracting semantic triplets...")
        parsed_result = await services['parser'].parse(
            response_candidate.text,
            response_candidate.causal_assertions
        )

        # Recursive Refinement Loop
        refinement_count = 0
        is_verified = False
        verification_result = None

        while refinement_count < request.max_refinement_iterations and not is_verified:
            # Stage 4: Verification
            logger.info(f"Verification iteration {refinement_count + 1}")
            verification_result = await services['truth_anchor'].verify(
                triplets=parsed_result.triplets,
                threshold=request.verification_threshold
            )

            # Stage 5: Causal Validation (if enabled)
            if request.enable_causal_validation:
                causal_check = await services['causal_validator'].validate(
                    assertions=response_candidate.causal_assertions,
                    verified_triplets=verification_result.matched_triplets
                )

                if not causal_check.is_valid:
                    verification_result.is_valid = False
                    verification_result.contradictions.extend(
                        causal_check.violations
                    )

            if verification_result.is_valid:
                is_verified = True
                break

            # Re-run inference with negative feedback
            refinement_count += 1
            if refinement_count < request.max_refinement_iterations:
                logger.warning(
                    f"Verification failed. Re-running with constraints. "
                    f"Contradictions: {verification_result.contradictions}"
                )

                # Add constraints to prompt
                constrained_prompt = (
                    f"{request.prompt}\n\n"
                    f"CONSTRAINT: Avoid these contradictions: "
                    f"{', '.join(verification_result.contradictions)}"
                )

                inference_req.prompt = constrained_prompt
                response_candidate = await services['inference'].generate(
                    inference_req
                )
                parsed_result = await services['parser'].parse(
                    response_candidate.text,
                    response_candidate.causal_assertions
                )

        # Construct final response
        if not is_verified:
            logger.error(
                f"Failed to verify after {refinement_count} iterations"
            )
            raise HTTPException(
                status_code=422,
                detail=f"Unable to ground response in knowledge base after "
                       f"{refinement_count} refinement iterations. "
                       f"Contradictions: {verification_result.contradictions}"
            )

        final_response = FinalResponse(
            text=response_candidate.text,
            verification_status=verification_result,
            refinement_iterations=refinement_count,
            causal_grounding=verification_result.matched_triplets,
            confidence=verification_result.similarity_score or 1.0,
            session_id=request.session_id
        )

        processing_time = (time.time() - start_time) * 1000

        return CAFResponse(
            final_response=final_response,
            processing_time_ms=processing_time,
            pipeline_metadata={
                "verification_method": verification_result.verification_method,
                "refinement_iterations": refinement_count,
                "total_triplets_verified": len(verification_result.matched_triplets)
            }
        )

    except Exception as e:
        logger.exception(f"Pipeline error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Causal Autonomy Framework",
        "version": "1.0.0",
        "status": "operational"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
