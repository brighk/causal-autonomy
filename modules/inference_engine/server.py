"""
Inference Engine standalone server.
Can be deployed separately for GPU isolation.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import asyncio
import sys
from pathlib import Path
from loguru import logger

# framework1/, for utils.*
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from .engine import InferenceEngine, GenerationConfig
from utils.config import get_settings


app = FastAPI(title="CAF Inference Engine", version="1.0.0")

# Global engine instance
engine: Optional[InferenceEngine] = None


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    session_id: Optional[str] = None
    constraints: Optional[List[str]] = None


@app.on_event("startup")
async def startup():
    """Initialize the inference engine"""
    global engine

    settings = get_settings()

    logger.info("Starting Inference Engine server...")

    engine = InferenceEngine(
        model_name=settings.model_name,
        tensor_parallel_size=settings.tensor_parallel_size,
        gpu_memory_utilization=settings.gpu_memory_utilization,
        use_vllm=settings.use_vllm
    )

    logger.info("Inference Engine ready")


@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate response with causal assertions"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    config = GenerationConfig(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p
    )

    try:
        result = await engine.generate(
            prompt=request.prompt,
            config=config,
            constraints=request.constraints
        )
        return result
    except Exception as e:
        logger.exception(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    if engine is None or not engine.is_healthy():
        raise HTTPException(status_code=503, detail="Engine not healthy")

    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    """Readiness check for k8s"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    return {"status": "ready"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
