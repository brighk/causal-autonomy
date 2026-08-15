"""
Pydantic models for strict type validation of CAF data packets.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime


class Triplet(BaseModel):
    """RDF Triplet (subject, predicate, object)"""
    subject: str = Field(..., description="Subject URI or literal")
    predicate: str = Field(..., description="Predicate URI")
    object_: str = Field(..., alias="object", description="Object URI or literal")

    class Config:
        populate_by_name = True


class CausalAssertion(BaseModel):
    """Causal assertion extracted from LLM response"""
    assertion_text: str = Field(..., description="Natural language assertion")
    triplets: List[Triplet] = Field(default_factory=list, description="Extracted RDF triplets")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")


class InferenceRequest(BaseModel):
    """Request payload for LLM inference"""
    prompt: str = Field(..., min_length=1, description="Natural language prompt")
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    session_id: Optional[str] = Field(default=None, description="Session tracking ID")


class ResponseCandidate(BaseModel):
    """Initial response from LLM before verification"""
    text: str = Field(..., description="Generated response text")
    causal_assertions: List[CausalAssertion] = Field(default_factory=list)
    generation_metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class VerificationResult(BaseModel):
    """Result from truth anchor verification"""
    is_valid: bool = Field(..., description="Whether assertion is grounded in KB")
    matched_triplets: List[Triplet] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    similarity_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    verification_method: str = Field(default="exact_match")


class FinalResponse(BaseModel):
    """Verified and refined final response"""
    text: str = Field(..., description="Final verified response")
    verification_status: VerificationResult
    refinement_iterations: int = Field(ge=0, description="Number of refinement loops")
    causal_grounding: List[Triplet] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    session_id: Optional[str] = None


class CAFRequest(BaseModel):
    """Complete CAF pipeline request"""
    prompt: str = Field(..., min_length=1)
    max_refinement_iterations: int = Field(default=3, ge=1, le=10)
    verification_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    enable_causal_validation: bool = Field(default=True)
    session_id: Optional[str] = None


class CAFResponse(BaseModel):
    """Complete CAF pipeline response"""
    final_response: FinalResponse
    processing_time_ms: float
    pipeline_metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthStatus(BaseModel):
    """System health check response"""
    status: str = Field(..., description="overall, inference, knowledge_graph, vector_db")
    components: Dict[str, bool] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0.0")
