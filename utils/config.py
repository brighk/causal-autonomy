"""
Configuration management using Pydantic Settings.
Loads configuration from environment variables and .env files.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings"""

    # Application
    app_name: str = Field(default="Causal Autonomy Framework", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: str = Field(default="development", description="Environment (development, production)")

    # API Gateway
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=4, description="Number of API workers")

    # Inference Engine
    inference_engine_url: str = Field(
        default="http://localhost:8001",
        description="Inference engine service URL"
    )
    model_name: str = Field(
        default="meta-llama/Llama-3-70b-chat-hf",
        description="LLM model name"
    )
    tensor_parallel_size: int = Field(default=1, description="Tensor parallelism degree")
    gpu_memory_utilization: float = Field(default=0.9, description="GPU memory utilization")
    use_vllm: bool = Field(default=True, description="Use vLLM for inference")

    # Apache Jena Fuseki
    fuseki_endpoint: str = Field(
        default="http://localhost:3030/dataset/query",
        description="Fuseki SPARQL endpoint"
    )
    fuseki_update_endpoint: str = Field(
        default="http://localhost:3030/dataset/update",
        description="Fuseki SPARQL update endpoint"
    )

    # ChromaDB
    chromadb_host: str = Field(default="localhost", description="ChromaDB host")
    chromadb_port: int = Field(default=8000, description="ChromaDB port")

    # Verification Settings
    verification_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for verification"
    )
    max_refinement_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum refinement iterations"
    )

    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    prometheus_port: int = Field(default=9090, description="Prometheus port")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        description="Log format"
    )

    # Security
    api_key: Optional[str] = Field(default=None, description="API authentication key")
    enable_cors: bool = Field(default=True, description="Enable CORS")

    # Data
    rdf_data_dir: str = Field(default="./data/rdf", description="RDF data directory")
    vector_data_dir: str = Field(default="./data/vectors", description="Vector data directory")

    # Performance
    request_timeout: int = Field(default=300, description="Request timeout in seconds")
    max_concurrent_requests: int = Field(default=10, description="Max concurrent requests")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
