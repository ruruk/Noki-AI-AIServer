"""
Configuration module for Noki AI Engine
"""
import os
from typing import List, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # FastAPI Configuration
    app_name: str = "Noki AI Engine"
    app_version: str = "1.0.0"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 2000
    
    # Pinecone Vector Database Configuration
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: str = "noki-ai-rd41mlf"
    pinecone_dimension: int = 1536  # OpenAI embedding dimension
    
    # Supabase Configuration (for vector storage)
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    supabase_vector_table: str = "ai_embeddings"
    
    # LangChain Configuration
    langchain_api_key: Optional[str] = None
    langchain_tracing_v2: bool = True
    langchain_project: str = "noki-ai-engine"
    
    # RAG Configuration
    retrieval_top_k: int = 6
    max_chat_history: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 100  # Reduced from 200 for better performance
    
    # Embedding Optimization
    embedding_batch_size: int = 10  # Process embeddings in batches
    embedding_cache_ttl: int = 3600  # Cache embeddings for 1 hour
    max_concurrent_embeddings: int = 5  # Limit concurrent embedding operations
    
    # Backend API Configuration
    backend_url: Optional[str] = os.getenv("BACKEND_URL", "http://localhost:3000")
    
    # Security
    secret_key: str = "your-secret-key-change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    backend_service_token: Optional[str] = None
    bearer_token: Optional[str] = None
    
    # CORS Configuration
    allowed_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    
    # Rate Limiting
    rate_limit_per_user: int = 100  # requests per hour
    rate_limit_per_conversation: int = 50  # requests per hour
    
    # Logging
    log_level: str = "INFO"
    
    # Metrics
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
