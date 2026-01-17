"""
Pydantic models for Noki AI Engine API schemas
"""
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum


class Stage(str, Enum):
    """AI processing stages"""
    THINKING = "thinking"
    INTENT = "intent"
    RESPONSE = "response"
    COMPLETE = "complete"


class IntentType(str, Enum):
    """Types of AI intents"""
    BACKEND_QUERY = "backend_query"
    PROPOSED_SCHEDULE = "proposed_schedule"
    PROPOSED_TASKS = "proposed_tasks"


class TaskStatus(str, Enum):
    """Task status options"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"


class Project(BaseModel):
    """Project model"""
    project_id: str
    title: str
    description: Optional[str] = None
    instructor: Optional[str] = None


class Task(BaseModel):
    """Task model"""
    task_id: str
    title: str
    description: Optional[str] = None
    due_datetime: Optional[datetime] = None
    status: Optional[TaskStatus] = None
    project_id: Optional[str] = None


class ChatInput(BaseModel):
    """Input model for chat requests"""
    user_id: str
    conversation_id: str
    prompt: str
    projects: Optional[List[Project]] = None
    tasks: Optional[List[Task]] = None
    todos: Optional[List[Any]] = None  # Todo items
    stage: Stage = Stage.THINKING
    metadata: Optional[Dict[str, Any]] = None
    conversation_history: Optional[List[Dict[str, Any]]] = None
    auth_token: Optional[str] = None  # JWT token for backend calls


class AIIntent(BaseModel):
    """AI intent model"""
    type: IntentType
    targets: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    payload: Optional[Dict[str, Any]] = None


class TokenUsage(BaseModel):
    """Token usage information"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    embedding_tokens: int = 0  # For embedding operations
    cost_estimate_usd: float = 0.0  # Estimated cost in USD


class AIResponse(BaseModel):
    """AI response model"""
    stage: Stage
    conversation_id: str
    text: Optional[str] = None
    blocks: Optional[List[Dict[str, Any]]] = None
    intent: Optional[AIIntent] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    token_usage: Optional[TokenUsage] = None


class ContextInput(BaseModel):
    """Input model for context requests"""
    conversation_id: str
    user_id: str
    context_data: Dict[str, Any]
    stage: Stage = Stage.RESPONSE


class EmbedResourceInput(BaseModel):
    """Input model for embedding resources"""
    user_id: str
    conversation_id: str
    resource_id: str
    resource_type: str  # "PDF" | "Website" | "YouTube"
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class EmbedMessageInput(BaseModel):
    """Input model for embedding messages"""
    user_id: str
    conversation_id: str
    message_id: str
    message_content: str
    metadata: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str


class MetricsResponse(BaseModel):
    """Metrics response"""
    requests_total: int
    errors_total: int
    avg_latency_ms: float
    stage_distribution: Dict[str, int]
    intent_frequency: Dict[str, int]
    token_usage: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
