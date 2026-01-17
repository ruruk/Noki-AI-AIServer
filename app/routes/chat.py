"""
Chat routes for the main AI interaction endpoints
"""
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.models.schemas import ChatInput, AIResponse, ContextInput
from app.services.llm import LLMService
from app.services.vector import VectorService
from app.services.planner import PlannerService
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()


def get_vector_service() -> VectorService:
    """Dependency to get vector service instance"""
    return VectorService()


def get_llm_service(vector_service: VectorService = Depends(get_vector_service)) -> LLMService:
    """Dependency to get LLM service instance"""
    return LLMService(vector_service)


def get_planner_service() -> PlannerService:
    """Dependency to get planner service instance"""
    return PlannerService()


def verify_backend_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify backend service token or bearer token"""
    # Check if we have either token configured
    if not settings.backend_service_token and not settings.bearer_token:
        return "no-auth"  # Development mode
    
    # Check backend_service_token first (legacy)
    if settings.backend_service_token and credentials.credentials == settings.backend_service_token:
        return credentials.credentials
    
    # Check bearer_token as fallback
    if settings.bearer_token and credentials.credentials == settings.bearer_token:
        return credentials.credentials
    
    raise HTTPException(status_code=401, detail="Invalid service token")
    
    return credentials.credentials


@router.post("/chat", response_model=AIResponse)
async def chat(
    chat_input: ChatInput,
    llm_service: LLMService = Depends(get_llm_service),
    planner_service: PlannerService = Depends(get_planner_service)
) -> AIResponse:
    """
    Main chat endpoint - processes user messages and returns structured AI responses
    
    This endpoint handles the complete chat flow:
    1. Processes user input
    2. Retrieves semantic context
    3. Determines if backend data is needed (intent)
    4. Generates structured response with UI blocks
    5. Saves message to vector store
    """
    try:
        logger.info(f"Processing chat request for user {chat_input.user_id}, conversation {chat_input.conversation_id}")
        
        # Process the chat request (now async)
        response = await llm_service.process_chat_request(chat_input)
        
        # If we have an intent, enhance the response with planner service
        if response.intent and response.stage == "intent":
            # The response is already properly formatted for intent stage
            pass
        elif response.blocks:
            # Enhance blocks using planner service if needed
            enhanced_blocks = []
            for block in response.blocks:
                if block.get("type") == "todo_list" and block.get("accept_decline"):
                    # This is a proposal that needs special handling
                    enhanced_blocks.append(block)
                else:
                    enhanced_blocks.append(block)
            response.blocks = enhanced_blocks
        
        logger.info(f"Chat response generated with stage: {response.stage}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error processing chat request"
        )


@router.post("/chat/context", response_model=AIResponse)
async def chat_with_context(
    context_input: ContextInput,
    llm_service: LLMService = Depends(get_llm_service),
    planner_service: PlannerService = Depends(get_planner_service)
) -> AIResponse:
    """
    Continue chat processing with backend-provided context data
    
    This endpoint is called by the backend after fulfilling an AI intent.
    The AI can then process the context data and generate a complete response.
    """
    try:
        logger.info(f"Processing context for conversation {context_input.conversation_id}")
        
        # Continue processing with the provided context
        response = llm_service.continue_with_context(
            conversation_id=context_input.conversation_id,
            user_id=context_input.user_id,
            context_data=context_input.context_data
        )
        
        # Enhance response with planner service based on context data
        if context_input.context_data:
            # Create a mock intent for the planner service
            from app.models.schemas import AIIntent, IntentType
            mock_intent = AIIntent(
                type=IntentType.BACKEND_QUERY,
                targets=["assignments", "schedule"],
                filters={},
                payload={}
            )
            
            enhanced_blocks = planner_service.create_intent_response(
                intent=mock_intent,
                context_data=context_input.context_data
            )
            
            if enhanced_blocks:
                response.blocks = enhanced_blocks
        
        logger.info(f"Context response generated with stage: {response.stage}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing context request: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error processing context request"
        )


@router.post("/chat/stream")
async def chat_stream(
    chat_input: ChatInput,
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Streaming chat endpoint for real-time responses
    
    This endpoint provides streaming responses for better UX.
    It emits multiple response chunks before completing.
    """
    try:
        from fastapi.responses import StreamingResponse
        import json
        
        async def generate_stream():
            # Emit thinking stage
            yield f"data: {json.dumps({'stage': 'thinking', 'conversation_id': chat_input.conversation_id, 'text': 'Processing your request...'})}\n\n"
            
            # Process the request
            response = llm_service.process_chat_request(chat_input)
            
            # Emit intent if present
            if response.intent:
                yield f"data: {json.dumps(response.dict())}\n\n"
                return
            
            # Emit response chunks
            if response.blocks:
                for i, block in enumerate(response.blocks):
                    chunk_response = AIResponse(
                        stage="response",
                        conversation_id=response.conversation_id,
                        text=f"Generating response part {i+1} of {len(response.blocks)}...",
                        blocks=[block]
                    )
                    yield f"data: {json.dumps(chunk_response.dict())}\n\n"
            
            # Emit final complete response
            final_response = AIResponse(
                stage="complete",
                conversation_id=response.conversation_id,
                text=response.text,
                blocks=response.blocks
            )
            yield f"data: {json.dumps(final_response.dict())}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        logger.error(f"Error in streaming chat: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error in streaming chat"
        )


@router.get("/chat/history/{conversation_id}")
async def get_chat_history(
    conversation_id: str,
    user_id: str,
    vector_service: VectorService = Depends(get_vector_service)
) -> Dict[str, Any]:
    """
    Get chat history for a conversation
    
    Returns recent messages and context for a conversation.
    """
    try:
        # Get recent chat history
        history = vector_service.get_recent_chat_history(
            user_id=user_id,
            conversation_id=conversation_id,
            limit=20  # Get more history for this endpoint
        )
        
        # Format history for response
        formatted_history = []
        for doc in history:
            formatted_history.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "timestamp": doc.metadata.get("created_at")
            })
        
        return {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "history": formatted_history,
            "count": len(formatted_history)
        }
        
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error getting chat history"
        )
