"""
LLM service with LangChain integration for prompt pipeline and RAG
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

from config import settings
from app.services.token_usage import TokenUsageService
from app.models.schemas import ChatInput, AIResponse, AIIntent, IntentType, Stage, TokenUsage
from app.services.vector import VectorService
from app.services.backend_client import BackendClient

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM operations with LangChain"""
    
    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service
        self.token_service = TokenUsageService()
        self.backend_client = BackendClient()
        
        # Initialize OpenAI model
        self.llm = ChatOpenAI(
            openai_api_key=settings.openai_api_key,
            model_name=settings.openai_model,
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens
        )
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=settings.max_chat_history,
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize prompt templates
        self._initialize_prompts()
    
    def _initialize_prompts(self):
        """Initialize prompt templates for different AI roles"""
        
        # System prompt for general AI assistant
        self.system_prompt = """You are Noki AI, an intelligent academic assistant designed to help students with their coursework, assignments, and study planning.

Your capabilities:
- Analyze academic content and provide detailed explanations
- Create study schedules and task lists
- Answer questions about course materials with depth and insight
- Help with project planning and time management
- Provide tutoring and learning support
- Analyze workload and provide prioritization recommendations
- Give detailed insights about projects, tasks, and deadlines

Guidelines:
- Always provide structured, actionable, and DETAILED responses
- Use the provided context (projects, tasks, todos, resources) to give comprehensive advice
- When analyzing data, provide insights, patterns, and recommendations
- Format responses as structured blocks for the UI
- Be thorough but organized - break down complex information
- Focus on academic productivity and learning
- ALWAYS reference conversation history when relevant
- If user asks about assignments/tasks/schedule and you have context data, provide DETAILED analysis
- Analyze due dates, priorities, and workload distribution
- Provide actionable recommendations based on the data

Conversation Context:
- User ID: {user_id}
- Conversation ID: {conversation_id}
- Available projects: {projects}
- Available tasks: {tasks}
- Available todos: {todos}
- Relevant resources: {resources}
- Recent conversation history: {conversation_history}

ANALYSIS REQUIREMENTS:
- When given projects/tasks/todos, analyze:
  * Due date patterns and urgency
  * Priority distribution
  * Workload balance
  * Potential conflicts or overlaps
  * Recommendations for scheduling
  * Progress tracking insights

- Provide detailed breakdowns:
  * Group items by project, priority, or due date
  * Highlight critical deadlines
  * Suggest optimal work order
  * Identify potential issues or concerns
  * Offer specific, actionable advice

IMPORTANT: When referencing conversation history, acknowledge what was discussed earlier and build upon it. Provide comprehensive, detailed analysis that helps students make informed decisions."""

        # Planner-specific prompt
        self.planner_prompt = """You are Noki AI in planning mode. Your job is to create structured study plans, schedules, and task lists.

When creating plans:
- Use ISO 8601 datetime format (YYYY-MM-DDTHH:MM:SSZ)
- Consider due dates and priorities
- Break down large tasks into manageable sessions
- Suggest realistic time allocations
- Include breaks and buffer time

Output format:
- Create todo_list blocks for task lists
- Use explanation_block for study strategies
- Include confirmation blocks for completed actions
- Always provide accept_decline options for proposed schedules"""

        # Tutor-specific prompt
        self.tutor_prompt = """You are Noki AI in tutoring mode. Your job is to explain concepts, provide learning guidance, and help with academic questions.

When tutoring:
- Explain concepts clearly and step-by-step
- Use examples and analogies
- Reference specific resources when available
- Encourage active learning
- Provide practice suggestions

Output format:
- Use explanation_block for detailed explanations
- Create resource_item blocks for relevant materials
- Include todo_list blocks for practice exercises
- Reference sources with proper citations"""

        # Research-specific prompt
        self.research_prompt = """You are Noki AI in research mode. Your job is to analyze resources, summarize information, and provide research insights.

When researching:
- Synthesize information from multiple sources
- Identify key concepts and themes
- Highlight important details and connections
- Provide balanced perspectives
- Suggest further research directions

Output format:
- Use explanation_block for summaries and analysis
- Create resource_item blocks for source materials
- Include todo_list blocks for research tasks
- Provide clear citations and references"""
    
    async def process_chat_request(self, chat_input: ChatInput) -> AIResponse:
        """
        Process a chat request and return AI response
        
        This is the main entry point for chat processing
        Enhanced to automatically fetch backend data when needed
        """
        try:
            # Step 1: Retrieve semantic context
            semantic_context = self._retrieve_context(chat_input)
            
            # Step 2: Determine if backend data is needed
            intent = self._determine_intent(chat_input, semantic_context)
            
            # Step 3: If intent detected and we have auth token, fetch data automatically
            # Get auth token from metadata if not directly in chat_input
            auth_token = chat_input.auth_token or (chat_input.metadata or {}).get("auth_token")
            
            if intent and intent.type == IntentType.BACKEND_QUERY and auth_token:
                try:
                    logger.info(f"Fetching backend data: {intent.targets}, period: {intent.payload.get('time_period', 'all')}")
                    
                    # Fetch data from backend
                    backend_data = await self.backend_client.fetch_data_for_ai(
                        user_id=chat_input.user_id,
                        auth_token=auth_token,
                        data_types=intent.targets or [],
                        time_period=intent.payload.get("time_period"),
                        project_ids=intent.filters.get("project_ids") if intent.filters else None,
                        include_completed=intent.payload.get("include_completed", False)
                    )
                    
                    # Update chat_input with fetched data
                    if "projects" in backend_data:
                        from app.models.schemas import Project
                        chat_input.projects = [
                            Project(project_id=p.get("project_id", ""), title=p.get("title", ""), 
                                   description=p.get("description"), instructor=p.get("instructor"))
                            for p in backend_data["projects"]
                        ]
                    
                    if "tasks" in backend_data:
                        from app.models.schemas import Task, TaskStatus
                        chat_input.tasks = []
                        for t in backend_data["tasks"]:
                            try:
                                due_datetime = None
                                if t.get("due_datetime"):
                                    # Handle various datetime formats
                                    dt_str = t["due_datetime"].replace("Z", "+00:00")
                                    due_datetime = datetime.fromisoformat(dt_str)
                                
                                chat_input.tasks.append(
                                    Task(
                                        task_id=t.get("task_id", ""),
                                        title=t.get("title", ""),
                                        description=t.get("description"),
                                        due_datetime=due_datetime,
                                        status=TaskStatus.DONE if t.get("status") == "done" else TaskStatus.NOT_STARTED,
                                        project_id=t.get("project_id")
                                    )
                                )
                            except Exception as e:
                                logger.warn(f"Error parsing task {t.get('task_id')}: {e}")
                                # Add task without due_datetime if parsing fails
                                chat_input.tasks.append(
                                    Task(
                                        task_id=t.get("task_id", ""),
                                        title=t.get("title", ""),
                                        description=t.get("description"),
                                        status=TaskStatus.NOT_STARTED,
                                        project_id=t.get("project_id")
                                    )
                                )
                    
                    if "todos" in backend_data:
                        chat_input.todos = backend_data["todos"]
                    
                    logger.info(f"Fetched {len(chat_input.projects or [])} projects, {len(chat_input.tasks or [])} tasks, {len(chat_input.todos or [])} todos")
                    
                except Exception as e:
                    logger.error(f"Failed to fetch backend data: {e}")
                    # Continue with existing context if fetch fails
            
            # Step 4: Generate response with enriched context
            response = await self._generate_response(chat_input, semantic_context)
            
            # Step 5: Save message to vector store
            self._save_message(chat_input)
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to process chat request: {e}")
            return AIResponse(
                stage=Stage.COMPLETE,
                conversation_id=chat_input.conversation_id,
                text="I apologize, but I encountered an error processing your request. Please try again.",
                blocks=[{
                    "type": "confirmation",
                    "message": "Error occurred. Please try again."
                }]
            )
    
    def continue_with_context(self, conversation_id: str, user_id: str,
                            context_data: Dict[str, Any]) -> AIResponse:
        """
        Continue processing after backend provides context data
        """
        try:
            # Get recent chat history for context
            recent_history = self.vector_service.get_recent_chat_history(
                user_id=user_id,
                conversation_id=conversation_id
            )
            
            # Generate response with the provided context and history
            response = self._generate_response_with_context(
                conversation_id, user_id, context_data, recent_history
            )
            
            # Save the context response to maintain conversation history
            self._save_context_response(conversation_id, user_id, context_data, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to continue with context: {e}")
            return AIResponse(
                stage=Stage.COMPLETE,
                conversation_id=conversation_id,
                text="I apologize, but I encountered an error processing the context data.",
                blocks=[{
                    "type": "confirmation",
                    "message": "Error occurred. Please try again."
                }]
            )
    
    def _retrieve_context(self, chat_input: ChatInput) -> List[Document]:
        """Retrieve semantic context from vector database"""
        try:
            # Get project and task IDs for filtering
            project_ids = [p.project_id for p in (chat_input.projects or [])]
            task_ids = [t.task_id for t in (chat_input.tasks or [])]
            
            # Search for relevant context
            context = self.vector_service.search_semantic_context(
                user_id=chat_input.user_id,
                conversation_id=chat_input.conversation_id,
                query=chat_input.prompt,
                project_ids=project_ids,
                task_ids=task_ids
            )
            
            # Add recent chat history
            recent_history = self.vector_service.get_recent_chat_history(
                user_id=chat_input.user_id,
                conversation_id=chat_input.conversation_id
            )
            
            return context + recent_history
            
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return []
    
    def _determine_intent(self, chat_input: ChatInput, context: List[Document]) -> Optional[AIIntent]:
        """Determine if backend data is needed - Enhanced with time period detection"""
        try:
            prompt_lower = chat_input.prompt.lower()
            
            # Check if we already have sufficient context data
            has_projects = len(chat_input.projects or []) > 0
            has_tasks = len(chat_input.tasks or []) > 0
            has_todos = len(chat_input.todos or []) > 0
            
            # Time period keywords
            time_keywords = {
                "today": ["today", "this day"],
                "this_week": ["this week", "this week's", "week", "weekly", "upcoming week"],
                "this_month": ["this month", "monthly", "this month's"],
                "next_two_months": ["next month", "next two months", "upcoming months"],
                "overdue": ["overdue", "past due", "late", "missed", "expired"],
            }
            
            # Detect time period
            detected_period = None
            for period, keywords in time_keywords.items():
                if any(keyword in prompt_lower for keyword in keywords):
                    detected_period = period
                    break
            
            # Entity keywords
            project_keywords = ["project", "projects", "course", "courses", "class", "classes"]
            task_keywords = ["task", "tasks", "assignment", "assignments", "homework", "deadline", "deadlines"]
            todo_keywords = ["todo", "todos", "to-do", "to-dos", "item", "items", "checklist"]
            
            # Comprehensive keywords
            comprehensive_keywords = [
                "all my", "show me all", "list all", "give me all", "what are all",
                "what are my", "complete list", "full list", "everything", "entire", "comprehensive"
            ]
            
            # Determine what user wants
            wants_projects = any(keyword in prompt_lower for keyword in project_keywords)
            wants_tasks = any(keyword in prompt_lower for keyword in task_keywords)
            wants_todos = any(keyword in prompt_lower for keyword in todo_keywords)
            wants_comprehensive = any(keyword in prompt_lower for keyword in comprehensive_keywords)
            
            # If no specific entity mentioned, assume tasks/todos (most common)
            if not wants_projects and not wants_tasks and not wants_todos:
                wants_tasks = True
                wants_todos = True
            
            # Determine if we need to fetch data
            needs_projects = wants_projects and (not has_projects or wants_comprehensive or detected_period)
            needs_tasks = wants_tasks and (not has_tasks or wants_comprehensive or detected_period)
            needs_todos = wants_todos and (not has_todos or wants_comprehensive or detected_period)
            
            # Get auth token from metadata if not directly in chat_input
            auth_token = chat_input.auth_token or (chat_input.metadata or {}).get("auth_token")
            
            # If we need any data and have auth token, return intent
            if (needs_projects or needs_tasks or needs_todos) and auth_token:
                targets = []
                if needs_projects:
                    targets.append("projects")
                if needs_tasks:
                    targets.append("tasks")
                if needs_todos:
                    targets.append("todos")
                
                filters = {}
                if chat_input.projects:
                    filters["project_ids"] = [p.project_id for p in chat_input.projects]
                
                payload = {
                    "time_period": detected_period or "all",
                    "include_completed": False
                }
                
                return AIIntent(
                    type=IntentType.BACKEND_QUERY,
                    targets=targets,
                    filters=filters,
                    payload=payload
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to determine intent: {e}")
            return None
    
    async def _generate_response(self, chat_input: ChatInput, context: List[Document]) -> AIResponse:
        """Generate AI response with structured blocks"""
        try:
            # Format context for prompt
            context_text = self._format_context(context)
            projects_text = self._format_projects(chat_input.projects or [])
            tasks_text = self._format_tasks(chat_input.tasks or [])
            todos_text = self._format_todos(chat_input.todos or [])
            conversation_history = self._format_conversation_history(
                chat_input.conversation_history or []
            )
            
            # Create system message
            system_message = SystemMessage(content=self.system_prompt.format(
                user_id=chat_input.user_id,
                conversation_id=chat_input.conversation_id,
                projects=projects_text,
                tasks=tasks_text,
                todos=todos_text,
                resources=context_text,
                conversation_history=conversation_history
            ))
            
            # Create human message
            human_message = HumanMessage(content=chat_input.prompt)
            
            # Estimate prompt tokens before making the call
            prompt_tokens = self.token_service.estimate_prompt_tokens(
                chat_input.dict(), context
            )
            
            # Get response from LLM
            messages = [system_message, human_message]
            response = self.llm(messages)
            
            # Count completion tokens
            completion_tokens = self.token_service.count_tokens(response.content)
            
            # Create token usage object
            token_usage_data = self.token_service.create_token_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model=settings.openai_model
            )
            token_usage = TokenUsage(**token_usage_data)
            
            # Parse response and create blocks
            blocks = self._parse_response_to_blocks(response.content)
            
            return AIResponse(
                stage=Stage.RESPONSE,
                conversation_id=chat_input.conversation_id,
                text=response.content,
                blocks=blocks,
                token_usage=token_usage
            )
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return AIResponse(
                stage=Stage.COMPLETE,
                conversation_id=chat_input.conversation_id,
                text="I apologize, but I couldn't generate a proper response.",
                blocks=[{
                    "type": "confirmation",
                    "message": "Response generation failed."
                }]
            )
    
    def _generate_response_with_context(self, conversation_id: str, user_id: str,
                                      context_data: Dict[str, Any], 
                                      recent_history: List[Document] = None) -> AIResponse:
        """Generate response with backend-provided context"""
        try:
            # Process the context data to create a meaningful response
            assignments = context_data.get("assignments", [])
            schedule = context_data.get("schedule", [])
            
            # Handle schedule as either list or dict
            schedule_items = schedule if isinstance(schedule, list) else schedule.get("items", [])
            available_slots = schedule if isinstance(schedule, dict) else []
            
            # Create a comprehensive response based on the context
            response_text = "Perfect! I've analyzed your assignments and schedule. "
            
            if assignments:
                response_text += f"I found {len(assignments)} assignments that need attention. "
            
            if schedule_items:
                response_text += f"I can see {len(schedule_items)} schedule items and will optimize your todo list accordingly. "
            elif available_slots:
                response_text += "I can see your available time slots and will optimize your todo list accordingly. "
            
            response_text += "Let me create a comprehensive todo list that aligns with your academic goals and schedule."
            
            # Estimate tokens for context processing
            context_text = str(context_data)
            prompt_tokens = self.token_service.count_tokens(context_text)
            completion_tokens = self.token_service.count_tokens(response_text)
            
            # Create token usage object
            token_usage_data = self.token_service.create_token_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model=settings.openai_model
            )
            token_usage = TokenUsage(**token_usage_data)
            
            return AIResponse(
                stage=Stage.RESPONSE,
                conversation_id=conversation_id,
                text=response_text,
                blocks=[{
                    "type": "explanation_block",
                    "title": "Context Analysis Complete",
                    "description": "I've processed your assignments and schedule data.",
                    "blocks": [
                        {
                            "title": "Assignments Found",
                            "description": f"Found {len(assignments)} assignments to work with",
                            "list": [assignment.get("title", "Untitled") for assignment in assignments[:3]]
                        },
                        {
                            "title": "Schedule Analysis",
                            "description": "Analyzed your available time slots",
                            "list": [f"Schedule items: {len(schedule_items)}"]
                        }
                    ]
                }],
                token_usage=token_usage
            )
            
        except Exception as e:
            logger.error(f"Failed to generate response with context: {e}")
            return AIResponse(
                stage=Stage.COMPLETE,
                conversation_id=conversation_id,
                text="I encountered an error processing the context data.",
                blocks=[{
                    "type": "confirmation",
                    "message": "Context processing failed."
                }]
            )
    
    def _format_context(self, context: List[Document]) -> str:
        """Format context documents for prompt"""
        if not context:
            return "No relevant context found."
        
        formatted = []
        for doc in context[:5]:  # Limit to top 5
            # Include more context for context responses
            if doc.metadata.get("stage") == "context_response":
                formatted.append(f"[Context Data]: {doc.page_content}")
            else:
                formatted.append(f"- {doc.page_content[:200]}...")
        
        return "\n".join(formatted)
    
    def _format_projects(self, projects: List) -> str:
        """Format projects for prompt"""
        if not projects:
            return "No projects available."
        
        formatted = []
        for project in projects:
            formatted.append(f"- {project.title}: {project.description or 'No description'}")
        
        return "\n".join(formatted)
    
    def _format_tasks(self, tasks: List) -> str:
        """Format tasks for prompt"""
        if not tasks:
            return "No tasks available."
        
        formatted = []
        for task in tasks:
            status = f" ({task.status})" if task.status else ""
            due = f" - Due: {task.due_datetime}" if task.due_datetime else ""
            formatted.append(f"- {task.title}{status}{due}")
        
        return "\n".join(formatted)
    
    def _format_conversation_history(self, context: List[Document]) -> str:
        """Format conversation history for prompt"""
        if not context:
            return "No previous conversation history."
        
        # Filter for chat messages (not resources)
        chat_messages = []
        for doc in context:
            if doc.metadata.get("type") == "chat":
                stage = doc.metadata.get("stage", "unknown")
                content = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                chat_messages.append(f"[{stage}]: {content}")
        
        if not chat_messages:
            return "No previous conversation history."
        
        # Return recent messages (limit to last 3)
        return "\n".join(chat_messages[-3:])
    
    def _parse_response_to_blocks(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured blocks"""
        # This is a simplified parser - in production, you'd want more sophisticated parsing
        blocks = []
        
        # For now, create a simple explanation block
        blocks.append({
            "type": "explanation_block",
            "title": "AI Response",
            "description": response_text,
            "blocks": []
        })
        
        return blocks
    
    def _save_message(self, chat_input: ChatInput):
        """Save chat message to vector store"""
        try:
            message_id, embedding_tokens = self.vector_service.embed_message(
                user_id=chat_input.user_id,
                conversation_id=chat_input.conversation_id,
                message_id=f"msg_{datetime.utcnow().timestamp()}",
                message_content=chat_input.prompt,
                metadata={
                    "stage": chat_input.stage,
                    "projects": [p.project_id for p in (chat_input.projects or [])],
                    "tasks": [t.task_id for t in (chat_input.tasks or [])]
                }
            )
            logger.info(f"Saved message {message_id} with {embedding_tokens} embedding tokens")
        except Exception as e:
            logger.error(f"Failed to save message: {e}")
    
    def _save_context_response(self, conversation_id: str, user_id: str, 
                              context_data: Dict[str, Any], response: AIResponse):
        """Save context response to vector store for conversation history"""
        try:
            # Create a detailed summary of the context response for embedding
            context_summary = f"Context processed: {response.text}"
            if response.blocks:
                context_summary += f" Generated {len(response.blocks)} response blocks."
            
            # Add specific assignment/schedule details to the summary
            assignments = context_data.get("assignments", [])
            schedule = context_data.get("schedule", [])
            
            if assignments:
                context_summary += f" Assignments available: "
                for assignment in assignments[:3]:  # Limit to first 3
                    title = assignment.get("title", "Untitled")
                    due_date = assignment.get("due_date", "No due date")
                    status = assignment.get("status", "Unknown status")
                    context_summary += f"{title} (due: {due_date}, status: {status}); "
            
            if schedule:
                context_summary += f" Schedule items: "
                for item in schedule[:3]:  # Limit to first 3
                    title = item.get("title", "Untitled")
                    start_time = item.get("start_time", "No time")
                    context_summary += f"{title} at {start_time}; "
            
            message_id, embedding_tokens = self.vector_service.embed_message(
                user_id=user_id,
                conversation_id=conversation_id,
                message_id=f"ctx_{datetime.utcnow().timestamp()}",
                message_content=context_summary,
                metadata={
                    "stage": "context_response",
                    "type": "chat",  # Ensure it's marked as chat for history retrieval
                    "context_data_keys": list(context_data.keys()),
                    "response_blocks_count": len(response.blocks or [])
                }
            )
            logger.info(f"Saved context response {message_id} with {embedding_tokens} embedding tokens")
        except Exception as e:
            logger.error(f"Failed to save context response: {e}")
