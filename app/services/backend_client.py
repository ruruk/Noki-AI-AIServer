"""
Backend API Client Service
Handles communication with the NestJS backend to fetch user data
"""
import logging
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from config import settings

logger = logging.getLogger(__name__)


class BackendClient:
    """Client for making requests to the NestJS backend"""
    
    def __init__(self):
        self.base_url = settings.backend_url or "http://localhost:3000"
        self.timeout = 30.0
        
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to backend"""
        url = f"{self.base_url}{endpoint}"
        default_headers = {
            "Content-Type": "application/json",
            **(headers or {})
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=default_headers,
                    params=params,
                    json=json_data
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Backend API error: {e}")
            raise Exception(f"Failed to communicate with backend: {str(e)}")
    
    async def get_user_projects(
        self,
        user_id: str,
        auth_token: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get all projects for a user"""
        try:
            headers = {"Authorization": f"Bearer {auth_token}"}
            params = filters or {}
            
            response = await self._make_request(
                method="GET",
                endpoint=f"/projects/user/{user_id}",
                headers=headers,
                params=params
            )
            
            # Handle both wrapped and unwrapped responses
            if isinstance(response, dict) and "data" in response:
                return response["data"]
            return response if isinstance(response, list) else []
        except Exception as e:
            logger.error(f"Error fetching projects: {e}")
            return []
    
    async def fetch_data_for_ai(
        self,
        user_id: str,
        auth_token: str,
        data_types: List[str],
        time_period: Optional[str] = None,
        project_ids: Optional[List[str]] = None,
        include_completed: bool = False
    ) -> Dict[str, Any]:
        """
        Fetch data for AI processing using the dedicated endpoint
        
        Args:
            user_id: User ID
            auth_token: JWT auth token
            data_types: List of data types to fetch (projects, tasks, todos)
            time_period: Optional time period filter (today, this_week, this_month, next_two_months, overdue, all)
            project_ids: Optional list of project IDs to filter by
            include_completed: Whether to include completed items
        """
        try:
            headers = {"Authorization": f"Bearer {auth_token}"}
            
            payload = {
                "data_types": data_types,
                "include_completed": include_completed
            }
            
            if time_period:
                payload["time_period"] = time_period
            
            if project_ids:
                payload["project_ids"] = project_ids
            
            response = await self._make_request(
                method="POST",
                endpoint="/ai/fetch-data",
                headers=headers,
                json_data=payload
            )
            
            # Handle wrapped response
            if isinstance(response, dict) and "data" in response:
                return response["data"]
            return response
            
        except Exception as e:
            logger.error(f"Error fetching data for AI: {e}")
            return {"projects": [], "tasks": [], "todos": []}
    
    async def get_conversation_history(
        self,
        conversation_id: str,
        auth_token: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get conversation history"""
        try:
            headers = {"Authorization": f"Bearer {auth_token}"}
            
            response = await self._make_request(
                method="GET",
                endpoint=f"/ai/get_conversation_history/{conversation_id}",
                headers=headers
            )
            
            messages = response.get("data", {}).get("messages", []) if isinstance(response, dict) else []
            
            # Return last N messages
            return messages[-limit:] if messages else []
            
        except Exception as e:
            logger.error(f"Error fetching conversation history: {e}")
            return []
