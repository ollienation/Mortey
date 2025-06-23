# core/p6_integration_manager.py - Structured P6 Database Integration

import logging
import asyncio
import time
import json
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from tools.p6_client import P6Client, FilterBuilder
from tools.p6_filter_mapper import P6FilterMapper
from core.enhanced_state import EnhancedAssistantState, ScratchpadManager
from core.error_handling import ErrorHandler, with_error_handling

logger = logging.getLogger("p6_integration_manager")

class QueryType(Enum):
    """Types of P6 queries"""
    PROJECT_SEARCH = "project_search"
    ACTIVITY_ANALYSIS = "activity_analysis"
    RESOURCE_QUERY = "resource_query"
    SCHEDULE_ANALYSIS = "schedule_analysis"
    COMPLEX_FILTER = "complex_filter"

@dataclass
class P6QueryRequest:
    """Structured P6 query request"""
    query_id: str = field(default_factory=lambda: f"p6_{int(time.time())}")
    query_type: QueryType = QueryType.PROJECT_SEARCH
    natural_language_query: str = ""
    structured_filters: Optional[Dict[str, Any]] = None
    requested_fields: Optional[List[str]] = None
    order_by: Optional[str] = None
    limit: int = 100
    requesting_agent: str = ""
    context_data: Optional[Dict[str, Any]] = None
    priority: int = 1  # 1=low, 5=high

@dataclass
class P6QueryResponse:
    """Structured P6 query response"""
    query_id: str = ""
    success: bool = False
    data: List[Dict[str, Any]] = field(default_factory=list)
    total_count: int = 0
    execution_time_ms: float = 0.0
    filter_applied: str = ""
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class P6IntegrationManager:
    """Manages structured P6 database integration with enhanced error handling"""
    
    def __init__(self):
        self.client: Optional[P6Client] = None
        self.filter_mapper = P6FilterMapper()
        self.query_cache: Dict[str, P6QueryResponse] = {}
        self.active_queries: Dict[str, asyncio.Task] = {}
        self.connection_pool_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_response_time": 0.0,
            "cache_hits": 0
        }
    
    async def initialize_connection(self, username: str, password: str, database: str) -> bool:
        """Initialize P6 connection with enhanced error handling"""
        try:
            self.client = P6Client()
            await asyncio.to_thread(self.client.initialize, username, password, database)
            
            # Test connection
            test_result = await self.execute_test_query()
            if test_result.success:
                logger.info("✅ P6 Integration Manager initialized successfully")
                return True
            else:
                logger.error(f"❌ P6 connection test failed: {test_result.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"❌ P6 initialization failed: {e}")
            return False
    
    async def execute_structured_query(
        self, 
        state: EnhancedAssistantState, 
        request: P6QueryRequest
    ) -> tuple[EnhancedAssistantState, P6QueryResponse]:
        """Execute structured P6 query with state management"""
        
        # Check cache first
        cache_key = self._generate_cache_key(request)
        if cache_key in self.query_cache:
            self.connection_pool_stats["cache_hits"] += 1
            cached_response = self.query_cache[cache_key]
            logger.info(f"P6 query cache hit: {request.query_id}")
            return state, cached_response
        
        # Store query context in scratchpad
        state = ScratchpadManager.set_data(
            state, 
            f"p6_query_{request.query_id}", 
            {
                "request": request,
                "status": "executing",
                "started_at": time.time()
            },
            request.requesting_agent,
            tags=["p6_query", "active"]
        )
        
        start_time = time.time()
        
        try:
            # Execute query based on type
            if request.query_type == QueryType.PROJECT_SEARCH:
                response = await self._execute_project_search(request)
            elif request.query_type == QueryType.ACTIVITY_ANALYSIS:
                response = await self._execute_activity_analysis(request)
            elif request.query_type == QueryType.RESOURCE_QUERY:
                response = await self._execute_resource_query(request)
            elif request.query_type == QueryType.SCHEDULE_ANALYSIS:
                response = await self._execute_schedule_analysis(request)
            elif request.query_type == QueryType.COMPLEX_FILTER:
                response = await self._execute_complex_filter(request)
            else:
                response = P6QueryResponse(
                    query_id=request.query_id,
                    success=False,
                    error_message=f"Unsupported query type: {request.query_type}"
                )
            
            # Update execution time
            response.execution_time_ms = (time.time() - start_time) * 1000
            
            # Update stats
            self.connection_pool_stats["total_queries"] += 1
            if response.success:
                self.connection_pool_stats["successful_queries"] += 1
            else:
                self.connection_pool_stats["failed_queries"] += 1
            
            # Update average response time
            total_time = (self.connection_pool_stats["avg_response_time"] * 
                         (self.connection_pool_stats["total_queries"] - 1) + 
                         response.execution_time_ms)
            self.connection_pool_stats["avg_response_time"] = total_time / self.connection_pool_stats["total_queries"]
            
            # Cache successful responses
            if response.success and len(response.data) > 0:
                self.query_cache[cache_key] = response
            
            # Update scratchpad with results
            state = ScratchpadManager.set_data(
                state,
                f"p6_query_{request.query_id}",
                {
                    "request": request,
                    "response": response,
                    "status": "completed" if response.success else "failed",
                    "completed_at": time.time()
                },
                request.requesting_agent,
                tags=["p6_query", "completed"]
            )
            
            return state, response
            
        except Exception as e:
            error_response = P6QueryResponse(
                query_id=request.query_id,
                success=False,
                error_message=f"Query execution failed: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
            self.connection_pool_stats["total_queries"] += 1
            self.connection_pool_stats["failed_queries"] += 1
            
            logger.error(f"❌ P6 query failed: {e}")
            return state, error_response
    
    async def _execute_project_search(self, request: P6QueryRequest) -> P6QueryResponse:
        """Execute project search query"""
        if not self.client:
            return P6QueryResponse(
                query_id=request.query_id,
                success=False,
                error_message="P6 client not initialized"
            )
        
        try:
            # Build filter using natural language
            filter_builder = self.filter_mapper.build_filter(
                request.natural_language_query,
                request.structured_filters.get("status") if request.structured_filters else None
            )
            
            # Add requested fields
            if request.requested_fields:
                filter_builder.select(*request.requested_fields)
            else:
                filter_builder.select("Id", "Name", "Description", "Status", "StartDate", "FinishDate")
            
            # Add ordering
            if request.order_by:
                parts = request.order_by.split()
                field = parts[0]
                desc = len(parts) > 1 and parts[1].lower() == "desc"
                filter_builder.order_by(field, desc=desc)
            
            # Execute query
            projects = await asyncio.to_thread(
                self.client.get_projects,
                filter_builder=filter_builder,
                limit=request.limit
            )
            
            # Build filter explanation
            filter_query = filter_builder.build_filter()
            filter_explanation = self.filter_mapper.get_filter_explanation(filter_query or "")
            
            return P6QueryResponse(
                query_id=request.query_id,
                success=True,
                data=projects,
                total_count=len(projects),
                filter_applied=filter_explanation,
                metadata={
                    "query_type": "project_search",
                    "filter_string": filter_query,
                    "requested_fields": request.requested_fields
                }
            )
            
        except Exception as e:
            return P6QueryResponse(
                query_id=request.query_id,
                success=False,
                error_message=f"Project search failed: {str(e)}"
            )
    
    async def _execute_activity_analysis(self, request: P6QueryRequest) -> P6QueryResponse:
        """Execute activity analysis query"""
        if not self.client:
            return P6QueryResponse(
                query_id=request.query_id,
                success=False,
                error_message="P6 client not initialized"
            )
        
        try:
            # Extract project ID from context or query
            project_id = None
            if request.context_data and "project_id" in request.context_data:
                project_id = request.context_data["project_id"]
            else:
                # Try to find project first
                projects = await asyncio.to_thread(
                    self.client.get_projects,
                    limit=1
                )
                if projects:
                    project_id = projects[0]["Id"]
            
            if not project_id:
                return P6QueryResponse(
                    query_id=request.query_id,
                    success=False,
                    error_message="No project context found for activity analysis"
                )
            
            # Build activity filter
            filter_builder = FilterBuilder()
            
            # Add natural language filtering
            if request.natural_language_query:
                nl_filter = self.filter_mapper.build_filter(request.natural_language_query)
                if nl_filter.build_filter():
                    filter_builder = nl_filter
            
            # Set fields
            if request.requested_fields:
                filter_builder.select(*request.requested_fields)
            else:
                filter_builder.select(
                    "Id", "Name", "Status", "PercentComplete",
                    "StartDate", "FinishDate", "Duration"
                )
            
            # Execute query
            activities = await asyncio.to_thread(
                self.client.get_activities,
                project_id=project_id,
                filter_builder=filter_builder,
                limit=request.limit
            )
            
            return P6QueryResponse(
                query_id=request.query_id,
                success=True,
                data=activities,
                total_count=len(activities),
                metadata={
                    "query_type": "activity_analysis",
                    "project_id": project_id
                }
            )
            
        except Exception as e:
            return P6QueryResponse(
                query_id=request.query_id,
                success=False,
                error_message=f"Activity analysis failed: {str(e)}"
            )
    
    async def _execute_resource_query(self, request: P6QueryRequest) -> P6QueryResponse:
        """Execute resource query"""
        # Implementation similar to other query types
        return P6QueryResponse(
            query_id=request.query_id,
            success=False,
            error_message="Resource queries not yet implemented"
        )
    
    async def _execute_schedule_analysis(self, request: P6QueryRequest) -> P6QueryResponse:
        """Execute schedule analysis"""
        # Implementation for schedule analysis
        return P6QueryResponse(
            query_id=request.query_id,
            success=False,
            error_message="Schedule analysis not yet implemented"
        )
    
    async def _execute_complex_filter(self, request: P6QueryRequest) -> P6QueryResponse:
        """Execute complex filter query"""
        # Implementation for complex filtering
        return P6QueryResponse(
            query_id=request.query_id,
            success=False,
            error_message="Complex filters not yet implemented"
        )
    
    async def execute_test_query(self) -> P6QueryResponse:
        """Execute test query to verify connection"""
        test_request = P6QueryRequest(
            query_type=QueryType.PROJECT_SEARCH,
            natural_language_query="active projects",
            limit=1,
            requesting_agent="system"
        )
        
        # Create minimal state for test
        test_state = {
            "scratchpad": {},
            "session_id": "test",
            "user_id": "system"
        }
        
        _, response = await self.execute_structured_query(test_state, test_request)
        return response
    
    def _generate_cache_key(self, request: P6QueryRequest) -> str:
        """Generate cache key for request"""
        key_data = {
            "query_type": request.query_type.value,
            "natural_language_query": request.natural_language_query,
            "structured_filters": request.structured_filters,
            "requested_fields": request.requested_fields,
            "order_by": request.order_by,
            "limit": request.limit
        }
        return f"p6_cache_{hash(json.dumps(key_data, sort_keys=True))}"
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return {
            **self.connection_pool_stats,
            "cache_size": len(self.query_cache),
            "active_queries": len(self.active_queries),
            "client_connected": self.client is not None
        }

# Global P6 integration manager
p6_integration_manager = P6IntegrationManager()
