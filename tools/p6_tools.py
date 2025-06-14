# tools/p6_tools.py - Comprehensive P6 Tool Suite
import logging
import asyncio
import json
import os
import time
from typing import Optional, List, Any, Union
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from tools.p6_client import P6Client, P6ClientConfig, P6AuthenticationError, P6APIError
from core.error_handling import ErrorHandler
from core.circuit_breaker import global_circuit_breaker
from config.settings import config

logger = logging.getLogger("p6_tools")

class P6ToolOperation(Enum):
    """P6 operation types for tracking and validation"""
    PROJECT_SEARCH = "project_search"
    ACTIVITY_ANALYSIS = "activity_analysis" 
    RESOURCE_QUERY = "resource_query"
    SCHEDULE_ANALYSIS = "schedule_analysis"
    DATA_EXPORT = "data_export"

@dataclass
class P6SecurityConfig:
    """Security configuration for P6 operations"""
    max_results_per_query: int = 1000
    allowed_operations: set[str] = field(default_factory=lambda: {
        "read", "search", "analyze", "export", "list"
    })
    read_only_mode: bool = True
    require_confirmation_for_writes: bool = True
    max_concurrent_requests: int = 5

class P6ToolsManager:
    """Manager for P6 tools with session management and security"""
    
    def __init__(self):
        self.client: Optional[P6Client] = None
        self.security_config = P6SecurityConfig()
        self.operation_stats: dict[str, int] = {}
        self.current_project_context: Optional[dict[str, Any]] = None
        
    async def initialize(self, username: str, password: str, database_name: str):
        """Initialize P6 client connection"""
        try:
            self.client = P6Client()
            auth_key = os.getenv("P6_AUTH_KEY")
            if auth_key:
                await self.client.initialize_with_auth_key(auth_key, database_name)
            else:
                await self.client.initialize(username, password, database_name)
                
            logger.info("✅ P6 tools manager initialized")
        except Exception as e:
            logger.error(f"❌ P6 tools initialization failed: {e}")
            raise

    async def initialize_with_auth_key(self, auth_key: str, database_name: str):
        """
        Initialise the underlying P6Client with a pre-encoded Basic-auth key.

        Mirrors the username/password path so AgentFactory can call it directly.
        """
        try:
            self.client = P6Client()
            await self.client.initialize_with_auth_key(auth_key, database_name)
            logger.info("✅ P6 tools manager initialised with auth key")
        except Exception as e:
            logger.error(f"❌ P6 tools initialisation failed: {e}")
            raise
    
    def _record_operation(self, operation: P6ToolOperation, success: bool = True):
        """Track operation statistics"""
        key = f"{operation.value}_{'success' if success else 'failure'}"
        self.operation_stats[key] = self.operation_stats.get(key, 0) + 1
    
    def _validate_operation(self, operation_type: str) -> tuple[bool, str]:
        """Validate operation against security policy"""
        if operation_type.lower() not in self.security_config.allowed_operations:
            return False, f"Operation '{operation_type}' not permitted"
        
        if not self.client:
            return False, "P6 client not initialized"
            
        return True, "Valid"
    
    def get_tools(self) -> List[Any]:
        """Get all P6 tools with proper client injection"""
        
        @tool
        def search_projects(
            search_query: str = Field(description="Natural language search query for projects"),
            status_filter: Optional[str] = Field(default=None, description="Project status filter (Active, Completed, etc.)"),
            max_results: int = Field(default=20, description="Maximum number of results to return")
        ) -> str:
            """
            Search for projects in Primavera P6 using natural language queries.
            Supports filtering by status, date ranges, and other project attributes.
            """
            return asyncio.create_task(
                self._search_projects_async(search_query, status_filter, max_results)
            ).result() if asyncio.get_event_loop().is_running() else \
            asyncio.run(self._search_projects_async(search_query, status_filter, max_results))
        
        @tool
        def analyze_project_schedule(
            project_identifier: str = Field(description="Project ID or name to analyze"),
            analysis_type: str = Field(default="overview", description="Type of analysis: overview, critical_path, progress, resources"),
            include_activities: bool = Field(default=True, description="Include activity details in analysis")
        ) -> str:
            """
            Analyze project schedule with comprehensive insights including critical path,
            progress tracking, and resource utilization.
            """
            return asyncio.create_task(
                self._analyze_schedule_async(project_identifier, analysis_type, include_activities)
            ).result() if asyncio.get_event_loop().is_running() else \
            asyncio.run(self._analyze_schedule_async(project_identifier, analysis_type, include_activities))
        
        @tool
        def get_activity_status(
            project_identifier: str = Field(description="Project ID or name"),
            activity_filter: Optional[str] = Field(default=None, description="Filter for specific activities"),
            status_type: str = Field(default="all", description="Status type: all, in_progress, completed, not_started, delayed")
        ) -> str:
            """
            Get detailed activity status information for a project including progress,
            start/finish dates, and resource assignments.
            """
            return asyncio.create_task(
                self._get_activity_status_async(project_identifier, activity_filter, status_type)
            ).result() if asyncio.get_event_loop().is_running() else \
            asyncio.run(self._get_activity_status_async(project_identifier, activity_filter, status_type))
        
        @tool
        def analyze_resource_utilization(
            project_identifier: Optional[str] = Field(default=None, description="Specific project ID or name, or None for all projects"),
            resource_type: str = Field(default="all", description="Resource type: all, labor, material, equipment"),
            time_period: Optional[str] = Field(default=None, description="Time period for analysis (this_month, this_quarter, etc.)")
        ) -> str:
            """
            Analyze resource utilization across projects with detailed allocation
            and availability information.
            """
            return asyncio.create_task(
                self._analyze_resources_async(project_identifier, resource_type, time_period)
            ).result() if asyncio.get_event_loop().is_running() else \
            asyncio.run(self._analyze_resources_async(project_identifier, resource_type, time_period))
        
        @tool 
        def export_project_data(
            project_identifier: str = Field(description="Project ID or name to export"),
            data_type: str = Field(default="schedule", description="Data type: schedule, resources, activities, wbs"),
            export_format: str = Field(default="json", description="Export format: json, csv, summary"),
            include_details: bool = Field(default=True, description="Include detailed information in export")
        ) -> str:
            """
            Export project data in various formats for further analysis or reporting.
            Integrates with existing file management tools for saving exported data.
            """
            return asyncio.create_task(
                self._export_project_data_async(project_identifier, data_type, export_format, include_details)
            ).result() if asyncio.get_event_loop().is_running() else \
            asyncio.run(self._export_project_data_async(project_identifier, data_type, export_format, include_details))
        
        @tool
        def get_p6_system_status() -> str:
            """
            Get current P6 system status including connection health, 
            available databases, and performance metrics.
            """
            return asyncio.create_task(
                self._get_system_status_async()
            ).result() if asyncio.get_event_loop().is_running() else \
            asyncio.run(self._get_system_status_async())
        
        return [
            search_projects,
            analyze_project_schedule, 
            get_activity_status,
            analyze_resource_utilization,
            export_project_data,
            get_p6_system_status
        ]
    
    async def _search_projects_async(
        self, 
        search_query: str, 
        status_filter: Optional[str], 
        max_results: int
    ) -> str:
        """Async implementation of project search"""
        start_time = time.time()
        
        try:
            # Validate operation
            is_valid, message = self._validate_operation("search")
            if not is_valid:
                return f"❌ {message}"
            
            # Build P6 filter query from natural language
            filter_parts = []
            
            # Convert natural language to P6 filter syntax
            if "active" in search_query.lower():
                filter_parts.append("Status = 'Active'")
            elif "completed" in search_query.lower():
                filter_parts.append("Status = 'Completed'")
            
            if status_filter:
                filter_parts.append(f"Status = '{status_filter}'")
            
            # Search in project name and description
            if search_query and not any(word in search_query.lower() for word in ['active', 'completed', 'status']):
                filter_parts.append(f"(Name contains '{search_query}' or Description contains '{search_query}')")
            
            filter_query = " and ".join(filter_parts) if filter_parts else None
            
            # Execute search with optimized fields
            fields = ["Id", "Name", "Description", "Status", "StartDate", "FinishDate", "LastUpdateDate"]
            projects = await self.client.get_projects(
                filter_query=filter_query,
                fields=fields,
                order_by="LastUpdateDate desc",
                limit=min(max_results, self.security_config.max_results_per_query)
            )
            
            duration = (time.time() - start_time) * 1000
            self._record_operation(P6ToolOperation.PROJECT_SEARCH, True)
            
            if not projects:
                return f"No projects found matching '{search_query}' (searched in {duration:.1f}ms)"
            
            # Format results for natural language response
            response = [f"Found {len(projects)} projects matching '{search_query}' (in {duration:.1f}ms):\n"]
            
            for i, project in enumerate(projects, 1):
                name = project.get('Name', 'Unknown')
                description = project.get('Description', 'No description')[:100]
                status = project.get('Status', 'Unknown')
                start_date = project.get('StartDate', 'Not set')
                
                response.append(f"{i}. **{name}** (ID: {project.get('Id', 'N/A')})")
                response.append(f"   Status: {status} | Start: {start_date}")
                response.append(f"   Description: {description}...")
                response.append("")
            
            # Store project context for follow-up queries
            if len(projects) == 1:
                self.current_project_context = projects[0]
            
            return "\n".join(response)
            
        except P6APIError as e:
            self._record_operation(P6ToolOperation.PROJECT_SEARCH, False)
            return f"❌ P6 API error during project search: {e}"
        except Exception as e:
            self._record_operation(P6ToolOperation.PROJECT_SEARCH, False)
            logger.error(f"Project search failed: {e}")
            return f"❌ Error searching projects: {e}"
    
    async def _analyze_schedule_async(
        self, 
        project_identifier: str, 
        analysis_type: str, 
        include_activities: bool
    ) -> str:
        """Async implementation of schedule analysis"""
        start_time = time.time()
        
        try:
            # Find project by ID or name
            project = await self._find_project(project_identifier)
            if not project:
                return f"❌ Project '{project_identifier}' not found"
            
            project_id = project['Id']
            project_name = project.get('Name', 'Unknown')
            
            # Get activities for analysis
            activity_fields = [
                "Id", "Name", "Status", "ActualStartDate", "ActualFinishDate",
                "PlannedStartDate", "PlannedFinishDate", "PercentComplete",
                "Duration", "RemainingDuration", "IsCritical", "TotalFloat"
            ]
            
            activities = await self.client.get_activities(
                project_id=project_id,
                fields=activity_fields,
                limit=1000
            )
            
            duration = (time.time() - start_time) * 1000
            self._record_operation(P6ToolOperation.SCHEDULE_ANALYSIS, True)
            
            # Perform analysis based on type
            if analysis_type.lower() == "critical_path":
                return await self._analyze_critical_path(project, activities, duration)
            elif analysis_type.lower() == "progress":
                return await self._analyze_progress(project, activities, duration)
            elif analysis_type.lower() == "resources":
                return await self._analyze_project_resources(project, activities, duration)
            else:
                return await self._analyze_schedule_overview(project, activities, duration, include_activities)
                
        except Exception as e:
            self._record_operation(P6ToolOperation.SCHEDULE_ANALYSIS, False)
            logger.error(f"Schedule analysis failed: {e}")
            return f"❌ Error analyzing schedule: {e}"
    
    async def _find_project(self, identifier: str) -> Optional[dict[str, Any]]:
        """Find project by ID or name"""
        try:
            # Try exact ID match first
            projects = await self.client.get_projects(
                filter_query=f"Id = '{identifier}'",
                fields=["Id", "Name", "Description", "Status", "StartDate", "FinishDate"],
                limit=1
            )
            
            if projects:
                return projects[0]
            
            # Try name match
            projects = await self.client.get_projects(
                filter_query=f"Name = '{identifier}'",
                fields=["Id", "Name", "Description", "Status", "StartDate", "FinishDate"],
                limit=1
            )
            
            if projects:
                return projects[0]
            
            # Try partial name match
            projects = await self.client.get_projects(
                filter_query=f"Name contains '{identifier}'",
                fields=["Id", "Name", "Description", "Status", "StartDate", "FinishDate"],
                limit=5
            )
            
            return projects[0] if projects else None
            
        except Exception as e:
            logger.error(f"Error finding project '{identifier}': {e}")
            return None
    
    async def _analyze_critical_path(
        self, 
        project: dict[str, Any], 
        activities: List[dict[str, Any]], 
        query_duration: float
    ) -> str:
        """Analyze critical path activities"""
        critical_activities = [
            activity for activity in activities 
            if activity.get('IsCritical') == True
        ]
        
        total_activities = len(activities)
        critical_count = len(critical_activities)
        
        response = [
            f"📊 Critical Path Analysis for {project.get('Name', 'Unknown Project')}",
            f"Query completed in {query_duration:.1f}ms\n",
            f"**Summary:**",
            f"- Total Activities: {total_activities:,}",
            f"- Critical Activities: {critical_count:,} ({(critical_count/total_activities*100):.1f}%)",
            f"- Project Status: {project.get('Status', 'Unknown')}\n"
        ]
        
        if critical_activities:
            response.append("**Critical Path Activities:**")
            for i, activity in enumerate(critical_activities[:10], 1):
                name = activity.get('Name', 'Unknown')
                status = activity.get('Status', 'Unknown')
                percent_complete = activity.get('PercentComplete', 0)
                total_float = activity.get('TotalFloat', 0)
                
                response.append(f"{i}. **{name}**")
                response.append(f"   Status: {status} | Progress: {percent_complete}% | Float: {total_float} days")
            
            if len(critical_activities) > 10:
                response.append(f"   ... and {len(critical_activities) - 10} more critical activities")
        
        return "\n".join(response)
    
    async def _get_system_status_async(self) -> str:
        """Get P6 system status"""
        try:
            if not self.client:
                return "❌ P6 client not initialized"
            
            # Simple connectivity test
            start_time = time.time()
            projects = await self.client.get_projects(limit=1)
            response_time = (time.time() - start_time) * 1000
            
            return f"""🔌 P6 System Status:
✅ **Connection**: Healthy
⚡ **Response Time**: {response_time:.1f}ms
📊 **Operations**: {sum(self.operation_stats.values())} total
🔒 **Security**: Read-only mode active
📁 **Project Context**: {self.current_project_context.get('Name') if self.current_project_context else 'None'}
"""
        except Exception as e:
            return f"❌ P6 system status check failed: {e}"

    def get_stub_tools(self):
        @tool
        def p6_not_ready() -> str:
            "Inform the user that P6 is not initialised"
            return "Primavera P6 connection not established yet. Use `connect to P6` first."
        return [p6_not_ready]

# Global P6 tools manager instance
p6_tools_manager = P6ToolsManager()
