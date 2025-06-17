# tools/p6_tools.py - Simplified P6 Tools following working.py patterns

import logging
import json
import time
from typing import Optional, List, Any, Dict
from dataclasses import dataclass
from langchain_core.tools import tool
from pydantic import Field

from tools.p6_client import P6Client, P6ClientConfig, P6AuthenticationError, P6APIError

logger = logging.getLogger("p6_tools")

@dataclass
class P6SecurityConfig:
    """Security configuration for P6 operations"""
    max_results_per_query: int = 1000
    read_only_mode: bool = True

class P6ToolsManager:
    """Simplified P6 tools manager following working.py session approach"""

    def __init__(self):
        self.client: Optional[P6Client] = None
        self.security_config = P6SecurityConfig()
        self.operation_stats: Dict[str, int] = {}
        self.current_project_context: Optional[Dict[str, Any]] = None

    def initialize(self, username: str, password: str, database_name: str):
        """Initialize P6 client connection - synchronous"""
        try:
            self.client = P6Client()
            self.client.initialize(username, password, database_name)
            logger.info("✅ P6 tools manager initialized")
        except Exception as e:
            logger.error(f"❌ P6 tools initialization failed: {e}")
            raise

    def initialize_with_auth_key(self, auth_key: str, database_name: str):
        """Initialize with pre-encoded auth key"""
        try:
            self.client = P6Client()
            self.client.initialize_with_auth_key(auth_key, database_name)
            logger.info("✅ P6 tools manager initialized with auth key")
        except Exception as e:
            logger.error(f"❌ P6 tools initialization failed: {e}")
            raise

    def _record_operation(self, operation: str, success: bool = True):
        """Track operation statistics"""
        key = f"{operation}_{'success' if success else 'failure'}"
        self.operation_stats[key] = self.operation_stats.get(key, 0) + 1

    def _validate_operation(self, operation_type: str) -> tuple[bool, str]:
        """Validate operation against security policy"""
        if not self.client:
            return False, "P6 client not initialized"
        return True, "Valid"

    def get_tools(self) -> List[Any]:
        """Get all P6 tools - simplified without async complexity"""

        @tool
        def search_projects(
            search_query: str = Field(description="Natural language search query for projects"),
            status_filter: Optional[str] = Field(default=None, description="Project status filter"),
            max_results: int = Field(default=20, description="Maximum number of results to return")
        ) -> str:
            """Search for projects in Primavera P6 using natural language queries."""
            try:
                return self._search_projects_sync(search_query, status_filter, max_results)
            except Exception as e:
                logger.error(f"Search projects failed: {e}")
                return f"❌ Error searching projects: {str(e)}"

        @tool
        def analyze_project_schedule(
            project_identifier: str = Field(description="Project ID or name to analyze"),
            analysis_type: str = Field(default="overview", description="Type of analysis: overview, critical_path, progress, resources"),
            include_activities: bool = Field(default=True, description="Include activity details in analysis")
        ) -> str:
            """Analyze project schedule with comprehensive insights."""
            try:
                return self._analyze_schedule_sync(project_identifier, analysis_type, include_activities)
            except Exception as e:
                logger.error(f"Schedule analysis failed: {e}")
                return f"❌ Error analyzing schedule: {str(e)}"

        @tool
        def get_activity_status(
            project_identifier: str = Field(description="Project ID or name"),
            activity_filter: Optional[str] = Field(default=None, description="Filter for specific activities"),
            status_type: str = Field(default="all", description="Status type: all, in_progress, completed, not_started, delayed")
        ) -> str:
            """Get detailed activity status information for a project."""
            try:
                return self._get_activity_status_sync(project_identifier, activity_filter, status_type)
            except Exception as e:
                logger.error(f"Activity status failed: {e}")
                return f"❌ Error getting activity status: {str(e)}"

        @tool
        def get_p6_system_status() -> str:
            """Get current P6 system status including connection health."""
            try:
                return self._get_system_status_sync()
            except Exception as e:
                logger.error(f"System status failed: {e}")
                return f"❌ Error getting system status: {str(e)}"

        return [search_projects, analyze_project_schedule, get_activity_status, get_p6_system_status]

    def _search_projects_sync(self, search_query: str, status_filter: Optional[str], max_results: int) -> str:
        """Synchronous project search following working.py patterns"""
        start_time = time.time()
        
        try:
            # Validate operation
            is_valid, message = self._validate_operation("search")
            if not is_valid:
                return f"❌ {message}"

            # Build filter query like working.py
            filter_parts = []
            if "active" in search_query.lower():
                filter_parts.append("Status = 'Active'")
            elif "completed" in search_query.lower():
                filter_parts.append("Status = 'Completed'")
            
            if status_filter:
                filter_parts.append(f"Status = '{status_filter}'")

            if search_query and not any(word in search_query.lower() for word in ['active', 'completed', 'status']):
                filter_parts.append(f"Name contains '{search_query}'")

            filter_query = " and ".join(filter_parts) if filter_parts else None

            # Execute search with fields like working.py
            fields = ["Id", "Name", "Description", "Status", "StartDate", "FinishDate"]
            projects = self.client.get_projects(
                filter_query=filter_query,
                fields=fields,
                order_by="Id asc",
                limit=min(max_results, self.security_config.max_results_per_query)
            )

            duration = (time.time() - start_time) * 1000
            self._record_operation("project_search", True)

            if not projects:
                return f"No projects found matching '{search_query}' (searched in {duration:.1f}ms)"

            # Format results
            response = [f"Found {len(projects)} projects matching '{search_query}' (in {duration:.1f}ms):\n"]
            
            for i, project in enumerate(projects, 1):
                name = project.get('Name', 'Unknown')
                project_id = project.get('Id', 'N/A')
                status = project.get('Status', 'Unknown')
                start_date = project.get('StartDate', 'Not set')
                
                response.append(f"{i}. **{name}** (ID: {project_id})")
                response.append(f"   Status: {status} | Start: {start_date}")
                response.append("")

            # Store context for follow-up queries
            if len(projects) == 1:
                self.current_project_context = projects[0]

            return "\n".join(response)

        except P6APIError as e:
            self._record_operation("project_search", False)
            return f"❌ P6 API error during project search: {e}"
        except Exception as e:
            self._record_operation("project_search", False)
            logger.error(f"Project search failed: {e}")
            return f"❌ Error searching projects: {e}"

    def _analyze_schedule_sync(self, project_identifier: str, analysis_type: str, include_activities: bool) -> str:
        """Synchronous schedule analysis"""
        start_time = time.time()
        
        try:
            # Find project by ID or name
            project = self._find_project_sync(project_identifier)
            if not project:
                return f"❌ Project '{project_identifier}' not found"

            project_id = project['Id']
            project_name = project.get('Name', 'Unknown')

            # Get activities if requested
            activities = []
            if include_activities:
                activity_fields = [
                    "Id", "Name", "Status", "ActualStartDate", "ActualFinishDate",
                    "PlannedStartDate", "PlannedFinishDate", "PercentComplete",
                    "Duration", "RemainingDuration"
                ]
                activities = self.client.get_activities(
                    project_id=project_id,
                    fields=activity_fields,
                    limit=100
                )

            duration = (time.time() - start_time) * 1000
            self._record_operation("schedule_analysis", True)

            # Format analysis results
            response = [
                f"📊 Schedule Analysis for {project_name}",
                f"Query completed in {duration:.1f}ms\n",
                f"**Project Details:**",
                f"- ID: {project_id}",
                f"- Status: {project.get('Status', 'Unknown')}",
                f"- Start Date: {project.get('StartDate', 'Not set')}",
                f"- Finish Date: {project.get('FinishDate', 'Not set')}\n"
            ]

            if activities:
                response.append(f"**Activities Summary:**")
                response.append(f"- Total Activities: {len(activities)}")
                
                # Count by status
                status_counts = {}
                for activity in activities:
                    status = activity.get('Status', 'Unknown')
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                for status, count in status_counts.items():
                    response.append(f"- {status}: {count}")

            return "\n".join(response)

        except Exception as e:
            self._record_operation("schedule_analysis", False)
            logger.error(f"Schedule analysis failed: {e}")
            return f"❌ Error analyzing schedule: {e}"

    def _find_project_sync(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Find project by ID or name - synchronous"""
        try:
            # Try exact ID match first
            projects = self.client.get_projects(
                filter_query=f"Id = '{identifier}'",
                fields=["Id", "Name", "Description", "Status", "StartDate", "FinishDate"],
                limit=1
            )
            
            if projects:
                return projects[0]

            # Try name match
            projects = self.client.get_projects(
                filter_query=f"Name = '{identifier}'",
                fields=["Id", "Name", "Description", "Status", "StartDate", "FinishDate"],
                limit=1
            )
            
            if projects:
                return projects[0]

            # Try partial name match
            projects = self.client.get_projects(
                filter_query=f"Name contains '{identifier}'",
                fields=["Id", "Name", "Description", "Status", "StartDate", "FinishDate"],
                limit=5
            )
            
            return projects[0] if projects else None

        except Exception as e:
            logger.error(f"Error finding project '{identifier}': {e}")
            return None

    def _get_activity_status_sync(self, project_identifier: str, activity_filter: Optional[str], status_type: str) -> str:
        """Get activity status - synchronous implementation"""
        try:
            project = self._find_project_sync(project_identifier)
            if not project:
                return f"❌ Project '{project_identifier}' not found"

            activities = self.client.get_activities(
                project_id=project['Id'],
                fields=["Id", "Name", "Status", "PercentComplete", "ActualStartDate", "ActualFinishDate"],
                limit=500
            )

            if not activities:
                return f"No activities found for project {project.get('Name', project_identifier)}"

            # Filter by status type if specified
            if status_type != "all":
                status_map = {
                    "in_progress": ["In Progress", "Started"],
                    "completed": ["Completed", "Finished"],
                    "not_started": ["Not Started", "Planning"],
                    "delayed": ["Delayed", "Behind Schedule"]
                }
                
                if status_type in status_map:
                    activities = [a for a in activities if a.get('Status', '') in status_map[status_type]]

            response = [
                f"📋 Activity Status for {project.get('Name', 'Unknown Project')}",
                f"Found {len(activities)} activities\n"
            ]

            for i, activity in enumerate(activities[:20], 1):  # Limit to first 20
                name = activity.get('Name', 'Unknown')
                status = activity.get('Status', 'Unknown')
                percent = activity.get('PercentComplete', 0)
                
                response.append(f"{i}. **{name}**")
                response.append(f"   Status: {status} | Progress: {percent}%")

            if len(activities) > 20:
                response.append(f"\n... and {len(activities) - 20} more activities")

            return "\n".join(response)

        except Exception as e:
            logger.error(f"Activity status failed: {e}")
            return f"❌ Error getting activity status: {e}"

    def _get_system_status_sync(self) -> str:
        """Get P6 system status - synchronous"""
        try:
            if not self.client:
                return "❌ P6 client not initialized"

            # Simple connectivity test like working.py
            start_time = time.time()
            projects = self.client.get_projects(limit=1)
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
        """Return stub tools when not initialized"""
        @tool
        def p6_not_ready() -> str:
            """Inform the user that P6 is not initialized"""
            return "Primavera P6 connection not established yet. Use `connect to P6` first."

        return [p6_not_ready]

# Global P6 tools manager instance
p6_tools_manager = P6ToolsManager()
