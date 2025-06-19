# tools/p6_tools.py - Enhanced Tools with FilterBuilder Integration

import logging
import json
import time
from typing import Optional, List, Any, Dict, Union
from dataclasses import dataclass
from langchain_core.tools import tool
from pydantic import Field
from tools.p6_filter_mapper import P6FilterMapper
from tools.p6_client import P6Client, P6ClientConfig, P6AuthenticationError, P6APIError, FilterBuilder

logger = logging.getLogger("p6_tools")

@dataclass
class P6SecurityConfig:
    """Security configuration for P6 operations"""
    max_results_per_query: int = 1000
    read_only_mode: bool = True
    enable_nl_filtering: bool = True
    total_filters: int = 5

class P6ToolsManager:
    """Enhanced P6 tools manager with FilterBuilder integration"""

    def __init__(self):
        self.client: Optional[P6Client] = None
        self.security_config = P6SecurityConfig()
        self.operation_stats: Dict[str, int] = {}
        self.current_project_context: Optional[Dict[str, Any]] = None
        self.filter_mapper = P6FilterMapper()
        
        # Track filter usage statistics
        self.filter_stats = {
            "total_filters_applied": 0,
            "filters_by_type": {},
            "filter_errors": 0,
            "total_filters": 0
        }

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
        """Get all P6 tools with enhanced FilterBuilder support"""
        @tool
        def search_projects(
            search_query: str = Field(description="Natural language search query for projects"),
            status_filter: Optional[str] = Field(default=None, description="Project status filter"),
            max_results: int = Field(default=20, description="Maximum number of results to return"),
            order_by_field: Optional[str] = Field(default=None, description="Field to order results by"),
            order_direction: Optional[str] = Field(default="asc", description="Order direction (asc or desc)"),
            additional_filters: Optional[str] = Field(default=None, description="Additional filter criteria in natural language")
        ) -> str:
            """
            Search for projects in Primavera P6 using natural language queries with enhanced filtering.
            Supports complex filters, ordering, and field selection.
            """
            try:
                return self._search_projects_sync(
                    search_query, 
                    status_filter, 
                    max_results, 
                    order_by_field, 
                    order_direction, 
                    additional_filters
                )
            except Exception as e:
                logger.error(f"Search projects failed: {e}")
                return f"❌ Error searching projects: {str(e)}"

        @tool
        def build_complex_filter(
            filter_description: str = Field(..., description="Description of the filter in natural language"),
            entity_type: str = Field(default="Project", description="Entity type (Project, Activity, Resource)"),
            fields_to_return: Optional[str] = Field(default=None, description="Comma-separated list of fields to return"),
            order_by: Optional[str] = Field(default=None, description="Field and direction for ordering (e.g., 'Name desc')")
        ) -> str:
            """Build a complex P6 filter from natural language description with enhanced validation"""
            
            # ✅ IMMEDIATE VALIDATION: Check for FieldInfo objects at entry point
            params = {
                'filter_description': filter_description,
                'entity_type': entity_type, 
                'fields_to_return': fields_to_return,
                'order_by': order_by
            }
            
            # Validate all parameters before processing
            validated_params = {}
            for key, value in params.items():
                if hasattr(value, '__class__') and 'FieldInfo' in str(type(value)):
                    logger.error(f"❌ CRITICAL: Received FieldInfo object for {key}: {type(value)}")
                    return f"❌ Parameter validation error: {key} received as FieldInfo object instead of string value"
                validated_params[key] = value
            
            try:
                return p6_tools_manager._build_complex_filter_sync(**validated_params)
            except Exception as e:
                logger.error(f"Build complex filter failed: {e}")
                return f"❌ Error building complex filter: {str(e)}"

        @tool
        def analyze_project_schedule(
            project_identifier: str = Field(description="Project ID or name to analyze"),
            analysis_type: str = Field(default="overview", description="Type of analysis: overview, critical_path, progress, resources"),
            include_activities: bool = Field(default=True, description="Include activity details in analysis"),
            activity_filter: Optional[str] = Field(default=None, description="Filter for activities in natural language")
        ) -> str:
            """
            Analyze project schedule with comprehensive insights and filtering capabilities.
            """
            try:
                return self._analyze_schedule_sync(project_identifier, analysis_type, include_activities, activity_filter)
            except Exception as e:
                logger.error(f"Schedule analysis failed: {e}")
                return f"❌ Error analyzing schedule: {str(e)}"

        @tool
        def get_activity_status(
            project_identifier: str = Field(description="Project ID or name"),
            activity_filter: Optional[str] = Field(default=None, description="Filter for specific activities in natural language"),
            status_type: str = Field(default="all", description="Status type: all, in_progress, completed, not_started, delayed"),
            order_by: Optional[str] = Field(default=None, description="Field to order activities by (e.g., 'StartDate asc')")
        ) -> str:
            """
            Get detailed activity status information for a project with enhanced filtering and ordering.
            """
            try:
                return self._get_activity_status_sync(project_identifier, activity_filter, status_type, order_by)
            except Exception as e:
                logger.error(f"Activity status failed: {e}")
                return f"❌ Error getting activity status: {str(e)}"

        @tool
        def get_p6_system_status() -> str:
            """Get current P6 system status including connection health and filter statistics."""
            try:
                return self._get_system_status_sync()
            except Exception as e:
                logger.error(f"System status failed: {e}")
                return f"❌ Error getting system status: {str(e)}"

        return [search_projects, build_complex_filter, analyze_project_schedule, get_activity_status, get_p6_system_status]

    def _search_projects_sync(
        self,
        search_query: str,
        status_filter: Optional[str],
        max_results: int,
        order_by_field: Optional[str] = None,
        order_direction: Optional[str] = "asc",
        additional_filters: Optional[str] = None
    ) -> str:
        """Enhanced project search using FilterBuilder"""
        start_time = time.time()
        is_valid, message = self._validate_operation("search")
        if not is_valid:
            return f"❌ {message}"

        try:
            # 2. Create filter builder
            filter_builder = FilterBuilder()
            filter_explanation = ""

            # Try using the enhanced filter mapper first
            try:
                # Add base search query
                # ✅ FIX: Get a FilterBuilder object from the mapper
                nl_builder = self.filter_mapper.build_filter(search_query, status_filter) # [6]
                # ✅ FIX: Extract the filter string from the FilterBuilder for methods expecting a string
                nl_filter_string = nl_builder.build_filter() # [3]

                if nl_filter_string: # Check if the filter string is not empty
                    self._add_raw_filter_to_builder(filter_builder, nl_filter_string) # Pass the string
                    filter_explanation = self.filter_mapper.get_filter_explanation(nl_filter_string) # Pass the string [6]

                # Add additional filters
                if additional_filters:
                    additional_nl_builder = self.filter_mapper.build_filter(additional_filters) # [6]
                    additional_nl_filter_string = additional_nl_builder.build_filter() # [3]

                    if additional_nl_filter_string: # Check if the additional filter string is not empty
                        if filter_builder._conditions: # Check if there are existing conditions in filter_builder
                            filter_builder.and_() # [3]
                        self._add_raw_filter_to_builder(filter_builder, additional_nl_filter_string) # Pass the string
                        filter_explanation += "\n" + self.filter_mapper.get_filter_explanation(additional_nl_filter_string) # Pass the string [6]

                self.filter_stats["total_filters"] += 1

            except Exception as e:
                # Mapper failures should NEVER break the tool – just log & fall back
                logger.warning(f"⚠️ Filter mapper error → creating simple filter: {e}")
                self.filter_stats["errors"] += 1
                filter_explanation = "Using simplified filter due to mapper error."

            # Fallback to simple name-based search if no conditions were added by the mapper
            if not filter_builder._conditions and search_query:
                 filter_builder.field("Name").contains(search_query) # [3]

            # Add status filter if provided (Note: This might be redundant if the mapper already handles status_filter)
            if status_filter:
                if filter_builder._conditions:
                    filter_builder.and_()
                filter_builder.field("Status").eq(status_filter)

            # Add ordering if specified
            if order_by_field:
                is_desc = order_direction and order_direction.lower() == "desc"
                filter_builder.order_by(order_by_field, desc=is_desc) # [3]

            # 3. Execute the P6 query with default fields
            fields = ["Id", "Name", "Description", "Status", "StartDate", "FinishDate"]
            filter_builder.select(*fields) # [3]

            projects = self.client.get_projects(
                filter_builder=filter_builder,
                limit=min(max_results, self.security_config.max_results_per_query),
            )

            duration = (time.time() - start_time) * 1000
            self._record_operation("project_search", True)
            # 4. Format the response
            if not projects:
                return (
                    f"No projects found matching '{search_query}' "
                    f"(searched in {duration:.1f} ms)"
                )

            response: list[str] = [
                f"Found {len(projects)} projects (in {duration:.1f} ms)",
                f"Filter applied: {filter_explanation}",
                "",
            ]

            for i, project in enumerate(projects, 1):
                response.append(
                    f"{i}. **{project.get('Name', 'Unknown')}** "
                    f"(ID {project.get('Id', 'N/A')})"
                )
                response.append(
                    f" Status: {project.get('Status', 'Unknown')} | "
                    f"Start: {project.get('StartDate', 'Not set')}"
                )
                response.append("")

            # Store project context if only one result
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

    def _build_complex_filter_sync(
        self,
        filter_description: str,
        entity_type: str = "Project",
        fields_to_return: Optional[str] = None,
        order_by: Optional[str] = None
    ) -> str:
        """Build a complex filter from natural language description with bulletproof parameter handling"""
        try:
            # ✅ BULLETPROOF: Convert all parameters to proper types
            filter_description = self._ensure_string_parameter(filter_description, "filter_description")
            entity_type = self._ensure_string_parameter(entity_type, "entity_type", default="Project")
            fields_to_return = self._ensure_string_parameter(fields_to_return, "fields_to_return", allow_none=True)
            order_by = self._ensure_string_parameter(order_by, "order_by", allow_none=True)
            
            filter_builder = FilterBuilder()
                
            # Parse the natural language description into a P6 filter
            try:
                # Use the mapper to build a filter string
                filter_string = self.filter_mapper.build_filter(filter_description)
                if filter_string:
                    # Add the raw filter to the builder
                    self._add_raw_filter_to_builder(filter_builder, filter_string)
                else:
                    # If no filter was built, provide guidance
                    return (
                        "❌ Couldn't extract a valid filter from the description. "
                        "Please use more specific terms like project names, dates, or status values."
                    )
            except Exception as e:
                logger.warning(f"Filter mapping failed: {e}")
                return (
                    f"❌ Error building filter: {str(e)}\n\n"
                    "Please use more explicit criteria like:\n"
                    "- 'Status equals Active'\n"
                    "- 'Name contains Bridge'\n"
                    "- 'StartDate greater than 2024-01-01'"
                )
            
            # Add field selection
            if fields_to_return:
                # Ensure fields_to_return is a string
                if not isinstance(fields_to_return, str):
                    logger.warning(f"fields_to_return is not a string: {type(fields_to_return)}")
                    # Try to get a string value if it's a Field object
                    if hasattr(fields_to_return, 'default'):
                        fields_to_return = fields_to_return.default
                    else:
                        fields_to_return = str(fields_to_return)
                
                if fields_to_return:
                    fields = [f.strip() for f in fields_to_return.split(",")]
                    filter_builder.select(*fields)
            
            # Add ordering
            if order_by:
                # Ensure order_by is a string
                if not isinstance(order_by, str):
                    logger.warning(f"order_by is not a string: {type(order_by)}")
                    # Try to get a string value if it's a Field object
                    if hasattr(order_by, 'default'):
                        order_by = order_by.default
                    else:
                        order_by = str(order_by)
                
                # Only proceed if order_by is a string and contains a space
                if isinstance(order_by, str) and " " in order_by:
                    parts = order_by.split()
                    field = parts[0]
                    is_desc = len(parts) > 1 and parts[1].lower() == "desc"
                    filter_builder.order_by(field, desc=is_desc)
            
            # Build the query params
            query_params = filter_builder.build_query_params()
            
            # Format the response
            result = [f"✅ Filter for {entity_type} created successfully:"]
            if "Filter" in query_params:
                result.append("\n**Filter:**")
                result.append(f"``````")
            if "Fields" in query_params:
                result.append("\n**Fields:**")
                result.append(f"``````")
            if "OrderBy" in query_params:
                result.append("\n**Order By:**")
                result.append(f"``````")
            result.append("\n**Natural Language Explanation:**")
            explanation = self.filter_mapper.get_filter_explanation(query_params.get("Filter", ""))
            result.append(explanation)
            result.append("\n**Usage Instructions:**")
            result.append("You can use this filter with other P6 tools like search_projects or get_activity_status.")
            return "\n".join(result)
        except Exception as e:
            logger.error(f"Building complex filter failed: {e}")
            return f"❌ Error building complex filter: {str(e)}"

    def _ensure_string_parameter(self, value: Any, param_name: str, default: str = None, allow_none: bool = False) -> Optional[str]:
        """Ensure parameter is a string, handling FieldInfo objects and other edge cases"""
        
        # Handle None values
        if value is None:
            if allow_none:
                return None
            if default is not None:
                return default
            raise ValueError(f"Parameter {param_name} cannot be None")
        
        # Handle FieldInfo objects (the main issue)
        if hasattr(value, '__class__') and 'FieldInfo' in str(type(value)):
            logger.warning(f"⚠️ Received FieldInfo object for {param_name}, extracting default value")
            if hasattr(value, 'default') and value.default is not None:
                extracted_value = value.default
                logger.debug(f"✅ Extracted default value: {extracted_value}")
                return self._ensure_string_parameter(extracted_value, param_name, default, allow_none)
            elif default is not None:
                return default
            elif allow_none:
                return None
            else:
                raise ValueError(f"FieldInfo object for {param_name} has no usable default value")
        
        # Handle already valid strings
        if isinstance(value, str):
            return value.strip() if value.strip() else (default if default is not None else (None if allow_none else ""))
        
        # Handle other types by converting to string
        try:
            converted = str(value).strip()
            logger.debug(f"✅ Converted {param_name} from {type(value)} to string: {converted}")
            return converted if converted else (default if default is not None else (None if allow_none else ""))
        except Exception as e:
            logger.error(f"❌ Failed to convert {param_name} to string: {e}")
            if default is not None:
                return default
            elif allow_none:
                return None
            else:
                raise ValueError(f"Cannot convert {param_name} to string: {e}")

    def _analyze_schedule_sync(
        self, 
        project_identifier: str, 
        analysis_type: str, 
        include_activities: bool,
        activity_filter: Optional[str] = None
    ) -> str:
        """Synchronous schedule analysis with enhanced filtering"""
        start_time = time.time()

        try:
            # Find project by ID or name
            project = self._find_project_sync(project_identifier)
            if not project:
                return f"❌ Project '{project_identifier}' not found"

            project_id = project['Id']
            project_name = project.get('Name', 'Unknown')

            # Get activities if requested with optional filtering
            activities = []
            if include_activities:
                activity_fields = [
                    "Id", "Name", "Status", "ActualStartDate", "ActualFinishDate",
                    "PlannedStartDate", "PlannedFinishDate", "PercentComplete",
                    "Duration", "RemainingDuration"
                ]
                
                # Create activity filter builder
                filter_builder = FilterBuilder().select(*activity_fields)
                
                # Add custom activity filter if provided
                if activity_filter:
                    try:
                        nl_filter = self.filter_mapper.build_filter(activity_filter)
                        if nl_filter:
                            self._add_raw_filter_to_builder(filter_builder, nl_filter)
                    except Exception as filter_err:
                        logger.warning(f"Activity filter error: {filter_err}")
                
                # Add ordering based on analysis type
                if analysis_type == "critical_path":
                    filter_builder.order_by("TotalFloat")
                elif analysis_type == "progress":
                    filter_builder.order_by("PercentComplete", desc=True)
                else:
                    filter_builder.order_by("StartDate")
                
                activities = self.client.get_activities(
                    project_id=project_id,
                    filter_builder=filter_builder,
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
                
                # Add analysis-specific details
                if analysis_type == "critical_path":
                    critical_activities = [a for a in activities if a.get('TotalFloat', 999) < 5]
                    response.append(f"\n**Critical Path Analysis:**")
                    response.append(f"- Critical Activities: {len(critical_activities)}")
                    
                    if critical_activities:
                        response.append(f"\nTop Critical Activities:")
                        for i, activity in enumerate(critical_activities[:5], 1):
                            response.append(f"{i}. {activity.get('Name')} (Float: {activity.get('TotalFloat', 'N/A')})")
                
                elif analysis_type == "progress":
                    completed = sum(1 for a in activities if a.get('PercentComplete', 0) == 100)
                    not_started = sum(1 for a in activities if a.get('PercentComplete', 0) == 0)
                    in_progress = len(activities) - completed - not_started
                    
                    avg_progress = sum(a.get('PercentComplete', 0) for a in activities) / len(activities) if activities else 0
                    
                    response.append(f"\n**Progress Analysis:**")
                    response.append(f"- Completed: {completed} ({completed/len(activities)*100:.1f}%)")
                    response.append(f"- In Progress: {in_progress} ({in_progress/len(activities)*100:.1f}%)")
                    response.append(f"- Not Started: {not_started} ({not_started/len(activities)*100:.1f}%)")
                    response.append(f"- Average Completion: {avg_progress:.1f}%")

            return "\n".join(response)

        except Exception as e:
            self._record_operation("schedule_analysis", False)
            logger.error(f"Schedule analysis failed: {e}")
            return f"❌ Error analyzing schedule: {e}"

    def _find_project_sync(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Find project by ID or name with enhanced search"""
        try:
            # Try exact ID match first
            filter_builder = FilterBuilder().field("Id").eq(identifier)
            projects = self.client.get_projects(
                filter_builder=filter_builder,
                limit=1
            )

            if projects:
                return projects[0]

            # Try exact name match
            filter_builder = FilterBuilder().field("Name").eq(identifier)
            projects = self.client.get_projects(
                filter_builder=filter_builder,
                limit=1
            )

            if projects:
                return projects[0]

            # Try partial name match
            filter_builder = FilterBuilder().field("Name").contains(identifier)
            projects = self.client.get_projects(
                filter_builder=filter_builder,
                limit=5
            )

            return projects[0] if projects else None
        except Exception as e:
            logger.error(f"Error finding project '{identifier}': {e}")
            return None

    def _get_activity_status_sync(
        self, 
        project_identifier: str, 
        activity_filter: Optional[str], 
        status_type: str,
        order_by: Optional[str] = None
    ) -> str:
        """Get activity status with enhanced filtering"""
        try:
            project = self._find_project_sync(project_identifier)
            if not project:
                return f"❌ Project '{project_identifier}' not found"
            
            # Create filter builder with field selection
            filter_builder = FilterBuilder().select(
                "Id", "Name", "Status", "PercentComplete", 
                "ActualStartDate", "ActualFinishDate"
            )
            
            # Add status type filter
            if status_type != "all":
                status_map = {
                    "in_progress": ["In Progress", "Started"],
                    "completed": ["Completed", "Finished"],
                    "not_started": ["Not Started", "Planning"],
                    "delayed": ["Delayed", "Behind Schedule"]
                }
                
                if status_type in status_map:
                    statuses = status_map[status_type]
                    if len(statuses) == 1:
                        filter_builder.field("Status").eq(statuses[0])
                    else:
                        filter_builder.field("Status").in_(*statuses)
            
            # Add custom activity filter if provided
            if activity_filter:
                try:
                    nl_filter = self.filter_mapper.build_filter(activity_filter)
                    if nl_filter:
                        if filter_builder._conditions:
                            filter_builder.and_()
                        self._add_raw_filter_to_builder(filter_builder, nl_filter)
                except Exception as filter_err:
                    logger.warning(f"Activity filter error: {filter_err}")
            
            # Add ordering if specified
            if order_by:
                parts = order_by.split()
                field = parts[0]
                is_desc = len(parts) > 1 and parts[1].lower() == "desc"
                filter_builder.order_by(field, desc=is_desc)
            else:
                # Default ordering
                filter_builder.order_by("Name")
            
            activities = self.client.get_activities(
                project_id=project['Id'],
                filter_builder=filter_builder,
                limit=500
            )

            if not activities:
                return f"No activities found for project {project.get('Name', project_identifier)}"

            response = [
                f"📋 Activity Status for {project.get('Name', 'Unknown Project')}",
                f"Found {len(activities)} activities\n"
            ]

            for i, activity in enumerate(activities[:20], 1):  # Limit to first 20
                name = activity.get('Name', 'Unknown')
                status = activity.get('Status', 'Unknown')
                percent = activity.get('PercentComplete', 0)
                response.append(f"{i}. **{name}**")
                response.append(f" Status: {status} | Progress: {percent}%")

            if len(activities) > 20:
                response.append(f"\n... and {len(activities) - 20} more activities")

            return "\n".join(response)
        except Exception as e:
            logger.error(f"Activity status failed: {e}")
            return f"❌ Error getting activity status: {e}"

    def _get_system_status_sync(self) -> str:
        """Get P6 system status with filter statistics"""
        try:
            if not self.client:
                return "❌ P6 client not initialized"

            # Simple connectivity test
            start_time = time.time()
            projects = self.client.get_projects(limit=1)
            response_time = (time.time() - start_time) * 1000

            return f"""🔌 P6 System Status:

✅ **Connection**: Healthy
⚡ **Response Time**: {response_time:.1f} ms
📊 **Operations**: {sum(self.operation_stats.values())} total
🔍 **Filters**: {self.filter_stats['total_filters']} applied ({self.filter_stats['errors']} errors)
🔒 **Security**: Read-only mode {'' if self.security_config.read_only_mode else 'in'}active
📁 **Project Context**: {self.current_project_context.get('Name') if self.current_project_context else 'None'}
🛠️ **Filter Builder**: Available and operational
"""
        except Exception as e:
            return f"❌ P6 system status check failed: {e}"

    def _add_raw_filter_to_builder(self, filter_builder: FilterBuilder, filter_string: str) -> None:
        """Add a raw filter string to a filter builder instance"""
        if not filter_string:
            return
        
        # Simple approach: just add the raw condition
        # This is a fallback when we can't parse the filter structure
        filter_builder._add_raw_condition(filter_string)
        
        # Track stats
        self.filter_stats["total_filters_applied"] += 1

    def get_stub_tools(self):
        """Return stub tools when not initialized"""
        @tool
        def p6_not_ready() -> str:
            """Inform the user that P6 is not initialized"""
            return "Primavera P6 connection not established yet. Use `connect to P6` first."

        return [p6_not_ready]

# Global P6 tools manager instance
p6_tools_manager = P6ToolsManager()