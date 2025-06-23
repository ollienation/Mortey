# tools/p6_client.py - Enhanced P6 Client with Comprehensive Filter Builder

import requests
import logging
import time
import json
import base64
import os
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("p6_client")

@dataclass
class P6ClientConfig:
    """Configuration for P6 client"""
    base_url: str = "http://primaverap6restapi.primaplan.com/p6ws/restapi"
    timeout: float = 15.0
    max_retries: int = 3

class P6AuthenticationError(Exception):
    """Authentication-specific error for P6 operations"""
    pass

class P6APIError(Exception):
    """General P6 API error"""
    pass

class InvalidFilterError(Exception):
    """Invalid filter construction error"""
    pass

class P6Operator(Enum):
    """P6 filter operators following Oracle grammar"""
    EQ = ":eq:"
    NEQ = ":neq:"
    GT = ":gt:"
    GTE = ":gte:"
    LT = ":lt:"
    LTE = ":lte:"
    LIKE = ":like:"
    IN = ":in:"
    IS_NULL = ":isnull:"
    IS_NOT_NULL = ":isnotnull:"
    AND = ":and:"
    OR = ":or:"

class FilterBuilder:
    """
    Comprehensive P6 filter builder with fluent API and grammar validation.
    
    Usage:
        filter_obj = (FilterBuilder()
            .field("Status").eq("Active")
            .and_()
            .field("StartDate").gte("2024-01-01")
            .order_by("Name", desc=True)
            .select("Name", "Status", "StartDate"))
    """
    
    def __init__(self):
        self._conditions: List[str] = []
        self._order_by: List[str] = []
        self._fields: List[str] = []
        self._current_field: Optional[str] = None
        self._in_group: bool = False
        self._group_stack: List[List[str]] = []
        
    def field(self, field_name: str) -> 'FilterBuilder':
        """Set the current field for operations"""
        if not field_name or not isinstance(field_name, str):
            raise InvalidFilterError(f"Invalid field name: {field_name}")
        self._current_field = field_name
        return self
        
    def eq(self, value: Any) -> 'FilterBuilder':
        """Add equality condition"""
        return self._add_condition(P6Operator.EQ, value)
        
    def neq(self, value: Any) -> 'FilterBuilder':
        """Add not-equal condition"""
        return self._add_condition(P6Operator.NEQ, value)
        
    def gt(self, value: Any) -> 'FilterBuilder':
        """Add greater-than condition"""
        return self._add_condition(P6Operator.GT, value)
        
    def gte(self, value: Any) -> 'FilterBuilder':
        """Add greater-than-or-equal condition"""
        return self._add_condition(P6Operator.GTE, value)
        
    def lt(self, value: Any) -> 'FilterBuilder':
        """Add less-than condition"""
        return self._add_condition(P6Operator.LT, value)
        
    def lte(self, value: Any) -> 'FilterBuilder':
        """Add less-than-or-equal condition"""
        return self._add_condition(P6Operator.LTE, value)
        
    def like(self, pattern: str) -> 'FilterBuilder':
        """Add LIKE condition with wildcards"""
        if not isinstance(pattern, str):
            raise InvalidFilterError("LIKE pattern must be a string")
        return self._add_condition(P6Operator.LIKE, pattern)
        
    def contains(self, text: str) -> 'FilterBuilder':
        """Add contains condition (convenience for LIKE %text%)"""
        if not isinstance(text, str):
            raise InvalidFilterError("Contains text must be a string")
        return self._add_condition(P6Operator.LIKE, f"%{text}%")
        
    def starts_with(self, prefix: str) -> 'FilterBuilder':
        """Add starts-with condition (convenience for LIKE prefix%)"""
        if not isinstance(prefix, str):
            raise InvalidFilterError("Prefix must be a string")
        return self._add_condition(P6Operator.LIKE, f"{prefix}%")
        
    def ends_with(self, suffix: str) -> 'FilterBuilder':
        """Add ends-with condition (convenience for LIKE %suffix)"""
        if not isinstance(suffix, str):
            raise InvalidFilterError("Suffix must be a string")
        return self._add_condition(P6Operator.LIKE, f"%{suffix}")
        
    def in_(self, *values) -> 'FilterBuilder':
        """Add IN condition"""
        if not values:
            raise InvalidFilterError("IN operator requires at least one value")
        return self._add_in_condition(values)
        
    def not_in(self, *values) -> 'FilterBuilder':
        """Add NOT IN condition (negated IN)"""
        if not values:
            raise InvalidFilterError("NOT IN operator requires at least one value")
        # P6 doesn't have direct NOT IN, so we use negation
        condition = self._build_in_condition(values)
        return self._add_raw_condition(f"NOT ({condition})")
        
    def is_null(self) -> 'FilterBuilder':
        """Add IS NULL condition"""
        return self._add_condition(P6Operator.IS_NULL, None)
        
    def is_not_null(self) -> 'FilterBuilder':
        """Add IS NOT NULL condition"""
        return self._add_condition(P6Operator.IS_NOT_NULL, None)
        
    def between(self, start_value: Any, end_value: Any) -> 'FilterBuilder':
        """Add BETWEEN condition (implemented as field >= start AND field <= end)"""
        if self._current_field is None:
            raise InvalidFilterError("No field set for BETWEEN operation")
        
        start_condition = f"{self._current_field}{P6Operator.GTE.value}'{self._escape_value(start_value)}'"
        end_condition = f"{self._current_field}{P6Operator.LTE.value}'{self._escape_value(end_value)}'"
        combined = f"({start_condition} {P6Operator.AND.value} {end_condition})"
        
        return self._add_raw_condition(combined)
        
    def and_(self) -> 'FilterBuilder':
        """Add AND logical operator"""
        if self._conditions:
            self._conditions.append(P6Operator.AND.value)
        return self
        
    def or_(self) -> 'FilterBuilder':
        """Add OR logical operator"""
        if self._conditions:
            self._conditions.append(P6Operator.OR.value)
        return self
        
    def group_start(self) -> 'FilterBuilder':
        """Start a grouped condition"""
        self._group_stack.append(self._conditions.copy())
        self._conditions = []
        self._in_group = True
        return self
        
    def group_end(self) -> 'FilterBuilder':
        """End a grouped condition"""
        if not self._group_stack:
            raise InvalidFilterError("No group to close")
            
        group_conditions = " ".join(self._conditions)
        self._conditions = self._group_stack.pop()
        
        if group_conditions:
            self._conditions.append(f"({group_conditions})")
            
        self._in_group = len(self._group_stack) > 0
        return self
        
    def order_by(self, field: str, desc: bool = False) -> 'FilterBuilder':
        """Add ORDER BY clause"""
        if not field or not isinstance(field, str):
            raise InvalidFilterError(f"Invalid field name for ORDER BY: {field}")
            
        direction = "desc" if desc else "asc"
        self._order_by.append(f"{field} {direction}")
        return self
        
    def select(self, *fields) -> 'FilterBuilder':
        """Set fields to select"""
        if not fields:
            raise InvalidFilterError("SELECT requires at least one field")
            
        for field in fields:
            if not isinstance(field, str) or not field:
                raise InvalidFilterError(f"Invalid field name: {field}")
                
        self._fields.extend(fields)
        return self
        
    def clear_select(self) -> 'FilterBuilder':
        """Clear field selection"""
        self._fields = []
        return self
        
    def clear_order(self) -> 'FilterBuilder':
        """Clear order by clauses"""
        self._order_by = []
        return self
        
    def build_filter(self) -> Optional[str]:
        """Build the filter string"""
        if not self._conditions:
            return None
            
        filter_str = " ".join(self._conditions)
        self._validate_filter_syntax(filter_str)
        return filter_str
        
    def build_order_by(self) -> Optional[str]:
        """Build the ORDER BY string"""
        return ", ".join(self._order_by) if self._order_by else None
        
    def build_fields(self) -> Optional[str]:
        """Build the fields string"""
        return ",".join(self._fields) if self._fields else None
        
    def build_query_params(self) -> Dict[str, str]:
        """Build complete query parameters"""
        params = {}
        
        filter_str = self.build_filter()
        if filter_str:
            params["Filter"] = filter_str
            
        order_str = self.build_order_by()
        if order_str:
            params["OrderBy"] = order_str
            
        fields_str = self.build_fields()
        if fields_str:
            params["Fields"] = fields_str
            
        return params
        
    def _add_condition(self, operator: P6Operator, value: Any) -> 'FilterBuilder':
        """Add a condition with field and operator"""
        if self._current_field is None:
            raise InvalidFilterError(f"No field set for {operator.value} operation")
            
        if operator in [P6Operator.IS_NULL, P6Operator.IS_NOT_NULL]:
            condition = f"{self._current_field} {operator.value}"
        else:
            escaped_value = self._escape_value(value)
            condition = f"{self._current_field}{operator.value}'{escaped_value}'"
            
        self._conditions.append(condition)
        return self
        
    def _add_in_condition(self, values: tuple) -> 'FilterBuilder':
        """Add IN condition"""
        if self._current_field is None:
            raise InvalidFilterError("No field set for IN operation")
            
        condition = self._build_in_condition(values)
        return self._add_raw_condition(condition)
        
    def _build_in_condition(self, values: tuple) -> str:
        """Build IN condition string"""
        escaped_values = [str(self._escape_value(v)) for v in values]
        values_str = ",".join(escaped_values)
        return f"{self._current_field}{P6Operator.IN.value}({values_str})"
        
    def _add_raw_condition(self, condition: str) -> 'FilterBuilder':
        """Add a raw condition string"""
        self._conditions.append(condition)
        return self
        
    def _escape_value(self, value: Any) -> str:
        """Escape value for P6 filter syntax"""
        if value is None:
            return ""
        
        # Convert to string and escape single quotes
        str_value = str(value)
        return str_value.replace("'", "''")
        
    def _validate_filter_syntax(self, filter_str: str) -> None:
        """Validate filter syntax against P6 grammar"""
        if not filter_str:
            return
            
        # Check for unmatched parentheses
        open_count = filter_str.count("(")
        close_count = filter_str.count(")")
        if open_count != close_count:
            raise InvalidFilterError(f"Unmatched parentheses in filter: {filter_str}")
            
        # Check for invalid operator sequences
        invalid_sequences = [":and: :and:", ":or: :or:", ":and: :or:", ":or: :and:"]
        for seq in invalid_sequences:
            if seq in filter_str:
                raise InvalidFilterError(f"Invalid operator sequence '{seq}' in filter")
                
        # Check for empty conditions
        if filter_str.strip().startswith((":and:", ":or:")) or filter_str.strip().endswith((":and:", ":or:")):
            raise InvalidFilterError("Filter cannot start or end with logical operator")
            
    @staticmethod
    def create_date_range(field: str, start_date: str, end_date: str) -> 'FilterBuilder':
        """Helper to create date range filter"""
        return (FilterBuilder()
                .field(field).gte(start_date)
                .and_()
                .field(field).lte(end_date))
                
    @staticmethod
    def create_status_filter(*statuses: str) -> 'FilterBuilder':
        """Helper to create status filter"""
        if len(statuses) == 1:
            return FilterBuilder().field("Status").eq(statuses[0])
        else:
            return FilterBuilder().field("Status").in_(*statuses)
            
    @staticmethod
    def create_project_search(name_pattern: str = None, status: str = None, 
                            start_after: str = None, end_before: str = None) -> 'FilterBuilder':
        """Helper to create common project search filter"""
        builder = FilterBuilder()
        conditions_added = False
        
        if name_pattern:
            builder.field("Name").contains(name_pattern)
            conditions_added = True
            
        if status:
            if conditions_added:
                builder.and_()
            builder.field("Status").eq(status)
            conditions_added = True
            
        if start_after:
            if conditions_added:
                builder.and_()
            builder.field("StartDate").gte(start_after)
            conditions_added = True
            
        if end_before:
            if conditions_added:
                builder.and_()
            builder.field("FinishDate").lte(end_before)
            conditions_added = True
            
        return builder

class P6Client:
    """Enhanced P6 REST API client with comprehensive filter support"""

    def __init__(self, config: Optional[P6ClientConfig] = None):
        self.config = config or P6ClientConfig()
        self.session = requests.Session()
        
        # Configure session with retry adapter
        adapter = requests.adapters.HTTPAdapter(max_retries=self.config.max_retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self._database: Optional[str] = None
        self._auth_token: Optional[str] = None
        self._authenticated = False

    def initialize(self, username: str, password: str, database_name: str):
        """Initialize client with authentication"""
        try:
            self._auth_token = base64.b64encode(f"{username}:{password}".encode()).decode()
            self._database = database_name
            self._authenticate()
            logger.info("✅ P6 client initialized successfully")
        except Exception as e:
            logger.error(f"❌ P6 client initialization failed: {e}")
            raise P6AuthenticationError(f"Failed to initialize P6 client: {e}")

    def initialize_with_auth_key(self, auth_key: str, database_name: str):
        """Initialize client with pre-encoded auth key"""
        try:
            # Validate auth key by decoding
            decoded = base64.b64decode(auth_key).decode('utf-8')
            username, password = decoded.split(':', 1)
            self._auth_token = auth_key
            self._database = database_name
            self._authenticate()
            logger.info("✅ P6 client initialized with auth key")
        except Exception as e:
            raise P6AuthenticationError(f"Invalid auth key or initialization failed: {e}")

    def _authenticate(self):
        """Perform login"""
        try:
            login_url = f"{self.config.base_url}/login"
            headers = {"authToken": self._auth_token}
            params = {"DatabaseName": self._database}

            response = self.session.post(
                login_url,
                headers=headers,
                params=params,
                timeout=self.config.timeout
            )

            if response.status_code != 200:
                raise P6AuthenticationError(f"Login failed: {response.status_code} - {response.text}")

            self._authenticated = True
            logger.info("✅ P6 authentication successful")
        except Exception as e:
            raise P6AuthenticationError(f"Authentication failed: {e}")

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        return {
            "Accept": "application/json",
            "AuthToken": self._auth_token
        }

    def make_request(self, endpoint: str,
                    params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make authenticated request with HTTP timing / logging"""
        if not self._authenticated:
            raise P6AuthenticationError("Client not authenticated")

        url = f"{self.config.base_url}/{endpoint.lstrip('/')}"
        request_params = {"DatabaseName": self._database}
        if params:
            request_params.update(params)

        try:
            start = time.time()                              # <- start timer
            response = self.session.get(
                url,
                params=request_params,
                headers=self._get_headers(),
                timeout=self.config.timeout
            )
            elapsed_ms = (time.time() - start) * 1000        # <- stop timer
            self._log_request_response(response, elapsed_ms) # <- pass duration
            if response.status_code != 200:
                raise P6APIError(f"API request failed: {response.status_code} - {response.text}")
            return response.json()

        except requests.exceptions.Timeout:
            raise P6APIError(f"Request timeout after {self.config.timeout}s")
        except Exception as e:
            logger.error(f"❌ P6 API request failed: {e}")
            raise P6APIError(f"Request failed: {e}")

    # --- 3. replace old helper with timed variant ------------------------------
    def _log_request_response(self,
                            response: requests.Response,
                            duration_ms: float) -> None:
        """Write concise INFO line plus verbose DEBUG dump."""
        req = response.request
        # human-friendly one-liner
        logger.info(f"[HTTP] {req.method} {req.url} → {response.status_code} "
                    f"in {duration_ms:.1f} ms")

        # full headers / body (first 400 B) for deep debugging
        logger.debug(
            f"Req-Headers: {dict(req.headers)}\n"
            f"Resp-Headers: {dict(response.headers)}\n"
            f"Resp-Body: {response.text[:400]}\n"
        )


    def get_projects(self, 
                    filter_builder: Optional[FilterBuilder] = None,
                    filter_query: Optional[str] = None,
                    fields: Optional[List[str]] = None,
                    order_by: Optional[str] = None,
                    limit: int = 100) -> List[Dict[str, Any]]:
        """Get projects with enhanced filtering support"""
        params = {}
        
        # Use FilterBuilder if provided, otherwise fall back to raw parameters
        if filter_builder:
            builder_params = filter_builder.build_query_params()
            params.update(builder_params)
        else:
            # Legacy support for raw parameters
            if fields:
                params["Fields"] = ",".join(fields)
            if order_by:
                params["OrderBy"] = order_by
            if filter_query:
                params["Filter"] = filter_query

        if limit:
            params["PageSize"] = limit

        result = self.make_request("project", params)
        return result if isinstance(result, list) else []

    def get_activities(
        self,
        project_id: Union[str, int],           # accept numeric or alphanumeric
        filter_builder: Optional[FilterBuilder] = None,
        filter_query: Optional[str] = None,
        fields: Optional[List[str]] = None,
        order_by: Optional[str] = None,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Return activities for one project with robust ID handling"""
        params: Dict[str, Any] = {}

        # ── PROJECT FILTER ──────────────────────────────────────────────
        if isinstance(project_id, (int, float)) or (
            isinstance(project_id, str) and project_id.isdigit()
        ):
            project_filter = f"ProjectObjectId:eq:{project_id}"
        else:
            project_filter = f"ProjectObjectId:eq:'{project_id}'"
        # ────────────────────────────────────────────────────────────────

        if filter_builder:
            builder_params = filter_builder.build_query_params()
            # merge project filter with any existing filter
            existing_filter = builder_params.get("Filter")
            params["Filter"] = (
                f"{project_filter} :and: ({existing_filter})"
                if existing_filter
                else project_filter
            )
            params.update({k: v for k, v in builder_params.items() if k != "Filter"})
        else:
            params["Filter"] = project_filter
            if filter_query:
                params["Filter"] = f"{project_filter} :and: ({filter_query})"
            if fields:
                params["Fields"] = ",".join(fields)
            if order_by:
                params["OrderBy"] = order_by

        params["PageSize"] = limit
        result = self.make_request("activity", params)
        return result if isinstance(result, list) else []

    def get_resources(self,
                     filter_builder: Optional[FilterBuilder] = None,
                     filter_query: Optional[str] = None,
                     fields: Optional[List[str]] = None,
                     order_by: Optional[str] = None,
                     limit: int = 100) -> List[Dict[str, Any]]:
        """Get resources with enhanced filtering support"""
        params = {}
        
        if filter_builder:
            builder_params = filter_builder.build_query_params()
            params.update(builder_params)
        else:
            if fields:
                params["Fields"] = ",".join(fields)
            if order_by:
                params["OrderBy"] = order_by
            if filter_query:
                params["Filter"] = filter_query

        if limit:
            params["PageSize"] = limit

        result = self.make_request("resource", params)
        return result if isinstance(result, list) else []

    def close(self):
        """Clean up client resources"""
        if hasattr(self.session, 'close'):
            self.session.close()
        logger.info("✅ P6 client closed")