# tools/p6_filter_mapper.py - Enhanced Filter Mapper

import logging
import json
import re
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from tools.p6_client import FilterBuilder

logger = logging.getLogger("p6_filter_mapper")

class P6FilterMapper:
    """
    Enhanced mapper from natural language queries to P6 REST API filter syntax.
    
    This class handles the conversion of user-friendly natural language queries
    into P6 filters using both direct string-based syntax and the FilterBuilder class.
    """

    def __init__(self, config_file: str = 'p6_filter_config.json'):
        """Initialize with a configuration file path"""
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"P6 filter configuration file not found: {config_file}")

    def extract_date(self, query: str) -> Optional[str]:
        """Extract date from query string"""
        # Enhanced date pattern to handle more formats
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
            r'\d{4}/\d{1,2}/\d{1,2}',  # YYYY/MM/DD
            r'\d{1,2}-\d{1,2}-\d{4}',  # MM-DD-YYYY
            r'\d{4}'  # Year only
        ]
        
        for pattern in date_patterns:
            dates = re.findall(pattern, query)
            if dates:
                date_value = dates[0]
                
                # Convert to YYYY-MM-DD format
                if len(date_value) == 4:  # Year only
                    date_value += "-01-01"
                elif '/' in date_value:
                    parts = date_value.split('/')
                    if len(parts[0]) == 4:  # YYYY/MM/DD
                        date_value = f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
                    else:  # MM/DD/YYYY
                        date_value = f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
                elif '-' in date_value and len(date_value.split('-')[0]) != 4:  # MM-DD-YYYY
                    parts = date_value.split('-')
                    date_value = f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
                
                return date_value
                
        return None

    def extract_name(self, query: str) -> Optional[str]:
        """Extract project name from query string"""
        # Enhanced name patterns
        name_patterns = [
            r'named\s+"([^"]+)"',  # named "Project Name"
            r'named\s+\'([^\']+)\'',  # named 'Project Name'
            r'named\s+([\w\s\-]+)(?:\s|$)',  # named Project Name
            r'with\s+name\s+"([^"]+)"',  # with name "Project Name"
            r'with\s+name\s+\'([^\']+)\'',  # with name 'Project Name'
            r'with\s+name\s+([\w\s\-]+)(?:\s|$)',  # with name Project Name
            r'called\s+"([^"]+)"',  # called "Project Name"
            r'called\s+\'([^\']+)\'',  # called 'Project Name'
            r'called\s+([\w\s\-]+)(?:\s|$)',  # called Project Name
            r'name\s+contains\s+"([^"]+)"',  # name contains "text"
            r'name\s+contains\s+\'([^\']+)\'',  # name contains 'text'
            r'name\s+contains\s+([\w\s\-]+)(?:\s|$)'  # name contains text
        ]

        for pattern in name_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
                
        return None

    def extract_id(self, query: str) -> Optional[str]:
        """Extract project ID from query string"""
        # Enhanced ID patterns
        id_patterns = [
            r'with\s+id\s+"([^"]+)"',  # with id "ABC-123"
            r'with\s+id\s+\'([^\']+)\'',  # with id 'ABC-123'
            r'with\s+id\s+([\w\-]+)',  # with id ABC-123
            r'id\s+"([^"]+)"',  # id "ABC-123"
            r'id\s+\'([^\']+)\'',  # id 'ABC-123'
            r'id\s+([\w\-]+)',  # id ABC-123
            r'id\s*=\s*"([^"]+)"',  # id = "ABC-123"
            r'id\s*=\s*\'([^\']+)\'',  # id = 'ABC-123'
            r'id\s*=\s*([\w\-]+)',  # id = ABC-123
            r'#([\w\-]+)'  # #ABC-123
        ]

        for pattern in id_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
                
        return None
        
    def extract_field_value_pairs(self, query: str) -> List[Tuple[str, str, str]]:
        """
        Extract field, operator, and value triplets from natural language query.
        Returns a list of tuples (field, operator, value)
        """
        result = []
        
        # Match patterns like "Field equals Value", "Field greater than Value", etc.
        patterns = [
            # Field = Value
            (r'(\w+)\s+(?:equals|equal to|=|==)\s+["\']?([^"\']+)["\']?', ':eq:'),
            # Field != Value
            (r'(\w+)\s+(?:not equal to|!=|<>)\s+["\']?([^"\']+)["\']?', ':neq:'),
            # Field > Value
            (r'(\w+)\s+(?:greater than|>)\s+["\']?([^"\']+)["\']?', ':gt:'),
            # Field >= Value
            (r'(\w+)\s+(?:greater than or equal to|>=)\s+["\']?([^"\']+)["\']?', ':gte:'),
            # Field < Value
            (r'(\w+)\s+(?:less than|<)\s+["\']?([^"\']+)["\']?', ':lt:'),
            # Field <= Value
            (r'(\w+)\s+(?:less than or equal to|<=)\s+["\']?([^"\']+)["\']?', ':lte:'),
            # Field contains Value
            (r'(\w+)\s+contains\s+["\']?([^"\']+)["\']?', ':like:'),
            # Field like Value
            (r'(\w+)\s+like\s+["\']?([^"\']+)["\']?', ':like:'),
            # Field in (Values)
            (r'(\w+)\s+in\s+\(?([^)]+)\)?', ':in:'),
            # Field is null
            (r'(\w+)\s+is\s+(?:null|empty)', ':isnull:'),
            # Field is not null
            (r'(\w+)\s+is\s+not\s+(?:null|empty)', ':isnotnull:')
        ]
        
        for pattern, operator in patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                field = match.group(1).strip()
                
                # Special case for null operators
                if operator in [':isnull:', ':isnotnull:']:
                    result.append((field, operator, None))
                else:
                    value = match.group(2).strip()
                    
                    # Special handling for IN operator
                    if operator == ':in:':
                        # Convert comma-separated list to proper format
                        values = [v.strip() for v in value.split(',')]
                        values = [v for v in values if v]  # Remove empty values
                        value = ','.join(values)
                    
                    # Special handling for LIKE operator
                    elif operator == ':like:' and '%' not in value:
                        value = f"%{value}%"  # Add wildcards if not present
                        
                    result.append((field, operator, value))
        
        return result

    def parse_query(self, query: str) -> List[str]:
        """Enhanced parser that converts natural language to P6 filter components"""
        query_lower = query.lower()
        filter_parts = []
        
        # 1. Try to extract explicit field-operator-value patterns
        field_value_pairs = self.extract_field_value_pairs(query)
        for field, operator, value in field_value_pairs:
            # Handle NULL operators
            if operator in [':isnull:', ':isnotnull:']:
                filter_parts.append(f"{field} {operator}")
            # Handle IN operator
            elif operator == ':in:':
                filter_parts.append(f"{field}{operator}({value})")
            # Handle normal operators
            else:
                filter_parts.append(f"{field}{operator}'{value}'")
        
        if field_value_pairs:
            return filter_parts
        
        # 2. Fall back to legacy extraction methods
        # Check for date-based queries
        if "start" in query_lower and "after" in query_lower:
            date_value = self.extract_date(query)
            if date_value:
                filter_parts.append(f"StartDate{self.config['filter_operators']['greater_than_or_equal']['symbol']}'{date_value}'")
        elif "start" in query_lower and "before" in query_lower:
            date_value = self.extract_date(query)
            if date_value:
                filter_parts.append(f"StartDate{self.config['filter_operators']['less_than_or_equal']['symbol']}'{date_value}'")
        elif "finish" in query_lower and "after" in query_lower:
            date_value = self.extract_date(query)
            if date_value:
                filter_parts.append(f"FinishDate{self.config['filter_operators']['greater_than_or_equal']['symbol']}'{date_value}'")
        elif "finish" in query_lower and "before" in query_lower:
            date_value = self.extract_date(query)
            if date_value:
                filter_parts.append(f"FinishDate{self.config['filter_operators']['less_than_or_equal']['symbol']}'{date_value}'")

        # Check for status-based queries
        for status in self.config['status_values']:
            if status.lower() in query_lower:
                filter_parts.append(f"Status{self.config['filter_operators']['equals']['symbol']}'{status}'")
                break

        # Check for name-based queries
        name = self.extract_name(query)
        if name:
            filter_parts.append(f"Name{self.config['filter_operators']['like']['symbol']}'%{name}%'")

        # Check for ID-based queries
        id_value = self.extract_id(query)
        if id_value:
            filter_parts.append(f"Id{self.config['filter_operators']['equals']['symbol']}'{id_value}'")

        # Default to name search if no specific filter detected and query is not empty
        if not filter_parts and query and not any(keyword in query_lower for keyword in ['status', 'start', 'finish']):
            # If query is just a simple term, assume it's a name search
            filter_parts.append(f"Name{self.config['filter_operators']['like']['symbol']}'%{query}%'")

        return filter_parts

    def build_filter(self, query: str, status_filter: Optional[str] = None) -> FilterBuilder: # Changed return type hint to FilterBuilder
        """Build a complete P6 filter with enhanced type safety"""
        # ✅ FIX: Initialize the FilterBuilder instance at the beginning
        builder = FilterBuilder()

        # Process the 'query' parameter to ensure it's a string or handled appropriately
        processed_query = query
        if processed_query is None:
            processed_query = "" # Treat None query as an empty string for parsing
        elif hasattr(processed_query, '__class__') and 'FieldInfo' in str(type(processed_query)):
            if hasattr(processed_query, 'default') and processed_query.default is not None:
                processed_query = str(processed_query.default)
            else:
                logger.warning("⚠️ Received FieldInfo object for query with no default, using empty string")
                processed_query = ""
        elif not isinstance(processed_query, str):
            try:
                processed_query = str(processed_query)
            except Exception as e:
                logger.error(f"❌ Cannot convert query '{query}' to string: {e}")
                processed_query = "" # Default to empty string on conversion error

        # Process the 'status_filter' parameter similarly
        processed_status_filter = status_filter
        if processed_status_filter is not None:
            if hasattr(processed_status_filter, '__class__') and 'FieldInfo' in str(type(processed_status_filter)):
                if hasattr(processed_status_filter, 'default') and processed_status_filter.default is not None:
                    processed_status_filter = str(processed_status_filter.default)
                else:
                    logger.warning("⚠️ Received FieldInfo object for status_filter with no default, treating as None")
                    processed_status_filter = None
            elif not isinstance(processed_status_filter, str):
                try:
                    processed_status_filter = str(processed_status_filter)
                except Exception as e:
                    logger.error(f"❌ Cannot convert status_filter '{status_filter}' to string: {e}")
                    processed_status_filter = None

        filter_parts = self.parse_query(processed_query)

        # If no filter parts were derived from the query, and no status filter is provided,
        # return the empty (but valid) FilterBuilder instance.
        if not filter_parts and not processed_status_filter:
            return builder

        # Add each filter part to the builder if derived from the query
        for i, part in enumerate(filter_parts):
            if i > 0:
                builder.and_()
            builder._add_raw_condition(part) # _add_raw_condition is a method of FilterBuilder [3]

        # Add status filter if provided and not already included in parsed filter_parts
        if processed_status_filter and not any("status" in p.lower() for p in filter_parts):
            if filter_parts: # Only add an "AND" if there are existing filter parts
                builder.and_()
            builder.field("Status").eq(processed_status_filter)

        return builder

    def get_filter_explanation(self, filter_query: str) -> str:
        """Generate a human-readable explanation of a P6 filter"""
        if not filter_query:
            return "No filter applied - showing all projects"

        parts = filter_query.split(f" {self.config['logical_operators']['and']['symbol']} ")
        explanations = []

        for part in parts:
            for field, desc in self.config['date_fields'].items():
                if field in part:
                    if self.config['filter_operators']['greater_than_or_equal']['symbol'] in part:
                        date = re.search(r"'([^']+)'", part).group(1)
                        explanations.append(f"{desc} on or after {date}")
                    elif self.config['filter_operators']['less_than_or_equal']['symbol'] in part:
                        date = re.search(r"'([^']+)'", part).group(1)
                        explanations.append(f"{desc} on or before {date}")
                    elif self.config['filter_operators']['greater_than']['symbol'] in part:
                        date = re.search(r"'([^']+)'", part).group(1)
                        explanations.append(f"{desc} after {date}")
                    elif self.config['filter_operators']['less_than']['symbol'] in part:
                        date = re.search(r"'([^']+)'", part).group(1)
                        explanations.append(f"{desc} before {date}")

            if "Status" in part:
                status_match = re.search(r"'([^']+)'", part)
                if status_match:
                    status = status_match.group(1)
                    explanations.append(f"Status is '{status}'")

            if "Name" in part:
                name_match = re.search(r"'([^']+)'", part)
                if name_match:
                    name = name_match.group(1)
                    name = name.replace('%', '')
                    explanations.append(f"Name contains '{name}'")

            if "Id" in part:
                id_match = re.search(r"'([^']+)'", part)
                if id_match:
                    id_value = id_match.group(1)
                    explanations.append(f"ID equals '{id_value}'")
                    
            # Handle IS NULL and IS NOT NULL
            is_null_match = re.search(r"(\w+)\s+:isnull:", part)
            if is_null_match:
                field = is_null_match.group(1)
                explanations.append(f"{field} is null")
                
            is_not_null_match = re.search(r"(\w+)\s+:isnotnull:", part)
            if is_not_null_match:
                field = is_not_null_match.group(1)
                explanations.append(f"{field} is not null")
                
            # Handle IN operator
            in_match = re.search(r"(\w+):in:\(([^)]+)\)", part)
            if in_match:
                field = in_match.group(1)
                values = in_match.group(2)
                explanations.append(f"{field} is one of [{values}]")

        return "Showing projects where " + " AND ".join(explanations)