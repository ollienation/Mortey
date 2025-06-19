# P6 Filter Mapper

A Python module for mapping natural language queries to Primavera P6 REST API filter syntax.

## Overview

This package provides a solution for converting user-friendly natural language queries into the specific filter syntax required by the Primavera P6 REST API. It enables AI assistants and applications to generate proper P6 API filter queries from natural language input.

## Files

1. `p6_filter_config.json` - Configuration file with filter operators and mappings
2. `p6_filter_mapper.py` - Python module for mapping natural language to P6 filters
3. `p6_tools_integration.py` - Example integration with p6_tools.py

## Usage

### Basic Usage

```python
from p6_filter_mapper import P6FilterMapper

# Initialize the mapper with the configuration file
mapper = P6FilterMapper('p6_filter_config.json')

# Convert a natural language query to P6 filter syntax
query = "active projects starting after 2024 named Bridge"
p6_filter = mapper.build_filter(query)

# Get a human-readable explanation of the filter
explanation = mapper.get_filter_explanation(p6_filter)

print(f"Query: '{query}'")
print(f"P6 Filter: '{p6_filter}'")
print(f"Explanation: {explanation}")
```

Output:
```
Query: 'active projects starting after 2024 named Bridge'
P6 Filter: 'StartDate:gte:'2024-01-01' :and: Status:eq:'Active' :and: Name:like:'%Bridge%''
Explanation: Showing projects where Project or activity start date on or after 2024-01-01 AND Status is 'Active' AND Name contains 'Bridge'
```

### Integration with P6 Tools

The `p6_tools_integration.py` file demonstrates how to integrate the P6FilterMapper with the p6_tools.py module:

```python
# Example of how the search_projects tool would be called
search_projects = tools[0]
result = search_projects("active projects starting after 2024")
print(result)
```

## Supported Query Types

The P6FilterMapper supports the following types of natural language queries:

1. **Date-based queries**:
   - "projects starting after 2024-01-01"
   - "projects finishing before 2025-12-31"

2. **Status-based queries**:
   - "active projects"
   - "completed projects"
   - "projects in progress"

3. **Name-based queries**:
   - "projects named Highway Construction"
   - "projects called Bridge Renovation"

4. **ID-based queries**:
   - "projects with ID P-1001"
   - "project #P-1001"

5. **Combined queries**:
   - "active projects starting after 2024 named Bridge"

## P6 Filter Syntax

The P6 REST API uses a specific filter syntax with operators like:

- `:eq:` - Equals
- `:gt:` - Greater than
- `:gte:` - Greater than or equal to
- `:lt:` - Less than
- `:lte:` - Less than or equal to
- `:like:` - Contains substring (with % as wildcard)
- `:and:` - Logical AND
- `:or:` - Logical OR

For example:
```
StartDate:gte:'2024-01-01' :and: Status:eq:'Active'
```

## Customization

You can customize the filter mappings by modifying the `p6_filter_config.json` file. This file contains:

- Filter operators
- Logical operators
- Date fields
- Status values
- Natural language to filter mappings

## License

MIT
