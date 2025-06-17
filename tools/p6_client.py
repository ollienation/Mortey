# tools/p6_client.py - Simplified P6 Client following working.py patterns

import requests
import logging
import time
import json
import base64
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger("p6_client")

@dataclass
class P6ClientConfig:
    """Simplified configuration matching working.py approach"""
    base_url: str = "http://primaverap6restapi.primaplan.com/p6ws/restapi"
    timeout: float = 15.0
    max_retries: int = 3

class P6AuthenticationError(Exception):
    """Authentication-specific error for P6 operations"""
    pass

class P6APIError(Exception):
    """General P6 API error"""
    pass

class P6Client:
    """Simplified P6 REST API client following working.py session management"""

    def __init__(self, config: Optional[P6ClientConfig] = None):
        self.config = config or P6ClientConfig()
        self.session = requests.Session()
        
        # Configure session with retry adapter like working.py
        adapter = requests.adapters.HTTPAdapter(max_retries=self.config.max_retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self._database: Optional[str] = None
        self._auth_token: Optional[str] = None
        self._authenticated = False

    def initialize(self, username: str, password: str, database_name: str):
        """Initialize client with authentication - synchronous like working.py"""
        try:
            # Create base64 auth token like working.py
            self._auth_token = base64.b64encode(f"{username}:{password}".encode()).decode()
            self._database = database_name
            
            # Perform login to establish JSESSIONID cookie
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
            
            # Perform login
            self._authenticate()
            logger.info("✅ P6 client initialized with auth key")
            
        except Exception as e:
            raise P6AuthenticationError(f"Invalid auth key or initialization failed: {e}")

    def _authenticate(self):
        """Perform login using working.py logic"""
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
        """Get headers for API requests - simplified like working.py"""
        return {
            "Accept": "application/json",
            "AuthToken": self._auth_token
        }

    def make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make authenticated request - synchronous following working.py pattern"""
        if not self._authenticated:
            raise P6AuthenticationError("Client not authenticated")

        url = f"{self.config.base_url}/{endpoint.lstrip('/')}"
        
        # Add database name to params like working.py
        request_params = {"DatabaseName": self._database}
        if params:
            request_params.update(params)

        try:
            response = self.session.get(
                url,
                params=request_params,
                headers=self._get_headers(),
                timeout=self.config.timeout
            )
            
            # Log request/response like working.py dump() function
            self._log_request_response(response)
            
            if response.status_code != 200:
                raise P6APIError(f"API request failed: {response.status_code} - {response.text}")
                
            return response.json()
            
        except requests.exceptions.Timeout:
            raise P6APIError(f"Request timeout after {self.config.timeout}s")
        except Exception as e:
            logger.error(f"❌ P6 API request failed: {e}")
            raise P5APIError(f"Request failed: {e}")

    def _log_request_response(self, response: requests.Response):
        """Log request/response details like working.py dump() function"""
        req = response.request
        logger.debug(
            f"{req.method} {req.url}\n"
            f"Req-Hdrs: {dict(req.headers)}\n"
            f"Resp-{response.status_code}: {dict(response.headers)}\n"
            f"Body: {response.text[:400]}\n"
        )

    def get_projects(self, filter_query: Optional[str] = None, 
                    fields: Optional[List[str]] = None, 
                    order_by: Optional[str] = None, 
                    limit: int = 100) -> List[Dict[str, Any]]:
        """Get projects with optional filtering - following working.py patterns"""
        params = {}
        
        if fields:
            params["Fields"] = ",".join(fields)
        if order_by:
            params["OrderBy"] = order_by
        if limit:
            params["PageSize"] = limit
        if filter_query:
            params["Filter"] = filter_query

        result = self.make_request("project", params)
        return result if isinstance(result, list) else []

    def get_activities(self, project_id: str, 
                      filter_query: Optional[str] = None,
                      fields: Optional[List[str]] = None, 
                      limit: int = 500) -> List[Dict[str, Any]]:
        """Get activities for a specific project"""
        params = {}
        
        if fields:
            params["Fields"] = ",".join(fields)
        if limit:
            params["PageSize"] = limit
            
        # Build filter with project ID
        project_filter = f"ProjectObjectId = {project_id}"
        if filter_query:
            project_filter += f" and {filter_query}"
        params["Filter"] = project_filter

        result = self.make_request("activity", params)
        return result if isinstance(result, list) else []

    def close(self):
        """Clean up client resources"""
        if hasattr(self.session, 'close'):
            self.session.close()
        logger.info("✅ P6 client closed")
