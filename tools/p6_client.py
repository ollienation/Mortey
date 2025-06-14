# tools/p6_client.py - Production-ready P6 HTTP Client
import asyncio
import aiohttp
import logging
import time
import json
import base64
import os
from typing import Optional, Any, List
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from asyncio import TaskGroup

from core.circuit_breaker import global_circuit_breaker
from core.error_handling import ErrorHandler
from config.settings import config

logger = logging.getLogger("p6_client")

@dataclass
class P6ClientConfig:
    """Configuration for P6 client with security and performance settings"""
    base_url: str = "http://primaverap6restapi.primaplan.com/p6ws/restapi"
    swagger_url: str = "http://primaverap6restapi.primaplan.com/p6ws/swagger.json"
    timeout: float = 30.0
    max_retries: int = 3
    rate_limit_per_second: float = 10.0
    connection_pool_size: int = 10
    session_timeout_minutes: int = 30

class P6AuthenticationError(Exception):
    """Authentication-specific error for P6 operations"""
    pass

class P6APIError(Exception):
    """General P6 API error"""
    pass

class P6Client:
    """
    Production-grade P6 REST API client with session management and error handling
    """
    
    def __init__(self, config: Optional[P6ClientConfig] = None):
        self.config = config or P6ClientConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        self.auth_headers: dict[str, str] = {}
        self.last_auth_time: float = 0
        self.request_times: List[float] = []
        
    async def initialize(self, username: str, password: str, database_name: str):
        """Initialize client with authentication"""
        self._db_name = database_name
        self._basic_key = base64.b64encode(f"{username}:{password}".encode()).decode()
        
        try:
            # Create aiohttp session with connection pooling
            connector = aiohttp.TCPConnector(
                limit=self.config.connection_pool_size,
                limit_per_host=self.config.connection_pool_size,
                keepalive_timeout=300
            )
            
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
            
            # Authenticate with P6
            await self._authenticate(username, password, database_name)
            logger.info("✅ P6 client initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ P6 client initialization failed: {e}")
            await self.close()
            raise P6AuthenticationError(f"Failed to initialize P6 client: {e}")
        
    # Persistant P6 session
    async def initialize_with_auth_key(
        self,
        auth_key: str,
        database_name: str,
    ) -> None:
        """
        1. Open a session
        2. Call /login with Basic auth   → get bearer token
        3. Store bearer token for all future requests
        """
        connector = aiohttp.TCPConnector(
            limit=self.config.connection_pool_size,
            limit_per_host=self.config.connection_pool_size,
            keepalive_timeout=300,
        )
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
        )
        try:
            await self._login_with_basic_key(auth_key, database_name)
            logger.info("✅ P6 session established (Bearer token in place)")
        except Exception as e:
            logger.error(f"❌ P6 client initialization failed: {e}")
            await self.close()
            raise P6AuthenticationError(f"Failed to initialize P6 client: {e}")

    # ------------------------------------------------------------------
    # shared login helper  (re-usable by the username/password path too)
    # ------------------------------------------------------------------
    async def _login_with_basic_key(
        self,
        auth_key: str,
        database_name: str,
    ) -> None:
        """Perform the real /login handshake and store the bearer token"""
        login_headers = {
            "Authorization": f"Basic {auth_key}",
            "DatabaseName": database_name,
            "Accept": "application/json",
        }

        async with self.session.post(
            f"{self.config.base_url}/login",
            headers=login_headers,
            params={"DatabaseName": database_name},  # some servers need the param too
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise P6AuthenticationError(
                    f"Login failed: {resp.status} – {text[:120]}"
                )

            # token can be in different places depending on patch level
            token = (
                resp.headers.get("X-Authorization")
                or resp.headers.get("X-Auth-Token")
                or (resp.cookies.get("X-Auth-Token").value if resp.cookies else None)
            )
            if not token:
                raise P6AuthenticationError("Login succeeded but no session token returned")

            # Replace Basic with Bearer for all subsequent requests
            self.auth_headers = {
                "Authorization": f"Bearer {token}",
                "DatabaseName": database_name,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            self.last_auth_time = time.time()
    
    async def make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[dict[str, Any]] = None,
        data: Optional[dict[str, Any]] = None,
        timeout_override: Optional[float] = None
    ) -> dict[str, Any]:
        """Make authenticated request to P6 API with comprehensive error handling"""
        
        # Rate limiting check
        await self._check_rate_limit()
        
        # Session validation
        if not self.session or self._is_session_expired():
            raise P6AuthenticationError("Session expired or not initialized")
        
        url = f"{self.config.base_url}/{endpoint.lstrip('/')}"
        
        # Prepare request with authentication headers
        headers = self.auth_headers.copy()
        headers["Content-Type"] = "application/json"
        
        timeout = timeout_override or self.config.timeout
        
        try:
            # Use circuit breaker for API calls
            return await global_circuit_breaker.call_with_circuit_breaker(
                "p6_api",
                self._execute_request,
                method, url, headers, params, data, timeout
            )
            
        except aiohttp.ClientResponseError as e:
            if e.status == 401:
                logger.warning("Token expired – re-authenticating")
                await self._login_with_basic_key(self._basic_key, self._db_name)
                return await self._execute_request(method, url, headers, params, data, timeout)
                
        except asyncio.TimeoutError:
            raise P6APIError(f"Request timeout after {timeout}s")
            
        except Exception as e:
            logger.error(f"❌ P6 API request failed: {e}")
            raise P6APIError(f"Request failed: {e}")
    
    async def initialize_with_auth_key(self, auth_key: str, database_name: str):
        """Initialize client with pre-encoded auth key"""
        try:
            # Create aiohttp session
            connector = aiohttp.TCPConnector(
                limit=self.config.connection_pool_size,
                limit_per_host=self.config.connection_pool_size,
                keepalive_timeout=300
            )
            
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
            
            # Set authentication headers directly
            self.auth_headers = {
                "Authorization": f"Basic {auth_key}",
                "DatabaseName": database_name,
                "Content-Type": "application/json"
            }
            
            logger.info("✅ P6 client initialized with auth key")
            
        except Exception as e:
            logger.error(f"❌ P6 client initialization failed: {e}")
            await self.close()
            raise P6AuthenticationError(f"Failed to initialize P6 client: {e}")

    async def _execute_request(
        self, 
        method: str, 
        url: str, 
        headers: dict[str, str],
        params: Optional[dict[str, Any]], 
        data: Optional[dict[str, Any]], 
        timeout: float
    ) -> dict[str, Any]:
        """Execute the actual HTTP request"""
        request_start = time.time()
        
        async with self.session.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=data,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            response_time = (time.time() - request_start) * 1000
            
            if response.status >= 400:
                error_text = await response.text()
                logger.error(f"❌ P6 API error {response.status}: {error_text}")
                response.raise_for_status()
            
            result = await response.json()
            
            logger.debug(f"✅ P6 API request completed in {response_time:.1f}ms: {method} {url}")
            return result
    
    async def _check_rate_limit(self):
        """Implement rate limiting to protect P6 backend"""
        current_time = time.time()
        
        # Remove old request times (older than 1 second)
        self.request_times = [
            req_time for req_time in self.request_times 
            if current_time - req_time < 1.0
        ]
        
        # Check if we're within rate limit
        if len(self.request_times) >= self.config.rate_limit_per_second:
            sleep_time = 1.0 - (current_time - self.request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.request_times.append(current_time)
    
    def _is_session_expired(self) -> bool:
        """Check if current session has expired"""
        if not self.last_auth_time:
            return True
        
        session_age_minutes = (time.time() - self.last_auth_time) / 60
        return session_age_minutes > self.config.session_timeout_minutes
    
    async def get_projects(
        self, 
        filter_query: Optional[str] = None,
        fields: Optional[List[str]] = None,
        order_by: Optional[str] = None,
        limit: int = 100
    ) -> List[dict[str, Any]]:
        """Get projects with optional filtering and field selection"""
        params = {"limit": limit}
        
        if filter_query:
            params["filter"] = filter_query
        if fields:
            params["fields"] = ",".join(fields)
        if order_by:
            params["orderby"] = order_by
            
        result = await self.make_request("GET", "/project", params=params)
        return result.get("data", [])
    
    async def get_activities(
        self,
        project_id: str,
        filter_query: Optional[str] = None,
        fields: Optional[List[str]] = None,
        limit: int = 500
    ) -> List[dict[str, Any]]:
        """Get activities for a specific project"""
        params = {
            "filter": f"ProjectId='{project_id}'"
        }
        
        if filter_query:
            params["filter"] += f" and {filter_query}"
        if fields:
            params["fields"] = ",".join(fields)
        if limit:
            params["limit"] = limit
            
        result = await self.make_request("GET", "/activity", params=params)
        return result.get("data", [])
    
    async def get_resources(
        self,
        project_id: Optional[str] = None,
        filter_query: Optional[str] = None,
        fields: Optional[List[str]] = None
    ) -> List[dict[str, Any]]:
        """Get resources with optional project filtering"""
        params = {}
        
        filters = []
        if project_id:
            filters.append(f"ProjectId='{project_id}'")
        if filter_query:
            filters.append(filter_query)
            
        if filters:
            params["filter"] = " and ".join(filters)
        if fields:
            params["fields"] = ",".join(fields)
            
        result = await self.make_request("GET", "/resource", params=params)
        return result.get("data", [])
    
    async def close(self):
        """Clean up client resources"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("✅ P6 client closed")
