# api/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import asyncio
import json
import uuid
import time
import logging
from contextlib import asynccontextmanager

# Import your existing components
from core.assistant_core import AssistantCore
from core.state import AssistantState
from config.settings import config

logger = logging.getLogger("api")

# Global assistant instance
assistant_core: Optional[AssistantCore] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global assistant_core
    
    # Startup
    logger.info("🚀 Starting Mortey Assistant API")
    assistant_core = AssistantCore()
    await assistant_core.initialize()
    logger.info("✅ Assistant Core initialized")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Mortey Assistant API")

# Create FastAPI app with lifespan management
app = FastAPI(
    title="Mortey Assistant API",
    description="Modular Agentic Assistant with LangGraph 0.4.8",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
