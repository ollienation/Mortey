# api/endpoints.py (continued in main.py)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint for interacting with the assistant
    """
    if not assistant_core:
        raise HTTPException(status_code=503, detail="Assistant not initialized")
    
    start_time = time.time()
    
    try:
        # Process the message
        result = await assistant_core.process_message(
            message=request.message,
            thread_id=request.thread_id,
            user_id=request.user_id
        )
        
        processing_time = time.time() - start_time
        
        # Extract agent information if available
        session_info = assistant_core.get_session_info()
        
        return ChatResponse(
            response=result.get("response", ""),
            session_id=result.get("session_id", ""),
            message_count=result.get("message_count", 0),
            agent_used=session_info.get("current_agent"),
            processing_time=processing_time,
            token_usage=session_info.get("token_usage")
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}"
        )

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Streaming chat endpoint for real-time responses
    """
    if not assistant_core:
        raise HTTPException(status_code=503, detail="Assistant not initialized")
    
    async def generate_stream():
        try:
            # For now, we'll simulate streaming by yielding the full response
            # In a full implementation, you'd modify AssistantCore to support streaming
            result = await assistant_core.process_message(
                message=request.message,
                thread_id=request.thread_id,
                user_id=request.user_id
            )
            
            response_text = result.get("response", "")
            
            # Simulate streaming by yielding chunks
            words = response_text.split()
            for i, word in enumerate(words):
                chunk = {
                    "delta": word + " ",
                    "finished": i == len(words) - 1,
                    "session_id": result.get("session_id", "")
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.05)  # Small delay for streaming effect
                
        except Exception as e:
            error_chunk = {
                "error": str(e),
                "finished": True
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"}
    )

@app.get("/session/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str) -> SessionInfo:
    """Get information about a specific session"""
    if not assistant_core:
        raise HTTPException(status_code=503, detail="Assistant not initialized")
    
    try:
        session_info = assistant_core.get_session_info()
        
        if not session_info or session_info.get("session_id") != session_id:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionInfo(**session_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation/{thread_id}", response_model=ConversationHistory)
async def get_conversation_history(thread_id: str) -> ConversationHistory:
    """Get conversation history for a thread"""
    if not assistant_core:
        raise HTTPException(status_code=503, detail="Assistant not initialized")
    
    try:
        history = await assistant_core.get_conversation_history(thread_id)
        
        return ConversationHistory(
            messages=history,
            total_messages=len(history),
            session_id=thread_id,
            user_id="default_user"  # You might want to track this better
        )
        
    except Exception as e:
        logger.error(f"Conversation history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=SystemStatus)
async def health_check() -> SystemStatus:
    """Health check endpoint"""
    if not assistant_core:
        return SystemStatus(
            status="unhealthy",
            supervisor_initialized=False,
            checkpointer_type="none",
            setup_complete=False,
            agents_available=[],
            session_active=False,
            langsmith_enabled=False,
            modern_patterns="LangGraph 0.4.8 + langgraph-supervisor",
            timestamp=time.time()
        )
    
    try:
        system_status = assistant_core.get_system_status()
        
        # Determine overall health
        health_indicators = [
            system_status.get("supervisor_initialized", False),
            system_status.get("setup_complete", False),
            len(system_status.get("agents_available", [])) > 0
        ]
        
        if all(health_indicators):
            status = "healthy"
        elif any(health_indicators):
            status = "degraded"
        else:
            status = "unhealthy"
        
        return SystemStatus(
            status=status,
            **system_status
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return SystemStatus(
            status="unhealthy",
            supervisor_initialized=False,
            checkpointer_type="error",
            setup_complete=False,
            agents_available=[],
            session_active=False,
            langsmith_enabled=False,
            modern_patterns="LangGraph 0.4.8 + langgraph-supervisor",
            timestamp=time.time()
        )

@app.post("/feedback")
async def provide_feedback(
    thread_id: str,
    feedback: str,
    feedback_type: Literal["approve", "reject", "modify"] = "modify"
):
    """Provide human feedback for human-in-the-loop scenarios"""
    if not assistant_core:
        raise HTTPException(status_code=503, detail="Assistant not initialized")
    
    try:
        result = await assistant_core.provide_human_feedback(
            thread_id=thread_id,
            feedback=feedback,
            feedback_type=feedback_type
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Administrative endpoints
@app.get("/admin/stats")
async def get_system_stats():
    """Get detailed system statistics (admin only)"""
    if not assistant_core:
        raise HTTPException(status_code=503, detail="Assistant not initialized")
    
    try:
        from config.llm_manager import llm_manager
        
        return {
            "system_status": assistant_core.get_system_status(),
            "session_info": assistant_core.get_session_info(),
            "llm_usage": llm_manager.get_usage_stats(),
            "workspace_info": {
                "workspace_dir": str(config.workspace_dir),
                "project_root": str(config.project_root)
            }
        }
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/reset")
async def reset_system():
    """Reset the assistant system (admin only)"""
    global assistant_core
    
    try:
        if assistant_core:
            # You might want to add cleanup methods to AssistantCore
            pass
        
        assistant_core = AssistantCore()
        await assistant_core.initialize()
        
        return {"message": "System reset successfully"}
        
    except Exception as e:
        logger.error(f"Reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
