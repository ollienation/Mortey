# run.py - ✅ Assistant Runner
import asyncio
import signal
import logging
import os
import uvicorn
from contextlib import suppress
from core.assistant_core import assistant

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("runner")

async def main():
    """Main entry point with graceful shutdown handling"""
    config = uvicorn.Config(
        "run:assistant.app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        lifespan="on",
        log_level="info"
    )
    
    server = uvicorn.Server(config)
    
    # Setup signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(server)))
    
    await server.serve()

async def shutdown(server: uvicorn.Server):
    """Handle graceful shutdown"""
    logger.info("🚦 Shutdown signal received")
    
    # Close WebSocket connections
    for handler in assistant.app.state.websocket_handlers:
        await handler.close()
    
    # Stop the server
    server.should_exit = True
    await assistant.graceful_shutdown()
    
    # Cancel all running tasks
    for task in asyncio.all_tasks():
        if task is not asyncio.current_task():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

if __name__ == "__main__":
    asyncio.run(main())
