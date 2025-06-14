import asyncio
import logging
import time
import uuid
import traceback
from typing import Optional, Any, List, Tuple
from pathlib import Path
import gradio as gr
import json

# Import your existing assistant components
from core.assistant_core import assistant
from config.settings import config
from config.llm_manager import llm_manager
from agents.agents import agent_factory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gradio_app")

class GradioAssistantInterface:
    """
    Gradio 5.33.2 interface wrapper for the LangGraph assistant framework.
    """
    
    def __init__(self):
        self.assistant_core = assistant
        self.session_cache: dict[str, dict[str, Any]] = {}
        self.is_initialized = False
        
    async def initialize_assistant(self):
        """Initialize the assistant core if not already done"""
        if not self.is_initialized:
            try:
                logger.info("🚀 Initializing assistant core for Gradio interface...")
                await self.assistant_core.initialize()
                self.is_initialized = True
                logger.info("✅ Assistant core initialized successfully")
                return "✅ Assistant initialized successfully!"
            except Exception as e:
                logger.error(f"❌ Failed to initialize assistant: {e}")
                return f"❌ Initialization failed: {str(e)}"
        return "✅ Assistant already initialized"
    
    def get_or_create_session_id(self, session_state: gr.State) -> Tuple[str, gr.State]:
        """Get or create a session ID for the current user session"""
        if session_state is None or not isinstance(session_state, dict) or 'session_id' not in session_state:
            session_id = str(uuid.uuid4())
            session_state = {'session_id': session_id, 'message_count': 0}
            logger.info(f"🆕 Created new session: {session_id}")
            return session_id, session_state
        return session_state['session_id'], session_state
    
    async def process_chat_message(
        self, 
        message: str, 
        history: List[List[str]], 
        session_state: gr.State,
        user_id: str = "gradio_user"
    ) -> Tuple[str, List[List[str]], gr.State]:
        """Process a chat message through the assistant framework"""
        try:
            if not self.is_initialized:
                await self.initialize_assistant()
            
            # Get or create session
            session_id, session_state = self.get_or_create_session_id(session_state)
            session_state['message_count'] = session_state.get('message_count', 0) + 1
            
            logger.info(f"📨 Processing message for session {session_id}: {message[:50]}...")
            
            # Process through assistant core
            result = await self.assistant_core.process_message(
                message=message,
                session_id=session_id,
                user_id=user_id
            )
            
            response = result.get('response', 'No response generated')
            
            # Update chat history
            history.append([message, response])
            
            logger.info(f"✅ Message processed successfully for session {session_id}")
            
            return "", history, session_state
            
        except Exception as e:
            error_msg = f"❌ Error processing message: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            # Add error to history
            history.append([message, error_msg])
            return "", history, session_state
    
    def clear_chat_history(self, session_state: gr.State) -> Tuple[List, gr.State]:
        """Clear chat history and reset session"""
        if session_state and 'session_id' in session_state:
            old_session = session_state['session_id']
            logger.info(f"🧹 Clearing history for session {old_session}")
        
        # Create new session
        new_session_id = str(uuid.uuid4())
        new_session_state = {'session_id': new_session_id, 'message_count': 0}
        
        return [], new_session_state
    
    # app.py - TOKEN LIMITED system status for Gradio
    async def get_system_status(self) -> str:
        """Get system status with limited scope for UI display"""
        try:
            if not self.is_initialized:
                return "❌ Assistant not initialized"
            
            # ✅ LIGHTWEIGHT: Get cached status only
            status = {
                "agents": len(self.assistant_core.agents),
                "models": len(llm_manager.models) if hasattr(llm_manager, 'models') else 0,
                "active_sessions": len(self.session_cache),
                "uptime": time.time() - getattr(self, 'start_time', time.time())
            }
            
            # ✅ SIMPLE: Basic status display
            status_text = f"""## 🖥️ System Status (Lightweight)

    ### 📊 Components
    - **Agents**: {status['agents']} initialized
    - **Models**: {status['models']} loaded  
    - **Sessions**: {status['active_sessions']} active
    - **Uptime**: {status['uptime']/3600:.1f} hours

    ### 🔧 Health Monitoring
    - Health checks run every 3-10 minutes (adaptive)
    - Token-limited testing (1 token max per check)
    - Intelligent caching (5 min for healthy components)

    💡 *Optimized for minimal resource usage*
    """
            
            return status_text
            
        except Exception as e:
            return f"❌ Error getting status: {str(e)}"
    
    async def upload_and_process_file(
        self, 
        file_path: str, 
        operation: str,
        session_state: gr.State
    ) -> Tuple[str, gr.State]:
        """Handle file uploads and processing"""
        try:
            if not file_path:
                return "❌ No file selected", session_state
            
            # Get session info
            session_id, session_state = self.get_or_create_session_id(session_state)
            
            # Copy file to workspace
            import shutil
            filename = Path(file_path).name
            workspace_path = config.workspace_dir / filename
            workspace_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, workspace_path)
            
            # Create message for file processing
            message = f"Please {operation} the file: {filename}"
            
            # Process through assistant
            result = await self.assistant_core.process_message(
                message=message,
                session_id=session_id,
                user_id="gradio_file_user"
            )
            
            response = result.get('response', 'File processing completed')
            
            return f"✅ File uploaded and processed: {filename}\n\n{response}", session_state
            
        except Exception as e:
            error_msg = f"❌ Error processing file: {str(e)}"
            logger.error(error_msg)
            return error_msg, session_state
    
    def create_interface(self) -> gr.Blocks:
        """Create the main Gradio 5.33.2 interface"""
        
        # Custom CSS
        custom_css = """
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
        }
        .chat-container {
            height: 600px !important;
        }
        .status-container {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        """
        
        with gr.Blocks(
            title="🤖 Mortey Assistant - LangGraph Framework",
            theme=gr.themes.Soft(),
            css=custom_css
        ) as interface:
            
            # Session state
            session_state = gr.State()
            
            # Header
            gr.Markdown("""
            # 🤖 Mortey Assistant
            ### Powered by LangGraph 0.4.8 + LangChain 0.3.64 + Python 3.13
            
            A sophisticated multi-agent assistant with specialized capabilities for coding, web search, and file management.
            """)
            
            # Main tabs
            with gr.Tabs():
                
                # Chat Tab
                with gr.Tab("💬 Chat", id="chat"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            chatbot = gr.Chatbot(
                                label="Chat with Mortey",
                                height=500,
                                show_copy_button=True,
                                show_share_button=True,
                                container=True,
                                elem_classes=["chat-container"]
                            )
                            
                            with gr.Row():
                                msg_input = gr.Textbox(
                                    placeholder="Type your message here... (e.g., 'write a Python function' or 'search for latest AI news')",
                                    label="Message",
                                    lines=3,
                                    max_lines=5,
                                    scale=4
                                )
                                send_btn = gr.Button("Send 📤", variant="primary", scale=1)
                            
                            with gr.Row():
                                clear_btn = gr.Button("Clear Chat 🧹", variant="secondary")
                                retry_btn = gr.Button("Retry Last 🔄", variant="secondary")
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### 🎯 Quick Actions")
                            
                            code_btn = gr.Button("💻 Code Help", variant="outline")
                            web_btn = gr.Button("🌐 Web Search", variant="outline")
                            file_btn = gr.Button("📁 File Management", variant="outline")
                            status_btn = gr.Button("📊 System Status", variant="outline")
                            
                            gr.Markdown("### 💡 Tips")
                            gr.Markdown("""
                            - **Coding**: Ask me to write, debug, or explain code
                            - **Research**: I can search the web for current information  
                            - **Files**: Upload files for analysis or ask me to create projects
                            - **Multi-step**: I can handle complex tasks across multiple tools
                            """)
                
                # File Management Tab
                with gr.Tab("📁 File Tools", id="files"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 📤 File Upload & Processing")
                            
                            file_upload = gr.File(
                                label="Upload File",
                                file_types=["text", ".py", ".js", ".html", ".css", ".json", ".yaml", ".md", ".csv"],
                                type="filepath"
                            )
                            
                            file_operation = gr.Dropdown(
                                choices=["analyze", "backup", "convert to markdown", "search content", "get info"],
                                label="Operation",
                                value="analyze"
                            )
                            
                            process_file_btn = gr.Button("Process File 🔄", variant="primary")
                            
                        with gr.Column():
                            file_result = gr.Textbox(
                                label="File Processing Result",
                                lines=15,
                                max_lines=20
                            )
                
                # System Status Tab
                with gr.Tab("📊 System Monitor", id="status"):
                    with gr.Column():
                        gr.Markdown("### 🖥️ Real-time System Status")
                        
                        status_display = gr.Markdown(
                            value="Click 'Refresh Status' to view system information",
                            elem_classes=["status-container"]
                        )
                        
                        with gr.Row():
                            refresh_status_btn = gr.Button("Refresh Status 🔄", variant="primary")
                            init_btn = gr.Button("Initialize Assistant 🚀", variant="secondary")
                        
                        init_result = gr.Textbox(
                            label="Initialization Result",
                            lines=3,
                            visible=False
                        )

                with gr.Tab("📊 Project Management (P6)", id="p6"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("### 🔌 P6 Connection")
                            
                            with gr.Row():
                                p6_username = gr.Textbox(
                                    label="P6 Username",
                                    type="text",
                                    placeholder="Enter P6 username"
                                )
                                p6_password = gr.Textbox(
                                    label="P6 Password", 
                                    type="password",
                                    placeholder="Enter P6 password"
                                )
                                p6_database = gr.Textbox(
                                    label="Database Name",
                                    value="PMDB",
                                    placeholder="P6 database name"
                                )
                            
                            connect_p6_btn = gr.Button("Connect to P6 🔌", variant="primary")
                            p6_connection_status = gr.Markdown("Status: Not connected")
                            
                            gr.Markdown("### 🎯 Quick P6 Actions")
                            
                            with gr.Row():
                                search_projects_btn = gr.Button("🔍 Search Projects", variant="outline")
                                project_schedule_btn = gr.Button("📅 Analyze Schedule", variant="outline") 
                                resource_analysis_btn = gr.Button("👥 Resource Analysis", variant="outline")
                                export_data_btn = gr.Button("📤 Export Data", variant="outline")
                            
                            project_query = gr.Textbox(
                                label="Project Management Query",
                                placeholder="Ask about projects, schedules, resources, or activities...",
                                lines=3
                            )
                            
                            execute_p6_query_btn = gr.Button("Execute Query 🚀", variant="primary")
                            
                        with gr.Column(scale=3):
                            p6_results = gr.Textbox(
                                label="P6 Query Results",
                                lines=20,
                                max_lines=25,
                                interactive=False
                            )
                            
                            with gr.Row():
                                clear_p6_btn = gr.Button("Clear Results 🧹", variant="secondary")
                                export_results_btn = gr.Button("Export Results 💾", variant="secondary")
            
            # Event handlers - Fixed for Gradio 5.33.2
            async def send_message(message, history, session_state):
                if not message.strip():
                    return "", history, session_state
                return await self.process_chat_message(message, history, session_state)
            
            async def quick_action(action_type, history, session_state):
                quick_messages = {
                    "code": "I need help with coding. Can you assist me with writing, debugging, or explaining code?",
                    "web": "Please search the web for the latest information on a topic I'll specify.",
                    "file": "I need help with file management. Can you help me organize, analyze, or process files?",
                    "status": "Please show me the current system status and health."
                }
                message = quick_messages.get(action_type, "Hello!")
                return await self.process_chat_message(message, history, session_state)

            async def connect_to_p6(username, password, database):
                """Connect to P6 system"""
                try:
                    if not username or not password:
                        return "❌ Please provide username and password"
                    
                    from tools.p6_tools import p6_tools_manager
                    await p6_tools_manager.initialize(username, password, database)
                    
                    return f"✅ Connected to P6 database: {database}"
                    
                except Exception as e:
                    return f"❌ Connection failed: {str(e)}"

            async def execute_p6_query(query, session_state):
                """Execute P6 query through assistant"""
                try:
                    if not query.strip():
                        return "Please enter a query", session_state
                    
                    # Process through assistant with P6 context
                    session_id, session_state = self.get_or_create_session_id(session_state)
                    
                    result = await self.assistant_core.process_message(
                        message=f"[P6 Query] {query}",
                        session_id=session_id,
                        user_id="gradio_p6_user"
                    )
                    
                    response = result.get('response', 'No response generated')
                    return response, session_state
                    
                except Exception as e:
                    error_msg = f"❌ P6 query failed: {str(e)}"
                    logger.error(error_msg)
                    return error_msg, session_state


            # Wire up P6 events
            connect_p6_btn.click(
                connect_to_p6,
                inputs=[p6_username, p6_password, p6_database],
                outputs=[p6_connection_status]
            )

            execute_p6_query_btn.click(
                execute_p6_query,
                inputs=[project_query, session_state],
                outputs=[p6_results, session_state]
            )

            # Wire up events - Gradio 5.33.2 compatible
            send_btn.click(
                send_message,
                inputs=[msg_input, chatbot, session_state],
                outputs=[msg_input, chatbot, session_state]
            )
            
            msg_input.submit(
                send_message,
                inputs=[msg_input, chatbot, session_state],
                outputs=[msg_input, chatbot, session_state]
            )
            
            clear_btn.click(
                self.clear_chat_history,
                inputs=[session_state],
                outputs=[chatbot, session_state]
            )
            
            # Quick action buttons
            code_btn.click(
                lambda h, s: quick_action("code", h, s),
                inputs=[chatbot, session_state],
                outputs=[chatbot, session_state]
            )
            
            web_btn.click(
                lambda h, s: quick_action("web", h, s),
                inputs=[chatbot, session_state],
                outputs=[chatbot, session_state]
            )
            
            file_btn.click(
                lambda h, s: quick_action("file", h, s),
                inputs=[chatbot, session_state],
                outputs=[chatbot, session_state]
            )
            
            status_btn.click(
                lambda h, s: quick_action("status", h, s),
                inputs=[chatbot, session_state],
                outputs=[chatbot, session_state]
            )
            
            # File processing
            process_file_btn.click(
                self.upload_and_process_file,
                inputs=[file_upload, file_operation, session_state],
                outputs=[file_result, session_state]
            )
            
            # System monitoring
            refresh_status_btn.click(
                self.get_system_status,
                outputs=[status_display]
            )
            
            async def init_assistant():
                result = await self.initialize_assistant()
                return result, gr.update(visible=True)
            
            init_btn.click(
                init_assistant,
                outputs=[init_result, init_result]
            )
            
            # Auto-initialize on startup
            interface.load(
                lambda: self.initialize_assistant(),
                outputs=[]
            )
        
        return interface

# Create the interface instance
gradio_app = GradioAssistantInterface()

def launch_app(
    share: bool = False,
    debug: bool = False,
    server_name: str = "127.0.0.1",
    server_port: int = 7860
):
    """Launch the Gradio 5.33.2 interface"""
    
    logger.info("🚀 Launching Gradio 5.33.2 Assistant Interface...")
    logger.info(f"📍 Server: http://{server_name}:{server_port}")
    
    # Create and launch the interface
    interface = gradio_app.create_interface()
    
    # ✅ FIXED: Gradio 5.33.2 compatible launch parameters
    interface.launch(
        share=share,
        debug=debug,
        server_name=server_name,
        server_port=server_port,
        show_error=True,
        # Removed deprecated parameters:
        # show_tips=True,  # Not available in Gradio 5.x
        # enable_queue=True,  # Queue is enabled by default
        # max_threads=10  # Not a launch parameter
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Mortey Assistant Gradio Interface")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--host", default="127.0.0.1", help="Server host address")
    parser.add_argument("--port", type=int, default=7860, help="Server port number")
    
    args = parser.parse_args()
    
    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🐛 Debug mode enabled")
    
    try:
        launch_app(
            share=args.share,
            debug=args.debug,
            server_name=args.host,
            server_port=args.port
        )
    except KeyboardInterrupt:
        logger.info("👋 Gracefully shutting down...")
    except Exception as e:
        logger.error(f"❌ Failed to launch app: {e}")
        logger.error(traceback.format_exc())
