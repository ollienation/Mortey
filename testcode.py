# testcode.py

#!/usr/bin/env python3
# test_coder_agent.py - Testing script for the coder agent
import asyncio
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

# Configure logging for the test
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_coder_agent.log')
    ]
)

logger = logging.getLogger("test_coder_agent")

async def test_coder_agent():
    """Test the coder agent with a GUI creation task"""
    
    logger.info("🚀 Starting Coder Agent Test")
    
    try:
        # Import components - with fallback handling for missing modules
        logger.info("📦 Importing components...")
        
        try:
            from core.assistant_core import AssistantCore
        except ImportError as e:
            logger.error(f"❌ Failed to import AssistantCore: {e}")
            logger.error("Make sure core/supervisor.py and core/checkpointer.py exist")
            return False
        
        from config.settings import config
        from agents.agents import agent_factory
        
        logger.info("✅ Components imported successfully")
        
        # Initialize the assistant core
        logger.info("🔧 Initializing Assistant Core...")
        assistant = AssistantCore()
        
        # Initialize the system
        logger.info("⚙️ Initializing system components...")
        await assistant.initialize()
        
        logger.info("✅ Assistant Core initialized successfully")
        
        # Verify coder agent is available
        logger.info("🤖 Checking agent availability...")
        agents = agent_factory.agents
        
        if 'coder' not in agents:
            logger.error("❌ Coder agent not found in available agents")
            logger.info(f"Available agents: {list(agents.keys())}")
            return False
        
        logger.info(f"✅ Coder agent found. Available agents: {list(agents.keys())}")
        
        # Define the test message
        test_message = """Please create a simple GUI application using Python's tkinter library. The requirements are:

1. Create a window with a green background
2. Add a red button in the center
3. The button should print "Button clicked!" when pressed
4. Save this as gui.py in the workspace

Please write clean, well-commented code and make sure it's ready to run."""
        
        logger.info("📝 Sending test message to coder agent...")
        logger.info(f"Message: {test_message[:100]}...")
        
        # Process the message through the assistant
        response = await assistant.process_message(
            message=test_message,
            session_id="test_coder_sesh",
            user_id="test_user"
        )
        
        logger.info("✅ Received response from coder agent")
        logger.info(f"Response length: {len(response.get('response', ''))}")
        
        # Log the response
        if response.get('response'):
            logger.info("📄 Coder Agent Response:")
            logger.info("-" * 50)
            logger.info(response['response'])
            logger.info("-" * 50)
        else:
            logger.warning("⚠️ Empty response from coder agent")
        
        # Check if gui.py was created
        gui_file_path = config.workspace_dir / "gui.py"
        
        if gui_file_path.exists():
            logger.info("✅ SUCCESS: gui.py file was created!")
            
            # Read and display the created file
            with open(gui_file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            logger.info("📄 Created gui.py content:")
            logger.info("-" * 50)
            logger.info(file_content)
            logger.info("-" * 50)
            
            # Basic validation of the file content
            validation_checks = [
                ("tkinter import", "tkinter" in file_content.lower() or "tk" in file_content),
                ("green background", "green" in file_content.lower()),
                ("red button", "red" in file_content.lower()),
                ("button creation", "button" in file_content.lower()),
                ("window/root creation", any(word in file_content.lower() for word in ["root", "window", "tk()"])),
            ]
            
            logger.info("🔍 Validating file content:")
            all_passed = True
            for check_name, condition in validation_checks:
                if condition:
                    logger.info(f"  ✅ {check_name}: PASS")
                else:
                    logger.warning(f"  ❌ {check_name}: FAIL")
                    all_passed = False
            
            if all_passed:
                logger.info("🎉 ALL VALIDATION CHECKS PASSED!")
            else:
                logger.warning("⚠️ Some validation checks failed")
            
            return True
            
        else:
            logger.error("❌ FAILURE: gui.py file was not created")
            logger.info(f"Expected location: {gui_file_path}")
            
            # List files in workspace for debugging
            try:
                workspace_files = list(config.workspace_dir.iterdir())
                logger.info(f"Files in workspace: {[f.name for f in workspace_files]}")
            except Exception as e:
                logger.error(f"Could not list workspace files: {e}")
            
            return False
    
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return False
    
    finally:
        # Cleanup
        try:
            logger.info("🧹 Performing cleanup...")
            await assistant.graceful_shutdown()
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

async def run_additional_tests():
    """Run additional tests with enhanced debugging"""
    
    logger.info("🔬 Running additional coder agent tests...")
    
    # ✅ ADD MISSING IMPORT:
    from config.settings import config
    
    test_cases = [
        {
            "name": "Simple Python Script",
            "message": "Create a simple Python script that prints 'Hello World' and save it as hello.py",
            "expected_file": "hello.py",
            "validation": lambda content: "hello" in content.lower() and "world" in content.lower()
        },
        {
            "name": "JSON File Creation", 
            "message": "Create a JSON file with sample user data (name, age, email) and save it as sample_data.json",
            "expected_file": "sample_data.json",
            "validation": lambda content: content.strip().startswith('{') and content.strip().endswith('}')
        }
    ]
    
    try:
        from core.assistant_core import AssistantCore
        assistant = AssistantCore()
        await assistant.initialize()
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"🧪 Running test {i}: {test_case['name']}")
            
            response = await assistant.process_message(
                message=test_case['message'],
                session_id=f"test_sesh_{i}",
                user_id="test_user"
            )
            
            # ✅ DEBUG THE RESPONSE:
            logger.info(f"  📄 Response: {response}")
            logger.info(f"  📄 Response length: {len(response.get('response', ''))}")
            
            expected_path = config.workspace_dir / test_case['expected_file']
            
            if expected_path.exists():
                logger.info(f"  ✅ File created: {test_case['expected_file']}")
                
                with open(expected_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                logger.info(f"  📄 File content length: {len(content)}")
                
                if test_case['validation'](content):
                    logger.info(f"  ✅ Content validation passed")
                else:
                    logger.warning(f"  ⚠️ Content validation failed")
                    logger.info(f"  📄 Content preview: {content[:200]}...")
            else:
                logger.error(f"  ❌ File not created: {test_case['expected_file']}")
                
                # List workspace files for debugging
                try:
                    workspace_files = list(config.workspace_dir.iterdir())
                    logger.info(f"  📂 Workspace files: {[f.name for f in workspace_files]}")
                except Exception as e:
                    logger.error(f"  Could not list workspace files: {e}")
        
        await assistant.graceful_shutdown()
        
    except Exception as e:
        logger.error(f"❌ Additional tests failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")

def main():
    """Main test runner"""
    
    print("=" * 60)
    print("🤖 MORTEY CODER AGENT TEST SUITE")
    print("=" * 60)
    
    try:
        # Run main test
        success = asyncio.run(test_coder_agent())
        
        if success:
            print("\n🎉 MAIN TEST PASSED!")
            
            # Ask if user wants to run additional tests
            try:
                run_more = input("\nRun additional tests? (y/n): ").lower().strip()
                if run_more in ['y', 'yes']:
                    print("\n" + "=" * 40)
                    asyncio.run(run_additional_tests())
            except KeyboardInterrupt:
                print("\n👋 Test interrupted by user")
            
        else:
            print("\n❌ MAIN TEST FAILED!")
            return 1
            
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 60)
    print("🏁 TEST SUITE COMPLETED")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
