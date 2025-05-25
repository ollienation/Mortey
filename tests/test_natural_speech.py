import asyncio
import sys
import signal
import os
sys.path.append('src')

from services.speech.speech_manager import SpeechManager, VoiceSettings

# Global reference for cleanup
speech_manager = None
cleanup_done = False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global cleanup_done
    if cleanup_done:
        print("\n🚨 Force exit!")
        os._exit(1)
    
    print(f"\n🛑 Received signal {signum}, cleaning up...")
    cleanup_done = True
    
    if speech_manager:
        speech_manager.cleanup()
    
    print("👋 Goodbye!")
    os._exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

async def test_natural_conversation():
    """Test natural conversation with proper cleanup"""
    global speech_manager
    
    print("🗣️ Testing Natural Conversation with Mortey")
    print("=" * 50)
    print("💡 Press Ctrl+C to exit gracefully")
    
    speech_manager = SpeechManager(VoiceSettings(rate=180, volume=0.9))
    
    try:
        await speech_manager.speak("Hello! I'm ready for natural conversation.")
        
        # Test 1: VAD-based listening
        print("\n1. Testing Voice Activity Detection...")
        print("Speak naturally - I'll detect when you're done:")
        
        result = await speech_manager.listen_with_vad(max_duration=10)
        if result:
            await speech_manager.speak(f"I understood: {result}")
        
        # Test 2: Continuous listening with shorter duration
        print("\n2. Testing Continuous Listening...")
        print("Starting continuous mode - speak with pauses:")
        
        conversation_count = 0
        
        async def conversation_callback(text):
            nonlocal conversation_count
            conversation_count += 1
            print(f"💬 Conversation {conversation_count}: {text}")
            await speech_manager.speak(f"Got it: {text}")
        
        # Start continuous listening
        await speech_manager.start_continuous_listening(callback=conversation_callback)
        
        # Shorter test duration with progress updates
        print("🎤 Continuous listening active for 15 seconds - speak with short pauses!")
        print("💡 Press Ctrl+C to stop early")
        
        for i in range(15):
            await asyncio.sleep(1)
            if i % 5 == 4:  # Every 5 seconds
                print(f"⏰ {15-i-1} seconds remaining...")
        
        # Stop continuous listening
        speech_manager.stop_continuous_listening()
        
        # Wait a moment for final processing
        await asyncio.sleep(2)
        
        print(f"\n✅ Processed {conversation_count} natural conversations!")
        
        await speech_manager.speak("Natural conversation test completed!")
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
    except Exception as e:
        print(f"❌ Test error: {e}")
    finally:
        # Always cleanup
        if speech_manager and not cleanup_done:
            speech_manager.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(test_natural_conversation())
    except KeyboardInterrupt:
        print("\n🛑 Program interrupted")
    except Exception as e:
        print(f"❌ Program error: {e}")
    finally:
        print("👋 Program finished")
        if not cleanup_done:
            os._exit(0)
