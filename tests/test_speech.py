# Set ALSA environment variables before importing anything
import os
os.environ['ALSA_PCM_CARD'] = '0'
os.environ['ALSA_PCM_DEVICE'] = '0'
os.environ['ALSA_CARD'] = '0'

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import asyncio
import sys
sys.path.append('src')

from services.speech.speech_manager import SpeechManager, VoiceSettings


async def test_speech_manager():
    """Test speech capabilities with Bumblebee wake word"""
    
    print("🐝 Testing Mortey Speech Manager with Bumblebee Wake Word")
    print("=" * 60)
    
    # Initialize speech manager
    voice_settings = VoiceSettings(rate=180, volume=0.9)
    speech_manager = SpeechManager(voice_settings)
    
    # Test 1: Text-to-Speech
    print("\n1. Testing Text-to-Speech...")
    await speech_manager.speak("Hello! I am Mortey, your voice assistant. Say Bumblebee to wake me up!")
    
    # Test 2: Speech-to-Text
    print("\n2. Testing Speech-to-Text...")
    print("Say something (you have 5 seconds):")
    
    result = await speech_manager.listen_once(timeout=5)
    if result and result != "timeout":
        print(f"✅ You said: '{result}'")
        await speech_manager.speak(f"I heard you say: {result}")
    else:
        print("❌ No speech detected or timeout")
    
    # Test 3: Bumblebee Wake Word Detection
    print("\n3. Testing Bumblebee Wake Word Detection...")
    print("🐝 Say 'Bumblebee' to test wake word detection (15 seconds):")
    
    wake_word_detected = False
    
    async def wake_word_callback(wake_word, full_text):
        nonlocal wake_word_detected
        wake_word_detected = True
        print(f"🎯 Wake word '{wake_word}' detected!")
        await speech_manager.speak("Bumblebee detected! I'm listening for your command.")
        speech_manager.stop_wake_word_detection()
    
    # Start wake word detection
    await speech_manager.start_wake_word_detection(callback=wake_word_callback)
    
    # Wait for wake word or timeout
    for i in range(15):
        if wake_word_detected:
            break
        await asyncio.sleep(1)
        if i % 3 == 0:
            print(f"   🐝 Listening for 'Bumblebee'... {15-i}s remaining")
    
    if not wake_word_detected:
        print("⏱️ Wake word test timed out")
        speech_manager.stop_wake_word_detection()
        await speech_manager.speak("Wake word test completed. No bumblebee detected.")
    
    # Test 4: Audio level monitoring
    print("\n4. Testing Audio Level Monitoring...")
    print("Make some noise for 3 seconds:")
    
    for i in range(30):  # 3 seconds at 10Hz
        level = speech_manager.get_audio_level()
        bar = "█" * int(level * 20)
        print(f"\rAudio Level: [{bar:<20}] {level:.2f}", end="", flush=True)
        await asyncio.sleep(0.1)
    
    print("\n")
    
    # Cleanup
    speech_manager.cleanup()
    
    print("\n🎉 Bumblebee speech testing complete!")
    await speech_manager.speak("Speech system test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_speech_manager())
