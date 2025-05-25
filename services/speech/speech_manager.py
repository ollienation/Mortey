# Set ALSA environment variables FIRST
import os
from config.settings import config

# Set ALSA environment variables dynamically
def setup_audio_environment():
    """Setup audio environment variables"""
    # Only set if not already configured
    if not os.getenv('ALSA_PCM_CARD'):
        os.environ['ALSA_PCM_CARD'] = '0'
        os.environ['ALSA_PCM_DEVICE'] = '0'
        os.environ['ALSA_CARD'] = '0'
        os.environ['JACK_NO_AUDIO_RESERVATION'] = '1'
        os.environ['JACK_NO_START_SERVER'] = '1'

# Call setup before other imports
setup_audio_environment()
# Suppress ALSA errors at the C library level
import ctypes
from ctypes import *

# Error handler to suppress ALSA messages
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

def py_error_handler(filename, line, function, err, fmt):
    pass  # Suppress all ALSA errors

c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

try:
    asound = cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except:
    pass  # If we can't load ALSA lib, continue anyway

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Load other stuff
import numpy as np
import sys
import warnings
import asyncio
import threading
import queue
import time
import struct
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Import audio libraries AFTER environment setup
import speech_recognition as sr
import pyttsx3
import numpy as np
import pyaudio
import pvporcupine

class SpeechState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    WAKE_WORD_DETECTION = "wake_word_detection"

@dataclass
class VoiceSettings:
    """Voice configuration settings"""
    rate: int = 180          # Words per minute
    volume: float = 0.9      # Volume level (0.0 to 1.0)
    voice_id: int = 0        # Voice selection (0 for default)
    language: str = "en"     # Language code

@dataclass
class AudioConfig:
    """Audio recording configuration"""
    sample_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1
    format: int = pyaudio.paInt16

class SpeechManager:
    """Comprehensive speech management for Mortey with Porcupine wake word detection"""
    
    def __init__(self, voice_settings: VoiceSettings = None, audio_config: AudioConfig = None):
        self.voice_settings = voice_settings or VoiceSettings()
        self.audio_config = audio_config or AudioConfig()
        
        # Initialize components
        self._init_tts()
        self._init_stt()
        self._init_audio()
        
        # State management
        self.current_state = SpeechState.IDLE
        self.is_listening = False
        self.wake_word_active = False
        
        # Porcupine components
        self.porcupine = None
        self.porcupine_audio_stream = None
        
        # Callbacks
        self.wake_word_callback: Optional[Callable] = None
        self.speech_callback: Optional[Callable] = None
        self.state_change_callback: Optional[Callable] = None
        
        # Wakeword cooldown
        self.last_wake_word_time = 0
        self.wake_word_cooldown = 3.0  # 3 seconds cooldown
        self.is_speaking = False  # Track TTS state

        # Add continuous listening state
        self.continuous_listening = False
        self.background_listener = None
        self.speech_buffer = []
        self.last_speech_time = 0
        self.silence_threshold = 1.5  # Seconds of silence before processing
        self.min_speech_duration = 0.5  # Minimum speech length to process
        
        # Audio level monitoring for visualizer
        self.audio_levels = queue.Queue(maxsize=100)

        # Interrupt tts
        self.tts_interrupted = False

        # Use config for audio device if specified
        if config.audio_device_index is not None:
            self.microphone = sr.Microphone(device_index=config.audio_device_index)
        else:
            self.microphone = sr.Microphone(device_index=None)  # Use default
        
    def _init_tts(self):
        """Initialize Text-to-Speech engine"""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configure voice settings
            self.tts_engine.setProperty('rate', self.voice_settings.rate)
            self.tts_engine.setProperty('volume', self.voice_settings.volume)
            
            # Get available voices and set preferred voice
            voices = self.tts_engine.getProperty('voices')
            if voices and len(voices) > self.voice_settings.voice_id:
                self.tts_engine.setProperty('voice', voices[self.voice_settings.voice_id].id)
            
            print(f"✅ TTS initialized with voice: {voices[self.voice_settings.voice_id].name if voices else 'default'}")
            
        except Exception as e:
            print(f"❌ TTS initialization failed: {e}")
            self.tts_engine = None
        
    def _init_stt(self):
        """Initialize Speech-to-Text with Intel PCH audio"""
        try:
            self.recognizer = sr.Recognizer()
            
            # Specifically use Intel PCH (card 0) for microphone
            # This ensures we're using the analog audio interface
            self.microphone = sr.Microphone(device_index=None)  # Default is usually card 0
            
            print("🎤 Calibrating microphone (Intel PCH CX8070)...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            print("✅ STT initialized with Intel PCH CX8070 Analog")
            
        except Exception as e:
            print(f"❌ STT initialization failed: {e}")
            self.recognizer = None
            self.microphone = None

    def _init_audio(self):
        """Initialize PyAudio with Intel PCH"""
        try:
            self.audio = pyaudio.PyAudio()
            print("✅ Audio system initialized (Intel PCH CX8070)")
        except Exception as e:
            print(f"❌ Audio initialization failed: {e}")
            self.audio = None

    async def speak(self, text: str, interrupt_current: bool = True) -> bool:
        """FIXED: Better interrupt handling"""
        if not self.tts_engine:
            print(f"❌ TTS not available, would say: {text}")
            return False
        
        try:
            self.is_speaking = True
            self.tts_interrupted = False  # Reset interrupt flag
            self._set_state(SpeechState.SPEAKING)
            
            if interrupt_current:
                self.tts_engine.stop()
            
            # Clean text before speaking (fixes issue 2)
            clean_text = self._clean_text_for_speech(text)
            print(f"🔊 Speaking: {clean_text[:50]}{'...' if len(clean_text) > 50 else ''}")
            
            # Start TTS in thread
            await asyncio.to_thread(self._speak_sync_interruptible, clean_text)
            
            # Only wait if not interrupted
            if not self.tts_interrupted:
                await asyncio.sleep(0.3)
            
            self.is_speaking = False
            self._set_state(SpeechState.IDLE)
            return True
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
            self.is_speaking = False
            self._set_state(SpeechState.IDLE)
            return False
            
    def _speak_sync_interruptible(self, text: str):
        """TTS that can be interrupted"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            # TTS was interrupted or failed
            pass

    def _clean_text_for_speech(self, text: str) -> str:
        """Clean text for better TTS output - removes special characters"""
        import re
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # **bold** -> bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # *italic* -> italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # `code` -> code
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Replace problematic punctuation
        text = text.replace('&', ' and ')
        text = text.replace('@', ' at ')
        text = text.replace('#', ' number ')
        text = text.replace('%', ' percent ')
        text = text.replace('$', ' dollars ')
        text = text.replace('€', ' euros ')
        
        # Remove excessive punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:]', ' ', text)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()


    def interrupt_speech(self):
        """Interrupt current TTS immediately - ENHANCED VERSION"""
        if self.is_speaking and self.tts_engine:
            print("🛑 Interrupting TTS for wake word")
            try:
                # Stop TTS engine immediately
                self.tts_engine.stop()
                
                # Set interrupt flags
                self.tts_interrupted = True
                self.is_speaking = False
                
                # Force state change
                self._set_state(SpeechState.IDLE)
                
            except Exception as e:
                print(f"❌ TTS interrupt error: {e}")
                # Force flags anyway
                self.tts_interrupted = True
                self.is_speaking = False

    async def listen_once(self, timeout: int = 10, phrase_time_limit: int = 10) -> Optional[str]:
        """Listen for a single speech input"""
        if not self.recognizer or not self.microphone:
            print("❌ STT not available")
            return None
        
        try:
            self._set_state(SpeechState.LISTENING)
            print(f"👂 Listening for {timeout}s...")
            
            # Listen for audio
            with self.microphone as source:
                try:
                    audio = self.recognizer.listen(
                        source, 
                        timeout=timeout, 
                        phrase_time_limit=phrase_time_limit
                    )
                except sr.WaitTimeoutError:
                    print("⏱️ Listening timeout")
                    self._set_state(SpeechState.IDLE)
                    return "timeout"
            
            self._set_state(SpeechState.PROCESSING)
            print("🧠 Processing speech...")
            
            # Use local Whisper for privacy and reliability
            text = await asyncio.to_thread(
                self.recognizer.recognize_whisper, 
                audio, 
                language="english"
            )
            
            print(f"✅ Recognized: {text}")
            self._set_state(SpeechState.IDLE)
            return text
            
        except sr.UnknownValueError:
            print("❓ Could not understand audio")
            self._set_state(SpeechState.IDLE)
            return None
        except Exception as e:
            print(f"❌ STT error: {e}")
            self._set_state(SpeechState.IDLE)
            return None
    
    async def start_wake_word_detection(self, callback: Callable = None):
        """Start Porcupine wake word detection with debouncing"""
        if not self.audio:
            print("❌ Audio system not available")
            return
        
        try:
            # Initialize Porcupine with LOWER sensitivity to reduce false positives
            access_key = os.getenv("PICOVOICE_ACCESS_KEY")
            if not access_key:
                print("❌ PICOVOICE_ACCESS_KEY not found in .env file")
                return
            
            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=['bumblebee'],
                sensitivities=[0.3]  # Reduced from 0.5 to 0.3 for less sensitivity
            )
            
            self.wake_word_callback = callback
            self.wake_word_active = True
            self._set_state(SpeechState.WAKE_WORD_DETECTION)
            
            print(f"🐝 Porcupine wake word detection started. Say 'Bumblebee' to activate!")
            
            # Start audio processing in background thread
            self._start_porcupine_thread()
            
        except Exception as e:
            print(f"❌ Porcupine initialization failed: {e}")

    def _start_porcupine_thread(self):
        """Start Porcupine audio processing with TTS interrupt capability"""
        
        # Get the main event loop BEFORE starting the thread
        try:
            main_loop = asyncio.get_running_loop()
        except RuntimeError:
            main_loop = asyncio.get_event_loop()
        
        def porcupine_worker():
            try:
                audio_stream = self.audio.open(
                    rate=self.porcupine.sample_rate,
                    channels=1,
                    format=pyaudio.paInt16,
                    input=True,
                    frames_per_buffer=self.porcupine.frame_length
                )
                
                while self.wake_word_active and self.porcupine:
                    try:
                        # Read audio data
                        pcm = audio_stream.read(self.porcupine.frame_length, exception_on_overflow=False)
                        pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                        
                        # Process with Porcupine
                        keyword_index = self.porcupine.process(pcm)
                        
                        if keyword_index >= 0:
                            current_time = time.time()
                            
                            # ENHANCED: Handle TTS interruption properly
                            if self.is_speaking:
                                # Interrupt TTS immediately
                                self.interrupt_speech()
                                print(f"🐝 Bumblebee detected - interrupting speech!")
                                
                                # Wait a moment for TTS to stop
                                time.sleep(0.1)
                                
                                # Start new conversation immediately (only if not in cooldown)
                                if current_time - self.last_wake_word_time > 1.0:  # Shorter cooldown during interrupts
                                    self.last_wake_word_time = current_time
                                    if self.wake_word_callback:
                                        asyncio.run_coroutine_threadsafe(
                                            self.wake_word_callback("bumblebee", "interrupt_detection"),
                                            main_loop
                                        )
                            
                            elif (current_time - self.last_wake_word_time > self.wake_word_cooldown):
                                # Normal wake word detection
                                print(f"🐝 Bumblebee wake word detected!")
                                self.last_wake_word_time = current_time
                                
                                if self.wake_word_callback:
                                    asyncio.run_coroutine_threadsafe(
                                        self.wake_word_callback("bumblebee", "porcupine_detection"),
                                        main_loop
                                    )
                            else:
                                # Cooldown period
                                print(f"⏱️ Wake word ignored (cooldown: {self.wake_word_cooldown - (current_time - self.last_wake_word_time):.1f}s)")
                                
                    except Exception as e:
                        if self.wake_word_active:
                            print(f"❌ Porcupine processing error: {e}")
                        break
                
                audio_stream.close()
                
            except Exception as e:
                print(f"❌ Porcupine thread error: {e}")
        
        self.porcupine_thread = threading.Thread(target=porcupine_worker, daemon=True)
        self.porcupine_thread.start()

    def stop_wake_word_detection(self):
        """Stop Porcupine wake word detection"""
        self.wake_word_active = False
        
        if self.porcupine:
            self.porcupine.delete()
            self.porcupine = None
        
        print("🛑 Bumblebee wake word detection stopped")
        self._set_state(SpeechState.IDLE)

    async def start_continuous_listening(self, callback: Callable = None):
        """Start continuous listening with improved real-time processing"""
        if not self.recognizer or not self.microphone:
            print("❌ STT not available for continuous listening")
            return
        
        self.continuous_listening = True
        self.speech_callback = callback
        
        # Initialize timing
        self.last_speech_time = time.time()
        self.speech_buffer = []
        
        # IMPROVED: Shorter silence threshold for more responsive processing
        self.silence_threshold = 0.8  # Reduced from 1.5 to 0.8 seconds
        
        print("🎤 Starting continuous listening with improved VAD...")
        
        try:
            main_loop = asyncio.get_running_loop()
        except RuntimeError:
            main_loop = asyncio.get_event_loop()
        
        def audio_callback(recognizer, audio):
            try:
                current_time = time.time()
                
                # Energy calculation
                audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
                energy = np.sqrt(np.mean(audio_data**2))
                
                voice_threshold = 30
                
                if energy > voice_threshold:
                    # Speech detected
                    self.last_speech_time = current_time
                    self.speech_buffer.append(audio)
                    print(f"🗣️ Speech detected (energy: {energy:.0f})")
                    
                else:
                    # Check for processing conditions
                    silence_duration = current_time - self.last_speech_time
                    
                    # IMPROVED: Process buffer more frequently
                    should_process = (
                        self.speech_buffer and (
                            silence_duration > self.silence_threshold or  # Normal silence threshold
                            len(self.speech_buffer) >= 10  # Or if buffer gets too large
                        )
                    )
                    
                    if should_process:
                        print(f"🔄 Processing {len(self.speech_buffer)} segments (silence: {silence_duration:.1f}s)")
                        asyncio.run_coroutine_threadsafe(
                            self._process_speech_buffer(), 
                            main_loop
                        )
                        
            except Exception as e:
                print(f"❌ Continuous listening error: {e}")
        
        # Start background listening
        self.background_listener = self.recognizer.listen_in_background(
            self.microphone,
            audio_callback,
            phrase_time_limit=None
        )
        
        print("✅ Continuous listening active - speak naturally!")

    async def _process_speech_buffer(self):
        """Process accumulated speech from buffer"""
        if not self.speech_buffer:
            print("⚠️ Speech buffer is empty - nothing to process")
            return
        
        try:
            print(f"🧠 Processing speech buffer with {len(self.speech_buffer)} segments...")
            
            # Combine audio segments
            combined_audio = self.speech_buffer[0]
            for audio_segment in self.speech_buffer[1:]:
                combined_audio = sr.AudioData(
                    combined_audio.get_raw_data() + audio_segment.get_raw_data(),
                    combined_audio.sample_rate,
                    combined_audio.sample_width
                )
            
            # Clear buffer BEFORE processing to prevent race conditions
            buffer_size = len(self.speech_buffer)
            self.speech_buffer = []
            
            # Recognize speech
            text = await asyncio.to_thread(
                self.recognizer.recognize_whisper,
                combined_audio,
                language="english"
            )
            
            if text.strip():
                print(f"✅ Continuous recognition ({buffer_size} segments): {text}")
                
                # Call the speech callback if provided
                if self.speech_callback:
                    await self.speech_callback(text.strip())
            else:
                print("❓ Empty text result from speech buffer")
            
        except sr.UnknownValueError:
            print("❓ Could not understand speech in buffer")
        except Exception as e:
            print(f"❌ Speech buffer processing error: {e}")

    async def listen_with_vad(self, max_duration: int = 30) -> Optional[str]:
        """Listen with Voice Activity Detection - stops when user stops speaking"""
        if not self.recognizer or not self.microphone:
            print("❌ STT not available")
            return None
        
        try:
            self._set_state(SpeechState.LISTENING)
            print("👂 Listening with VAD - speak naturally, I'll detect when you're done...")
            
            # Use a longer timeout but with VAD logic
            with self.microphone as source:
                # Start with ambient noise adjustment
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen with dynamic timeout
                audio = self.recognizer.listen(
                    source,
                    timeout=2,  # Initial timeout to start
                    phrase_time_limit=max_duration  # Maximum duration
                )
            
            self._set_state(SpeechState.PROCESSING)
            print("🧠 Processing natural speech...")
            
            # Recognize with Whisper
            text = await asyncio.to_thread(
                self.recognizer.recognize_whisper,
                audio,
                language="english"
            )
            
            print(f"✅ VAD Recognition: {text}")
            self._set_state(SpeechState.IDLE)
            return text
            
        except sr.WaitTimeoutError:
            print("⏱️ No speech detected")
            self._set_state(SpeechState.IDLE)
            return None
        except sr.UnknownValueError:
            print("❓ Could not understand speech")
            self._set_state(SpeechState.IDLE)
            return None
        except Exception as e:
            print(f"❌ VAD listening error: {e}")
            self._set_state(SpeechState.IDLE)
            return None

    async def start_continuous_listening_debug(self, callback: Callable = None):
        """Debug version with energy monitoring"""
        if not self.recognizer or not self.microphone:
            print("❌ STT not available for continuous listening")
            return
        
        self.continuous_listening = True
        self.speech_callback = callback
        
        # CRITICAL FIX: Initialize last_speech_time
        self.last_speech_time = time.time()
        self.speech_buffer = []
        
        print("🎤 Starting DEBUG continuous listening...")
        
        try:
            main_loop = asyncio.get_running_loop()
        except RuntimeError:
            main_loop = asyncio.get_event_loop()
        
        def audio_callback(recognizer, audio):
            try:
                current_time = time.time()
                
                # Energy calculation
                audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
                energy = np.sqrt(np.mean(audio_data**2))
                
                # Show energy levels for debugging
                if energy > 30:  # Show any significant audio
                    print(f"🔊 Audio energy: {energy:.0f} (threshold: 40)")
                
                voice_threshold = 30  # FIXED: Lowered from 200 to 40
                
                if energy > voice_threshold:
                    self.last_speech_time = current_time
                    self.speech_buffer.append(audio)
                    print(f"🗣️ SPEECH DETECTED! Energy: {energy:.0f}")
                    
                else:
                    silence_duration = current_time - self.last_speech_time
                    
                    if (self.speech_buffer and 
                        silence_duration > self.silence_threshold):
                        
                        print(f"🔄 Processing {len(self.speech_buffer)} audio segments after {silence_duration:.1f}s silence")
                        asyncio.run_coroutine_threadsafe(
                            self._process_speech_buffer(), 
                            main_loop
                        )
                        
            except Exception as e:
                print(f"❌ Debug listening error: {e}")
        
        self.background_listener = self.recognizer.listen_in_background(
            self.microphone,
            audio_callback,
            phrase_time_limit=None
        )
        
        print("✅ DEBUG continuous listening active!")

    def stop_continuous_listening(self):
        """Stop continuous listening with immediate buffer processing"""
        print("🛑 Stopping continuous listening...")
        self.continuous_listening = False
        
        if self.background_listener:
            try:
                self.background_listener(wait_for_stop=True)  # Wait for proper stop
                self.background_listener = None
            except Exception as e:
                print(f"⚠️ Background listener stop warning: {e}")
        
        # IMPROVED: Process any remaining speech immediately
        if self.speech_buffer:
            print(f"🔄 Processing final buffer with {len(self.speech_buffer)} segments...")
            # Use asyncio.create_task for immediate processing
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._process_speech_buffer())
            except RuntimeError:
                # If no event loop, process synchronously
                asyncio.run(self._process_speech_buffer())
        
        print("✅ Continuous listening stopped")
    
    def adjust_vad_sensitivity(self, voice_threshold: int = 30, silence_threshold: float = 1.5):
        """Adjust Voice Activity Detection sensitivity"""
        self.voice_threshold = voice_threshold
        self.silence_threshold = silence_threshold
        print(f"🎛️ VAD adjusted: voice={voice_threshold}, silence={silence_threshold}s")
    
    def get_audio_level(self) -> float:
        """Get current audio input level for visualizer"""
        if not self.audio:
            return 0.0
        
        try:
            # Quick audio level check
            stream = self.audio.open(
                format=self.audio_config.format,
                channels=self.audio_config.channels,
                rate=self.audio_config.sample_rate,
                input=True,
                frames_per_buffer=self.audio_config.chunk_size
            )
            
            data = stream.read(self.audio_config.chunk_size, exception_on_overflow=False)
            stream.close()
            
            # Convert to numpy array and calculate RMS
            audio_data = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_data**2))
            
            # Normalize to 0-1 range
            level = min(rms / 3000.0, 1.0)
            
            # Add to queue for visualizer
            if not self.audio_levels.full():
                self.audio_levels.put(level)
            
            return level
            
        except Exception as e:
            return 0.0
    
    def _set_state(self, new_state: SpeechState):
        """Update speech state and notify callback"""
        if self.current_state != new_state:
            old_state = self.current_state
            self.current_state = new_state
            
            if self.state_change_callback:
                try:
                    self.state_change_callback(old_state, new_state)
                except Exception as e:
                    print(f"❌ State change callback error: {e}")
    
    def get_available_voices(self) -> list:
        """Get list of available TTS voices"""
        if not self.tts_engine:
            return []
        
        try:
            voices = self.tts_engine.getProperty('voices')
            return [{"id": i, "name": voice.name, "lang": voice.languages} 
                   for i, voice in enumerate(voices)]
        except:
            return []
    
    def set_voice(self, voice_id: int):
        """Change TTS voice"""
        if not self.tts_engine:
            return False
        
        try:
            voices = self.tts_engine.getProperty('voices')
            if 0 <= voice_id < len(voices):
                self.tts_engine.setProperty('voice', voices[voice_id].id)
                self.voice_settings.voice_id = voice_id
                print(f"🎵 Voice changed to: {voices[voice_id].name}")
                return True
        except Exception as e:
            print(f"❌ Voice change failed: {e}")
        
        return False
    
    def cleanup(self):
        """Enhanced cleanup that actually stops everything"""
        try:
            print("🧹 Starting cleanup...")
            
            # Force stop continuous listening first
            if self.continuous_listening:
                self.continuous_listening = False
                if self.background_listener:
                    try:
                        self.background_listener(wait_for_stop=False)  # Don't wait, force stop
                        self.background_listener = None
                    except:
                        pass
            
            # Force stop wake word detection
            if self.wake_word_active:
                self.wake_word_active = False
                if self.porcupine:
                    try:
                        self.porcupine.delete()
                        self.porcupine = None
                    except:
                        pass
            
            # Stop TTS
            if self.tts_engine:
                try:
                    self.tts_engine.stop()
                except:
                    pass
            
            # Terminate audio
            if self.audio:
                try:
                    self.audio.terminate()
                    self.audio = None
                except:
                    pass
            
            # Clear buffers
            self.speech_buffer = []
            
            print("✅ Cleanup completed")
            
        except Exception as e:
            print(f"❌ Cleanup error: {e}")