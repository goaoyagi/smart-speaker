#!/usr/bin/env python3
"""
Speaker module - Text-to-speech using Piper-Plus
"""

from piper import PiperVoice
import tempfile
import wave
import os

# Configuration
SPEAKER_DEVICE = os.getenv("SPEAKER_DEVICE", "plughw:0,0")
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "./models/tsukuyomi.onnx")
PIPER_CONFIG_PATH = os.getenv("PIPER_CONFIG_PATH", "./models/config.json")


class Speaker:
    def __init__(self):
        print("Initializing Speaker (Piper-Plus)...")
        self.piper_model = PiperVoice.load(
            PIPER_MODEL_PATH,
            config_path=PIPER_CONFIG_PATH
        )
        self.speaker_device = SPEAKER_DEVICE
        print("Speaker initialized!")
    
    def speak(self, text):
        """Convert text to speech using Piper-Plus"""
        print(f"Speaking: {text}")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_file = f.name
        
        # Synthesize audio directly to WAV file
        with wave.open(temp_file, 'wb') as wav_file:
            self.piper_model.synthesize(
                text,
                wav_file,
                speaker_id=0,
                length_scale=1.0
            )
        
        # Play audio
        os.system(f"aplay -D {self.speaker_device} {temp_file}")
        
        # Clean up
        os.unlink(temp_file)
