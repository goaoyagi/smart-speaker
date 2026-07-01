#!/usr/bin/env python3
"""
Listener module - Speech recognition using Whisper
"""

import numpy as np
from faster_whisper import WhisperModel
import tempfile
import subprocess
import wave
import shutil
import os

# Configuration
MIC_DEVICE = os.getenv("MIC_DEVICE", "hw:0,0")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
CHANNELS = int(os.getenv("CHANNELS", "1"))
RECORD_SECONDS = int(os.getenv("RECORD_SECONDS", "10"))
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")


class Listener:
    def __init__(self):
        print("Initializing Listener (Whisper)...")
        self.whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8"
        )
        print("Listener initialized!")
    
    def record_audio(self, duration=RECORD_SECONDS):
        """Record audio from USB microphone using arecord"""
        print(f"Recording for {duration} seconds...")
        
        # Record to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_file = f.name
        
        # Use arecord to capture audio
        cmd = [
            'arecord',
            '-f', 'S16_LE',
            '-r', str(SAMPLE_RATE),
            '-c', str(CHANNELS),
            '-D', MIC_DEVICE,
            '-d', str(duration),
            temp_file
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Load audio file
        with wave.open(temp_file, 'rb') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            audio_array = np.frombuffer(frames, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            
            # Debug: check audio levels
            max_level = np.max(np.abs(audio_array))
            print(f"Audio max level: {max_level:.4f}")
            if max_level < 0.01:
                print("Warning: Audio level is very low. Microphone may not be working.")
        
        # Save a copy for analysis
        shutil.copy(temp_file, "/home/gopi/smart-speaker/last_record.wav")
        print("Saved a copy of record to /home/gopi/smart-speaker/last_record.wav")
        
        # Clean up
        os.unlink(temp_file)
        
        return audio_array
    
    def transcribe(self, audio_array):
        """Transcribe audio to text using Whisper"""
        print("Transcribing...")
        
        segments, info = self.whisper_model.transcribe(
            audio_array,
            language="ja",
            beam_size=5
        )
        
        text = ""
        for segment in segments:
            text += segment.text
        
        print(f"Recognized: {text}")
        return text
