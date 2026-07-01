#!/usr/bin/env python3
"""
Main orchestrator - Coordinates all components for voice assistant
"""

import numpy as np
from listener import Listener
from retriever import Retriever
from composer import Composer
from brain import Brain
from speaker import Speaker


class VoiceAssistant:
    def __init__(self):
        print("Initializing Voice Assistant...")
        
        # Initialize all components
        self.listener = Listener()
        self.retriever = Retriever()
        self.composer = Composer()
        self.brain = Brain()
        self.speaker = Speaker()
        
        print("Voice Assistant initialized!")
    
    def listen_and_respond(self):
        """Main conversation loop"""
        print("\n=== Voice Assistant Ready ===")
        print("Speak now (will record for 10 seconds)...")
        
        try:
            # Record audio
            audio_array = self.listener.record_audio()
            
            # Check if there is valid speech input (prevent hallucination on silence)
            max_level = np.max(np.abs(audio_array))
            if max_level < 0.03:
                print(f"No speech detected (Audio level too low: {max_level:.4f}).")
                self.speaker.speak("音声が検出されませんでした。")
                return
            
            # Transcribe
            text = self.listener.transcribe(audio_array)
            
            if not text.strip():
                print("No speech detected.")
                return
            
            # Search web
            search_results = self.retriever.search_web(text)
            
            # Compose prompt with search context
            prompt = self.composer.compose_prompt(text, search_results)
            
            # Generate AI response
            response = self.brain.generate_response(prompt)
            
            # Speak response
            self.speaker.speak(response)
            
        except KeyboardInterrupt:
            print("\nExiting...")
        except Exception as e:
            print(f"Error: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        pass


def main():
    assistant = VoiceAssistant()
    try:
        assistant.listen_and_respond()
    finally:
        assistant.cleanup()


if __name__ == "__main__":
    main()
