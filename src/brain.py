#!/usr/bin/env python3
"""
Brain module - AI response generation using Ollama
"""

import requests
import os

# Configuration
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


class Brain:
    def __init__(self):
        print("Initializing Brain (Ollama)...")
        self.ollama_api_url = OLLAMA_API_URL
        self.ollama_model = OLLAMA_MODEL
        print("Brain initialized!")
    
    def generate_response(self, prompt):
        """Generate response using Ollama/Qwen 2.5 with search context"""
        print("Generating response with AI...")
        
        try:
            response = requests.post(
                self.ollama_api_url,
                json={
                    'model': self.ollama_model,
                    'prompt': prompt,
                    'stream': False
                },
                timeout=120  # Increased timeout for AI generation
            )
            response.raise_for_status()
            data = response.json()
            
            answer = data.get('response', '').strip()
            print(f"AI Response: {answer}")
            return answer
            
        except Exception as e:
            print(f"AI generation error: {e}")
            return "申し訳ありませんが、回答を生成できませんでした。"
