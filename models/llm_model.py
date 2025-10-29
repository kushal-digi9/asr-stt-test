import httpx
import asyncio
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class LLMModel:
    def __init__(self, 
                 ollama_url: str = "http://localhost:11434",
                 model_name: str = "qwen2.5:0.5b-instruct",
                 echo_mode: bool = True,
                 timeout: float = 30.0):
        """
        Initialize LLM model using Ollama API or echo mode.
        
        Args:
            ollama_url: URL of the Ollama service
            model_name: Name of the model to use in Ollama
            echo_mode: If True, just return the input text (for testing)
            timeout: Request timeout in seconds
        """
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.echo_mode = echo_mode
        self.client = httpx.AsyncClient(timeout=timeout)
        
        if self.echo_mode:
            logger.info("LLM initialized in echo mode")
        else:
            logger.info(f"LLM initialized with Ollama at {ollama_url}, model: {model_name}")
        
    async def generate_response(self, prompt: str, max_tokens: int = 150) -> str:
        
        if self.echo_mode:
            return prompt
        
        try:
            persona_prompt = (
                "You are a helpful and polite receptionist at a hospital. "
                "Answer patient queries in Hindi about hospital services, appointments, departments, visiting hours, and general information. "
                "Always respond in Hindi language only. "
                "Keep responses brief, clear, and professional. "
                f"\n\nरोगी (Patient): {prompt}\nरिसेप्शनिस्ट (Receptionist):"
            )

            response = await self.client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": persona_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": max_tokens,
                        "top_p": 0.9,
                        "stop": ["\n\n", "User:", "Assistant:"]
                    }
                }
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return f"[LLM Error] {prompt}"

        except httpx.ConnectError:
            logger.error("Could not connect to Ollama service")
            return "[LLM Offline] Please try again later."

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"[LLM Error] {prompt}"


    
    async def health_check(self) -> bool:
        """Check if Ollama service is available."""
        if self.echo_mode:
            return True
            
        try:
            response = await self.client.get(f"{self.ollama_url}/api/tags")
            return response.status_code == 200
        except:
            return False
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
