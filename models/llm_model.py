import httpx
import asyncio
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class LLMModel:
    def __init__(self, 
                 ollama_url: str = "http://localhost:11434",
                 model_name: str = "llama3.2:1b",
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

    async def generate_response(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Generate response using Ollama API or echo mode.
        
        Args:
            prompt: Input text to process
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        if self.echo_mode:
            await asyncio.sleep(0.1)
            
            company_context = "I'm an AI voice agent from Digi9 Reach Info Systems, a product-based startup specializing in voice AI agents. "
            
            if "hello" in prompt.lower() or "hi" in prompt.lower():
                return f"{company_context}Hello! Welcome to Digi9 Reach. How can I assist you with our voice AI solutions today?"
            elif "नमस्ते" in prompt or "namaste" in prompt.lower():
                return f"नमस्ते! मैं Digi9 Reach Info Systems का AI voice agent हूं। हम voice AI agents में specialize करते हैं। आपकी कैसे सहायता कर सकता हूं?"
            elif "good morning" in prompt.lower():
                return f"{company_context}Good morning! I hope you're having a wonderful day. How can Digi9 Reach help you today?"
            elif "how are you" in prompt.lower() or "what do you do" in prompt.lower():
                return f"{company_context}I'm doing great! I represent Digi9 Reach Info Systems, where we build cutting-edge voice AI agents for businesses. What can I help you with?"
            elif "company" in prompt.lower() or "digi9" in prompt.lower():
                return "Digi9 Reach Info Systems is an innovative product-based startup focused on developing advanced voice AI agents. We create intelligent conversational solutions for businesses."
            elif "voice ai" in prompt.lower() or "ai agent" in prompt.lower():
                return "At Digi9 Reach, we specialize in voice AI agents that can understand multiple languages and provide natural conversations. Our technology helps businesses automate customer interactions."
            else:
                return f"{company_context}Thank you for reaching out. At Digi9 Reach, we're always ready to help with voice AI solutions. What would you like to know?"
            
        try:
            # Call Ollama API
            response = await self.client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": max_tokens,
                        "top_p": 0.9,
                        "stop": ["\n\n", "Human:", "Assistant:"]
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
            return f"[LLM Offline] {prompt}"
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
