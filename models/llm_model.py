import httpx
import asyncio
import os
from typing import Optional
import logging
import time
import json

logger = logging.getLogger(__name__)

class LLMModel:
    def __init__(self, 
                 ollama_url: str = "http://localhost:11434",
                 model_name: str = "llama3.2:1b",
                 echo_mode: bool = False,
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
        self.timeout = timeout
        
        logger.info(f"🧠 Initializing LLM Model: {model_name}")
        logger.info(f"🧠 Ollama URL: {ollama_url}")
        logger.info(f"🧠 Echo mode: {echo_mode}")
        logger.info(f"🧠 Timeout: {timeout}s")
        
        self.client = httpx.AsyncClient(timeout=timeout)
        
        if self.echo_mode:
            logger.info("🧠 LLM initialized in echo mode - will return input text")
        else:
            logger.info(f"🧠 LLM initialized with Ollama at {ollama_url}, model: {model_name}")

        async def generate_response_with_context(
        self, 
        prompt: str, 
        db_context: str = "",
        max_tokens: int = 150
    ) -> str:
        """
        Generate response with database context.
        
        Args:
            prompt: User's question
            db_context: Formatted database information
            max_tokens: Maximum tokens to generate
        """
        logger.info(f"🧠 Starting LLM generation with database context")
        
        if self.echo_mode:
            return prompt
        
        try:
            # Build enhanced prompt with database context
            context_section = ""
            if db_context:
                context_section = (
                    f"\n\n=== अस्पताल की जानकारी (Hospital Information) ===\n"
                    f"{db_context}\n"
                    f"=== जानकारी समाप्त ===\n\n"
                )
            
            persona_prompt = (
                "You are a helpful and polite receptionist at a hospital. "
                "Answer patient queries in Hindi about hospital services, appointments, departments, visiting hours, and general information. "
                "Always respond in Hindi language only. "
                "Keep responses brief, clear, and professional. "
                "Use the hospital information provided below to answer accurately."
                f"{context_section}"
                f"रोगी (Patient): {prompt}\n"
                "रिसेप्शनिस्ट (Receptionist):"
            )
            
            # Rest of the generation logic (same as generate_response)
            request_data = {
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
            
            response = await self.client.post(
                f"{self.ollama_url}/api/generate",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "").strip()
                logger.info(f"✅ LLM generation with context completed")
                return response_text
            else:
                logger.error(f"🧠 Ollama API error: {response.status_code}")
                return f"[LLM Error] {prompt}"
                
        except Exception as e:
            logger.error(f"🧠 LLM generation with context failed: {e}")
            return f"[LLM Error] {prompt}"
        
    async def generate_response(self, prompt: str, max_tokens: int = 150) -> str:
        logger.info(f"🧠 Starting LLM generation")
        logger.info(f"🧠 Input prompt length: {len(prompt)} characters")
        logger.info(f"🧠 Max tokens: {max_tokens}")
        logger.info(f"🧠 Input preview: '{prompt[:100]}{'...' if len(prompt) > 100 else ''}'")
        
        if self.echo_mode:
            logger.info("🧠 Echo mode: returning input text")
            return prompt
        
        try:
            # Create persona prompt
            persona_prompt = (
                "You are a helpful and polite receptionist at a hospital. "
                "Answer patient queries in Hindi about hospital services, appointments, departments, visiting hours, and general information. "
                "Always respond in Hindi language only. "
                "Keep responses brief, clear, and professional. "
                f"\n\nरोगी (Patient): {prompt}\nरिसेप्शनिस्ट (Receptionist):"
            )
            
            logger.info(f"🧠 Full prompt length: {len(persona_prompt)} characters")
            logger.info(f"🧠 Persona prompt preview: '{persona_prompt[:150]}{'...' if len(persona_prompt) > 150 else ''}'")

            # Prepare request
            request_data = {
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
            
            logger.info(f"🧠 Request parameters: temperature=0.7, top_p=0.9, max_tokens={max_tokens}")
            logger.info(f"🧠 Sending request to: {self.ollama_url}/api/generate")
            
            # Make request
            request_start = time.time()
            response = await self.client.post(
                f"{self.ollama_url}/api/generate",
                json=request_data
            )
            request_time = time.time() - request_start
            
            logger.info(f"🧠 Request completed in {request_time:.3f}s")
            logger.info(f"🧠 Response status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "").strip()
                
                # Log response details
                logger.info(f"🧠 Response length: {len(response_text)} characters")
                logger.info(f"🧠 Response preview: '{response_text[:100]}{'...' if len(response_text) > 100 else ''}'")
                
                # Log performance metrics if available
                if "eval_duration" in result:
                    eval_time = result["eval_duration"] / 1e9  # Convert nanoseconds to seconds
                    logger.info(f"🧠 Model evaluation time: {eval_time:.3f}s")
                
                if "total_duration" in result:
                    total_time = result["total_duration"] / 1e9
                    logger.info(f"🧠 Total generation time: {total_time:.3f}s")
                
                logger.info(f"✅ LLM generation completed successfully")
                return response_text
            else:
                logger.error(f"🧠 Ollama API error: {response.status_code}")
                logger.error(f"🧠 Response body: {response.text}")
                return f"[LLM Error] {prompt}"

        except httpx.ConnectError as e:
            logger.error(f"🧠 Could not connect to Ollama service: {e}")
            logger.error(f"🧠 Check if Ollama is running at {self.ollama_url}")
            return "[LLM Offline] Please try again later."

        except httpx.TimeoutException as e:
            logger.error(f"🧠 Request timeout after {self.timeout}s: {e}")
            return "[LLM Timeout] Request took too long."

        except Exception as e:
            logger.error(f"🧠 LLM generation failed: {e}")
            logger.error(f"🧠 Error type: {type(e).__name__}")
            return f"[LLM Error] {prompt}"


    
    async def preload_model(self) -> bool:
        """Preload the model in Ollama to ensure it's ready."""
        logger.info(f"🧠 Starting LLM model preload: {self.model_name}")
        
        if self.echo_mode:
            logger.info("🧠 Echo mode: skipping model preload")
            return True
            
        try:
            # First check if model is already available
            logger.info(f"🧠 Checking existing models at {self.ollama_url}/api/tags")
            try:
                check_start = time.time()
                response = await self.client.get(f"{self.ollama_url}/api/tags")
                check_time = time.time() - check_start
                logger.info(f"🧠 Model check completed in {check_time:.3f}s")
                
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [model.get("name", "") for model in models]
                    logger.info(f"🧠 Available models: {model_names}")
                    
                    if any(self.model_name in name for name in model_names):
                        logger.info(f"✅ LLM model {self.model_name} already available")
                        return True
                    else:
                        logger.info(f"🧠 Model {self.model_name} not found, will pull it")
                else:
                    logger.warning(f"🧠 Failed to check models: {response.status_code}")
            except Exception as e:
                logger.warning(f"🧠 Could not check existing models: {e}")
            
            # Pull the model if not already available
            logger.info(f"🧠 Pulling LLM model: {self.model_name}")
            logger.info(f"🧠 This may take several minutes for large models...")
            
            pull_start = time.time()
            pull_response = await self.client.post(
                f"{self.ollama_url}/api/pull",
                json={"name": self.model_name}
            )
            pull_time = time.time() - pull_start
            
            logger.info(f"🧠 Pull request completed in {pull_time:.3f}s")
            logger.info(f"🧠 Pull response status: {pull_response.status_code}")
            
            if pull_response.status_code == 200:
                logger.info(f"✅ LLM model {self.model_name} preloaded successfully")
                return True
            else:
                logger.warning(f"🧠 Failed to preload model: {pull_response.status_code}")
                logger.warning(f"🧠 Response: {pull_response.text}")
                return False
                
        except Exception as e:
            logger.error(f"🧠 Failed to preload LLM model: {e}")
            logger.error(f"🧠 Error type: {type(e).__name__}")
            return False

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
