from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import torchaudio
import asyncio
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class ASRModel:
    def __init__(self, 
                 model_name: str = "ai4bharat/IndicConformer",
                 hf_token: Optional[str] = None,
                 mock_mode: bool = False):
        """
        Initialize ASR model with HF authentication and optimization.
        
        Args:
            model_name: HuggingFace model identifier
            hf_token: HuggingFace access token
            mock_mode: If True, return mock transcription for testing
        """
        self.mock_mode = mock_mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.mock_mode:
            logger.info("ASR initialized in mock mode")
            return
            
        try:
            logger.info(f"Loading ASR model {model_name} on {self.device}")
            
            # Load processor with HF token
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                token=hf_token,
                trust_remote_code=True
            )
            
            # Load model with optimizations
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                token=hf_token,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            self.model.eval()  # Set to evaluation mode
            logger.info("ASR model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ASR model: {e}")
            logger.info("Falling back to mock mode")
            self.mock_mode = True

    async def transcribe(self, wav_path: str) -> str:
        """
        Transcribe audio file to text.
        
        Args:
            wav_path: Path to WAV audio file
            
        Returns:
            Transcribed text
        """
        if self.mock_mode:
            await asyncio.sleep(0.1)  # Simulate processing time
            return "Hello, this is a mock transcription for testing."
            
        try:
            # Load audio in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            speech_array, sampling_rate = await loop.run_in_executor(
                None, torchaudio.load, wav_path
            )
            
            # Preprocess audio
            speech_array = speech_array.squeeze()
            if speech_array.dim() > 1:
                speech_array = speech_array.mean(dim=0)  # Convert to mono
                
            # Process inputs
            inputs = self.processor(
                speech_array, 
                sampling_rate=sampling_rate, 
                return_tensors="pt"
            ).to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=1,  # Faster inference
                    do_sample=False
                )
                
            # Decode result
            text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"ASR transcription failed: {e}")
            return f"[ASR Error] Could not transcribe audio: {str(e)}"
