from transformers import AutoTokenizer, AutoModel
import torch
import soundfile as sf
import asyncio
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class TTSModel:
    def __init__(self, 
                 model_name: str = "ai4bharat/indic-parler-tts",
                 hf_token: Optional[str] = None,
                 mock_mode: bool = False):
        """
        Initialize TTS model with HF authentication and optimization.
        
        Args:
            model_name: HuggingFace model identifier
            hf_token: HuggingFace access token
            mock_mode: If True, generate silent audio for testing
        """
        self.mock_mode = mock_mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 16000
        
        if self.mock_mode:
            logger.info("TTS initialized in mock mode")
            return
            
        try:
            logger.info(f"Loading TTS model {model_name} on {self.device}")
            
            # Load tokenizer with HF token
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token,
                trust_remote_code=True
            )
            
            # Load model with optimizations
            self.model = AutoModel.from_pretrained(
                model_name,
                token=hf_token,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            self.model.eval()  # Set to evaluation mode
            logger.info("TTS model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            logger.info("Falling back to mock mode")
            self.mock_mode = True

    async def synthesize(self, text: str, output_path: str = "output.wav") -> str:
        """
        Synthesize text to speech and save as WAV file.
        
        Args:
            text: Text to synthesize
            output_path: Path to save the generated audio
            
        Returns:
            Path to the generated audio file
        """
        if self.mock_mode:
            await asyncio.sleep(0.1)  # Simulate processing time
            # Generate 1 second of silence as mock audio
            silence = np.zeros(self.sample_rate, dtype=np.float32)
            sf.write(output_path, silence, self.sample_rate)
            return output_path
            
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate speech in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            
            def _generate_speech():
                with torch.no_grad():
                    # Note: Actual API may vary for indic-parler-tts
                    # This is a generic implementation that may need adjustment
                    speech_output = self.model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=0.7,
                        max_length=1024
                    )
                    return speech_output
            
            speech = await loop.run_in_executor(None, _generate_speech)
            
            # Convert to numpy and save
            if hasattr(speech, 'audio'):
                audio_data = speech.audio.cpu().numpy().squeeze()
            else:
                # Fallback if the output format is different
                audio_data = speech.cpu().numpy().squeeze()
                
            # Ensure audio is in the right format
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=0)  # Convert to mono
                
            # Normalize audio
            if audio_data.max() > 1.0 or audio_data.min() < -1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Save as WAV file
            sf.write(output_path, audio_data, self.sample_rate)
            return output_path
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            # Generate silence as fallback
            silence = np.zeros(self.sample_rate, dtype=np.float32)
            sf.write(output_path, silence, self.sample_rate)
            return output_path
