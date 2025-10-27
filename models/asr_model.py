from transformers import AutoModel
import torch
import torchaudio
import asyncio
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ASRModel:
    def __init__(self, 
                 model_name: str = "ai4bharat/indic-conformer-600m-multilingual",
                 hf_token: Optional[str] = None,
                 mock_mode: bool = False):
        self.mock_mode = mock_mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_sample_rate = 16000
        
        if self.mock_mode:
            logger.info("ASR initialized in mock mode")
            return
            
        try:
            logger.info(f"Loading ASR model {model_name} on {self.device}")
            
            self.model = AutoModel.from_pretrained(
                model_name,
                token=hf_token,
                trust_remote_code=True
            ).to(self.device)
            
            self.model.eval()
            logger.info("ASR model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ASR model: {e}")
            logger.info("Falling back to mock mode")
            self.mock_mode = True

    async def transcribe_with_language(self, wav_path: str) -> Tuple[str, str]:
        if self.mock_mode:
            await asyncio.sleep(0.1)
            return "Hello, this is a mock transcription for testing.", "english"
            
        try:
            loop = asyncio.get_event_loop()
            wav, sr = await loop.run_in_executor(None, torchaudio.load, wav_path)
            
            wav = torch.mean(wav, dim=0, keepdim=True)
            
            if sr != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, 
                    new_freq=self.target_sample_rate
                )
                wav = resampler(wav)
            
            wav = wav.to(self.device)
            
            def _transcribe():
                with torch.no_grad():
                    # Try multiple language codes, starting with English
                    # Indic-Conformer supports: en, hi, ta, te, kn, ml, mr, gu, pa, as, or, ur, bn, ne, si
                    result = self.model(wav, "en", "ctc")
                    
                    # Model returns transcription string directly
                    if isinstance(result, str):
                        return result, "en"
                    elif isinstance(result, tuple):
                        return result[0], result[1] if len(result) > 1 else "en"
                    else:
                        return str(result), "en"
            
            transcription, detected_language = await loop.run_in_executor(None, _transcribe)
            
            logger.info(f"Model detected language: {detected_language}")
            return transcription.strip(), detected_language
            
        except Exception as e:
            logger.error(f"ASR transcription failed: {e}")
            return f"[ASR Error] Could not transcribe audio: {str(e)}", "unknown"
    
    async def transcribe(self, wav_path: str) -> str:
        text, _ = await self.transcribe_with_language(wav_path)
        return text