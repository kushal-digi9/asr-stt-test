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
            )
            # Try GPU first, fall back to CPU if needed
            try:
                if torch.cuda.is_available():
                    self.model = self.model.to(self.device)
                    logger.info(f"ASR model on GPU: {self.device}")
                else:
                    self.device = "cpu"
                    logger.info("ASR model on CPU")
            except Exception as e:
                logger.warning(f"Could not move model to GPU: {e}, using CPU")
                self.device = "cpu"
            
            self.model.eval()
            logger.info("ASR model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ASR model: {e}")
            logger.info("Falling back to mock mode")
            self.mock_mode = True

    async def transcribe(self, wav_path: str) -> str:
        """Transcribe Hindi audio to text."""
        if self.mock_mode:
            await asyncio.sleep(0.1)
            return "नमस्ते, यह एक परीक्षण है।"
            
        try:
            loop = asyncio.get_event_loop()
            wav, sr = await loop.run_in_executor(None, torchaudio.load, wav_path)
            
            # Convert to mono
            wav = torch.mean(wav, dim=0, keepdim=True)
            
            # Resample if needed
            if sr != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, 
                    new_freq=self.target_sample_rate
                )
                wav = resampler(wav)
            
            # Move to device
            wav = wav.to(self.device)
            
            def _transcribe():
                with torch.no_grad():
                    logger.info(f"Audio tensor shape: {wav.shape}, dtype: {wav.dtype}, device: {wav.device}")
                    logger.info("Running Hindi transcription with CTC decoding")
                    # Hindi-only transcription
                    transcription = self.model(wav, "hi", "ctc")
                    return str(transcription).strip()
            
            transcription = await loop.run_in_executor(None, _transcribe)
            return transcription
            
        except Exception as e:
            logger.error(f"ASR transcription failed: {e}")
            return f"[ASR Error] Could not transcribe audio: {str(e)}"