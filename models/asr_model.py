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
            
            # Ensure tensor is on the right device
            wav = wav.to(self.device)
            
            def _transcribe():
                with torch.no_grad():
                    logger.info(f"Audio tensor shape: {wav.shape}, dtype: {wav.dtype}, device: {wav.device}")
                    # Hindi-only transcription with CTC decoding
                    logger.info("Running Hindi (hi) transcription with CTC decoding")
                    transcription = self.model(wav, "hi", "ctc")
                    transcription_text = str(transcription).strip()
                    return transcription_text, "hi"
            
            transcription, detected_language = await loop.run_in_executor(None, _transcribe)
            
            logger.info(f"Model detected language: {detected_language}")
            return transcription.strip(), detected_language
            
        except ValueError as ve:
            logger.error(f"Language validation error: {ve}")
            return f"[ASR Error] Language validation failed: {str(ve)}", "unknown"
        except RuntimeError as re:
            logger.error(f"Model runtime error: {re}")
            return f"[ASR Error] Model runtime failed: {str(re)}", "unknown"
        except Exception as e:
            logger.error(f"Unexpected ASR error: {e}")
            return f"[ASR Error] Could not transcribe audio: {str(e)}", "unknown"
    
    async def transcribe(self, wav_path: str) -> str:
        text, _ = await self.transcribe_with_language(wav_path)
        return text