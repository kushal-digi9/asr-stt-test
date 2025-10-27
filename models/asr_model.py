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
                    # Debug logging
                    logger.info(f"Audio tensor shape: {wav.shape}, dtype: {wav.dtype}, device: {wav.device}")
                    
                    # Try multiple language codes with better validation
                    # Start with English first as it's more common
                    language_codes = ["en", "hi", "bn", "te", "ta", "mr", "gu", "kn", "ml", "or", "pa"]
                    
                    transcription_results = []
                    
                    for lang_code in language_codes:
                        try:
                            logger.info(f"Attempting transcription with language: {lang_code}")
                            transcription = self.model(wav, lang_code, "ctc")
                            
                            # Model returns string, not tuple - handle accordingly
                            transcription_text = str(transcription).strip()
                            
                            # Basic validation - check if we get reasonable output
                            if transcription_text and len(transcription_text) > 0:
                                # Store result for comparison
                                transcription_results.append((transcription_text, lang_code))
                                logger.info(f"Got transcription with {lang_code}: '{transcription_text[:50]}...'")
                                
                                # For English, check if output contains mostly Latin characters
                                if lang_code == "en":
                                    latin_chars = sum(1 for c in transcription_text if c.isascii() and c.isalpha())
                                    total_chars = sum(1 for c in transcription_text if c.isalpha())
                                    if total_chars > 0 and (latin_chars / total_chars) > 0.7:
                                        logger.info(f"English transcription validated (Latin ratio: {latin_chars/total_chars:.2f})")
                                        return transcription_text, lang_code
                                
                                # For Indian languages, check if output contains appropriate script
                                elif lang_code in ["hi", "mr", "ne"]:  # Devanagari script
                                    devanagari_chars = sum(1 for c in transcription_text if '\u0900' <= c <= '\u097F')
                                    total_chars = sum(1 for c in transcription_text if c.isalpha())
                                    if total_chars > 0 and (devanagari_chars / total_chars) > 0.5:
                                        logger.info(f"Devanagari script validated for {lang_code}")
                                        return transcription_text, lang_code
                                
                                # For other languages, accept if we get reasonable text
                                elif len(transcription_text.split()) > 2:  # At least 3 words
                                    logger.info(f"Reasonable transcription length for {lang_code}")
                                    return transcription_text, lang_code
                                
                        except ValueError as ve:
                            logger.warning(f"Language validation error for {lang_code}: {ve}")
                            continue
                        except RuntimeError as re:
                            logger.warning(f"Model runtime error for {lang_code}: {re}")
                            continue
                        except Exception as e:
                            logger.warning(f"Unexpected error for {lang_code}: {e}")
                            continue
                    
                    # If no language passed validation, return the first successful result (likely English)
                    if transcription_results:
                        best_result = transcription_results[0]  # First result (English)
                        logger.info(f"Using first successful transcription: {best_result[1]}")
                        return best_result[0], best_result[1]
                    
                    # If all languages failed, try with default "en" one more time
                    logger.warning("All language attempts failed, trying default 'en'")
                    transcription = self.model(wav, "en", "ctc")
                    return str(transcription), "en"
            
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