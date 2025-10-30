from transformers import AutoModel
import torch
import torchaudio
import asyncio
from typing import Optional, Tuple
import logging
import time
import os

logger = logging.getLogger(__name__)

class ASRModel:
    def __init__(self, 
                 model_name: str = "ai4bharat/indic-conformer-600m-multilingual",
                 hf_token: Optional[str] = None,
                 mock_mode: bool = False):
        self.mock_mode = mock_mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_sample_rate = 16000
        self.model_name = model_name
        
        logger.info(f"🎤 Initializing ASR Model: {model_name}")
        logger.info(f"🎤 Device: {self.device}")
        logger.info(f"🎤 Target sample rate: {self.target_sample_rate} Hz")
        logger.info(f"🎤 Mock mode: {self.mock_mode}")
        
        if self.mock_mode:
            logger.info("🎤 ASR initialized in mock mode - will return dummy transcriptions")
            return
            
        try:
            logger.info(f"🎤 Loading ASR model from HuggingFace: {model_name}")
            logger.info(f"🎤 Using token: {'Yes' if hf_token else 'No'}")
            
            load_start = time.time()
            self.model = AutoModel.from_pretrained(
                model_name,
                token=hf_token,
                trust_remote_code=True
            )
            load_time = time.time() - load_start
            logger.info(f"🎤 Model loaded from HuggingFace in {load_time:.2f}s")
            
            # Try GPU first, fall back to CPU if needed
            try:
                if torch.cuda.is_available():
                    logger.info(f"🎤 Moving model to GPU: {self.device}")
                    move_start = time.time()
                    self.model = self.model.to(self.device)
                    move_time = time.time() - move_start
                    logger.info(f"🎤 Model moved to GPU in {move_time:.2f}s")
                    logger.info(f"🎤 GPU memory after model load: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                else:
                    self.device = "cpu"
                    logger.info("🎤 Using CPU for ASR inference")
            except Exception as e:
                logger.warning(f"🎤 Could not move model to GPU: {e}, using CPU")
                self.device = "cpu"
            
            logger.info("🎤 Setting model to evaluation mode")
            self.model.eval()
            
            # Enable FP16 for faster inference
            if torch.cuda.is_available():
                logger.info("🎤 Converting model to FP16 for faster inference")
                fp16_start = time.time()
                self.model = self.model.half()
                fp16_time = time.time() - fp16_start
                logger.info(f"🎤 Model converted to FP16 in {fp16_time:.2f}s")
                logger.info(f"🎤 GPU memory after FP16 conversion: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            
            logger.info("✅ ASR model loaded and ready for inference")
            
        except Exception as e:
            logger.error(f"Failed to load ASR model: {e}")
            logger.info("Falling back to mock mode")
            self.mock_mode = True

    async def transcribe(self, wav_path: str) -> str:
        """Transcribe Hindi audio to text."""
        logger.info(f"🎤 Starting ASR transcription for: {os.path.basename(wav_path)}")
        
        if self.mock_mode:
            logger.info("🎤 Mock mode: returning dummy transcription")
            await asyncio.sleep(0.1)
            return "नमस्ते, यह एक परीक्षण है।"
            
        try:
            # Load audio file
            logger.info(f"🎤 Loading audio file: {wav_path}")
            file_size = os.path.getsize(wav_path) / 1024  # KB
            logger.info(f"🎤 Audio file size: {file_size:.1f} KB")
            
            loop = asyncio.get_event_loop()
            load_start = time.time()
            wav, sr = await loop.run_in_executor(None, torchaudio.load, wav_path)
            load_time = time.time() - load_start
            logger.info(f"🎤 Audio loaded in {load_time:.3f}s - Shape: {wav.shape}, Sample rate: {sr} Hz")
            
            # Convert to mono
            logger.info("🎤 Converting to mono channel")
            wav = torch.mean(wav, dim=0, keepdim=True)
            logger.info(f"🎤 Mono audio shape: {wav.shape}")
            
            # Resample if needed
            if sr != self.target_sample_rate:
                logger.info(f"🎤 Resampling from {sr} Hz to {self.target_sample_rate} Hz")
                resample_start = time.time()
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, 
                    new_freq=self.target_sample_rate
                )
                wav = resampler(wav)
                resample_time = time.time() - resample_start
                logger.info(f"🎤 Resampling completed in {resample_time:.3f}s")
            
            # Move to device
            logger.info(f"🎤 Moving audio to device: {self.device}")
            wav = wav.to(self.device)
            logger.info(f"🎤 Audio on device: {wav.device}, dtype: {wav.dtype}")
            
            def _transcribe():
                with torch.no_grad():
                    logger.info(f"🎤 Starting transcription inference")
                    logger.info(f"🎤 Audio tensor: shape={wav.shape}, dtype={wav.dtype}, device={wav.device}")
                    logger.info(f"🎤 Model dtype: {self.model.dtype if hasattr(self.model, 'dtype') else 'unknown'}")
                    
                    # Convert audio to FP16 for inference if model is FP16
                    if hasattr(self.model, 'dtype') and self.model.dtype == torch.float16:
                        logger.info("🎤 Converting audio to FP16 for model compatibility")
                        wav_fp16 = wav.half()
                        logger.info(f"🎤 FP16 audio: shape={wav_fp16.shape}, dtype={wav_fp16.dtype}")
                        inference_start = time.time()
                        transcription = self.model(wav_fp16, "hi", "ctc")
                        inference_time = time.time() - inference_start
                    else:
                        logger.info("🎤 Using FP32 for inference")
                        inference_start = time.time()
                        transcription = self.model(wav, "hi", "ctc")
                        inference_time = time.time() - inference_start
                    
                    logger.info(f"🎤 Inference completed in {inference_time:.3f}s")
                    logger.info(f"🎤 Raw transcription: '{str(transcription).strip()}'")
                    return str(transcription).strip()
            
            transcription = await loop.run_in_executor(None, _transcribe)
            logger.info(f"✅ ASR transcription completed: '{transcription[:100]}{'...' if len(transcription) > 100 else ''}'")
            return transcription
            
        except Exception as e:
            logger.error(f"ASR transcription failed: {e}")
            return f"[ASR Error] Could not transcribe audio: {str(e)}"