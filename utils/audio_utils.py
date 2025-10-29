import soundfile as sf
import io
import librosa
import numpy as np
from fastapi import UploadFile, HTTPException
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

def validate_wav_file(file_content: bytes) -> bool:
    try:
        
        if len(file_content) < 44: 
            return False
            
       
        if file_content[:4] != b'RIFF':
            return False
            
        # Check WAV format
        if file_content[8:12] != b'WAVE':
            return False
            
        return True
    except Exception:
        return False

def save_uploaded_wav(file: UploadFile, path: str) -> str:
    try:
        
        file_content = file.file.read()
        
       
        max_size = 50 * 1024 * 1024  
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size is {max_size // (1024*1024)}MB"
            )
        
        # Validate WAV format
        if not validate_wav_file(file_content):
            raise HTTPException(
                status_code=400, 
                detail="Invalid WAV file format"
            )
        
        # Save to disk
        with open(path, "wb") as f:
            f.write(file_content)
            
        logger.info(f"Saved audio file: {path} ({len(file_content)} bytes)")
        return path
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save audio file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save audio file")

def load_audio_bytes(wav_path: str) -> bytes:
    try:
        with open(wav_path, "rb") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to load audio file {wav_path}: {e}")
        raise

def preprocess_audio(wav_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    try:
        # Load audio with librosa for better format support
        audio, sr = librosa.load(wav_path, sr=target_sr, mono=True)
        
        # Normalize audio
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / np.max(np.abs(audio))
        
        logger.info(f"Preprocessed audio: {len(audio)} samples at {sr}Hz")
        return audio, sr
        
    except Exception as e:
        logger.error(f"Failed to preprocess audio {wav_path}: {e}")
        raise

def convert_to_wav_bytes(audio_array: np.ndarray, sample_rate: int = 16000) -> bytes:
    try:
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, sample_rate, format="WAV")
        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        logger.error(f"Failed to convert audio to WAV bytes: {e}")
        raise

def trim_silence(audio_path: str, output_path: str = None, threshold_db: float = -40.0) -> str:
    try:
        if output_path is None:
            output_path = audio_path
            
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Trim silence from beginning and end
        trimmed_audio, _ = librosa.effects.trim(
            audio, 
            top_db=-threshold_db,
            frame_length=2048,
            hop_length=512
        )
        
        # Add small fade in/out to prevent clicks
        fade_samples = int(0.01 * sr)  # 10ms fade
        if len(trimmed_audio) > fade_samples * 2:
            # Fade in
            trimmed_audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
            # Fade out
            trimmed_audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        # Save trimmed audio
        sf.write(output_path, trimmed_audio, sr)
        
        original_duration = len(audio) / sr
        trimmed_duration = len(trimmed_audio) / sr
        
        logger.info(f"Trimmed audio: {original_duration:.2f}s → {trimmed_duration:.2f}s (saved {original_duration - trimmed_duration:.2f}s)")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to trim silence from {audio_path}: {e}")
        return audio_path

def get_audio_duration(wav_path: str) -> float:
    try:
        audio, sr = librosa.load(wav_path, sr=None)
        return len(audio) / sr
    except Exception as e:
        logger.error(f"Failed to get audio duration for {wav_path}: {e}")
        return 0.0
