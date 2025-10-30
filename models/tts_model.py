import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import asyncio
import numpy as np
from typing import Optional
import logging
import time
import os
from utils.audio_utils import trim_silence

logger = logging.getLogger(__name__)

class TTSModel:
    def __init__(
        self, 
        model_name: str = "parler-tts/parler-tts-mini-v1.1",
        hf_token: Optional[str] = None,
        mock_mode: bool = False
    ):
        self.mock_mode = mock_mode
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        logger.info(f"Initializing TTS Model: {model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mock mode: {self.mock_mode}")
        
        if self.mock_mode:
            logger.info("TTS initialized in mock mode - will generate silence")
            return
            
        try:
            logger.info(f"Loading TTS model from HuggingFace: {model_name}")
            logger.info(f"Using token: {'Yes' if hf_token else 'No'}")
            
            load_start = time.time()
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                model_name,
                token=hf_token
            ).to(self.device)
            load_time = time.time() - load_start
            logger.info(f"Model loaded in {load_time:.2f}s")
            
            try:
                if torch.cuda.is_available():
                    logger.info("Converting model to FP16 for faster inference")
                    fp16_start = time.time()
                    self.model = self.model.half().to(self.device)
                    fp16_time = time.time() - fp16_start
                    logger.info(f"Model converted to FP16 in {fp16_time:.2f}s")
                    logger.info(f"GPU memory after FP16 conversion: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                else:
                    self.model = self.model.to(torch.float32)
                    logger.info("Using FP32 for CPU mode")
            except Exception as e:
                logger.warning(f"Could not convert to FP16: {e}")
                pass
            
            logger.info("Loading tokenizers...")
            tokenizer_start = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token
            )
            
            self.description_tokenizer = AutoTokenizer.from_pretrained(
                self.model.config.text_encoder._name_or_path
            )
            tokenizer_time = time.time() - tokenizer_start
            logger.info(f"Tokenizers loaded in {tokenizer_time:.2f}s")
             
            logger.info("Setting model to evaluation mode")
            self.model.eval()
            logger.info("TTS model loaded and ready for synthesis")
            
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            logger.info("Falling back to mock mode")
            self.mock_mode = True

    async def synthesize(
        self, 
        text: str, 
        output_path: str = "output.wav",
        description: Optional[str] = None
    ) -> str:
        logger.info(f"Starting TTS synthesis for: {os.path.basename(output_path)}")
        logger.info(f"Text length: {len(text)} characters")
        logger.info(f"Text preview: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        if self.mock_mode:
            logger.info("Mock mode: generating silence")
            await asyncio.sleep(0.1)
            silence = np.zeros(24000, dtype=np.float32)
            sf.write(output_path, silence, 24000)
            logger.info(f"Mock audio saved to: {output_path}")
            return output_path
            
        try:
            if description is None:
                description = (
                    "A female speaker with an Indian accent delivers speech at a moderate pace. "
                    "Her voice is clear and warm, with excellent recording quality and minimal background noise. "
                    "She speaks Hindi naturally with proper pronunciation and Indian intonation."
                )
            
            logger.info(f"Voice description: '{description[:100]}{'...' if len(description) > 100 else ''}'")
            logger.info(f"Output path: {output_path}")
            
            loop = asyncio.get_event_loop()
            
            def _generate_speech():
                with torch.inference_mode():
                    logger.info("🔊 Tokenizing description and text...")
                    tokenize_start = time.time()
                    
                    # Tokenizers return input_ids as torch.long by default — KEEP IT!
                    description_inputs = self.description_tokenizer(
                        description,
                        return_tensors="pt"
                    ).to(self.device)

                    prompt_inputs = self.tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=400
                    ).to(self.device)
                    
                    tokenize_time = time.time() - tokenize_start
                    logger.info(f"🔊 Tokenization completed in {tokenize_time:.3f}s")
                    logger.info(f"🔊 Description tokens: {description_inputs.input_ids.shape}")
                    logger.info(f"🔊 Prompt tokens: {prompt_inputs.input_ids.shape}")

                    logger.info("🔊 Starting speech generation...")
                    logger.info(f"🔊 Model dtype: {self.model.dtype if hasattr(self.model, 'dtype') else 'unknown'}")
                    logger.info(f"🔊 Generation parameters: do_sample=True, temperature=1.0")
                    
                    generation_start = time.time()
                    generation = self.model.generate(
                        input_ids=description_inputs.input_ids,
                        attention_mask=description_inputs.attention_mask,
                        prompt_input_ids=prompt_inputs.input_ids,
                        prompt_attention_mask=prompt_inputs.attention_mask,
                        do_sample=True,
                        temperature=1.0,
                        max_new_tokens=512
                    )
                    generation_time = time.time() - generation_start
                    
                    # Handle dict vs tensor output
                    if isinstance(generation, dict):
                        logger.info(f"🔊 Generation returned dict with keys: {list(generation.keys())}")
                        audio_tensor = generation["sequences"]
                    else:
                        audio_tensor = generation

                    logger.info(f"🔊 Speech generation completed in {generation_time:.3f}s")
                    logger.info(f"🔊 Generated audio tensor shape: {audio_tensor.shape}")
                    return audio_tensor
                
            generation = await loop.run_in_executor(None, _generate_speech)
            logger.info("Converting generated audio to numpy array...")
            audio_arr = generation.cpu().float().numpy().squeeze()
            logger.info(f"Raw audio shape: {audio_arr.shape}, dtype: {audio_arr.dtype}")
            
            if audio_arr.ndim > 1:
                logger.info("Converting multi-dimensional audio to mono")
                audio_arr = audio_arr.mean(axis=0)
                logger.info(f"Mono audio shape: {audio_arr.shape}")
            
            logger.info("Processing and normalizing audio...")
            audio_arr = np.clip(audio_arr, -1.0, 1.0)
            max_abs = np.max(np.abs(audio_arr))
            if max_abs > 0:
                audio_arr = audio_arr / max_abs * 0.9
            else:
                logger.warning("Generated audio is completely silent!")
            
            logger.info(f"Saving audio to: {output_path}")
            save_start = time.time()
            sf.write(output_path, audio_arr, self.model.config.sampling_rate)
            save_time = time.time() - save_start
            
            file_size = os.path.getsize(output_path) / 1024
            duration = len(audio_arr) / self.model.config.sampling_rate
            
            logger.info(f"Audio saved in {save_time:.3f}s")
            logger.info(f"File size: {file_size:.1f} KB")
            logger.info(f"Audio duration: {duration:.2f} seconds")
            logger.info(f"Sample rate: {self.model.config.sampling_rate} Hz")
            logger.info("TTS synthesis completed successfully")
            return output_path
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}", exc_info=True)
            silence = np.zeros(24000, dtype=np.float32)
            sf.write(output_path, silence, 24000)
            logger.info(f"Fallback silence saved to: {output_path}")
            return output_path
