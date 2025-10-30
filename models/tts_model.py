import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import asyncio
import numpy as np
from typing import Optional
import logging
from utils.audio_utils import trim_silence

logger = logging.getLogger(__name__)

class TTSModel:
    def __init__(self, 
                 model_name: str = "ai4bharat/indic-parler-tts",
                 hf_token: Optional[str] = None,
                 mock_mode: bool = False):
        self.mock_mode = mock_mode
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        if self.mock_mode:
            logger.info("TTS initialized in mock mode")
            return
            
        try:
            logger.info(f"Loading TTS model {model_name} on {self.device}")
            
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                model_name,
                token=hf_token
            ).to(self.device)
            # Force FP32 for stability on some GPUs
            try:
                self.model = self.model.to(torch.float32)
            except Exception:
                pass
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token
            )
            
            self.description_tokenizer = AutoTokenizer.from_pretrained(
                self.model.config.text_encoder._name_or_path
            )
             
            self.model.eval()
            logger.info("TTS model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            logger.info("Falling back to mock mode")
            self.mock_mode = True

    async def synthesize(self, 
                        text: str, 
                        output_path: str = "output.wav",
                        description: Optional[str] = None) -> str:
        if self.mock_mode:
            await asyncio.sleep(0.1)
            silence = np.zeros(24000, dtype=np.float32)
            sf.write(output_path, silence, 24000)
            return output_path
            
        try:
            if description is None:
                description = (
                    "Amrita speaks with a high pitch at a slow pace. Her voice is clear, with excellent recording quality and only moderate background noise."
                )
            
            logger.info(f"Synthesizing: '{text[:50]}...'")
            
            loop = asyncio.get_event_loop()
            
            def _generate_speech():
                # Deterministic, fp32, no autocast
                with torch.inference_mode():
                    description_input_ids = self.description_tokenizer(
                        description,
                        return_tensors="pt"
                    ).to(self.device)

                    prompt_input_ids = self.tokenizer(
                        text,
                        return_tensors="pt"
                    ).to(self.device)

                    # Greedy decoding for stability; reduce token count for latency
                    generation = self.model.generate(
                        input_ids=description_input_ids.input_ids,
                        attention_mask=description_input_ids.attention_mask,
                        prompt_input_ids=prompt_input_ids.input_ids,
                        prompt_attention_mask=prompt_input_ids.attention_mask
                    )

                    return generation
            
            generation = await loop.run_in_executor(None, _generate_speech)
            audio_arr = generation.cpu().numpy().squeeze()
            
            if audio_arr.ndim > 1:
                audio_arr = audio_arr.mean(axis=0)
            
            # Clamp and normalize audio to proper volume range
            audio_arr = np.clip(audio_arr, -1.0, 1.0)
            max_abs = np.max(np.abs(audio_arr))
            if max_abs > 0:
                # Normalize to 0.9 to prevent clipping
                audio_arr = audio_arr / max_abs * 0.9
                logger.info(f"Normalized audio from max {max_abs:.6f} to 0.9")
            else:
                logger.warning("Generated audio is completely silent!")
            
            sf.write(output_path, audio_arr, self.model.config.sampling_rate)
            
            logger.info(f"Generated audio: {len(audio_arr)} samples, range: [{audio_arr.min():.6f}, {audio_arr.max():.6f}]")
            return output_path
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            silence = np.zeros(24000, dtype=np.float32)
            sf.write(output_path, silence, 24000)
            return output_path