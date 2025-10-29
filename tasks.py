import time
import random
from typing import Dict, Any
from celery import shared_task
from utils.redis_client import set_json


@shared_task(name="process_pipeline")
def process_pipeline_task(session_id: str, audio_info: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Dummy ASR → LLM → TTS pipeline simulation.
    Sleeps to simulate latency and stores a session record in Redis.
    """
    # Simulate ASR latency
    time.sleep(1 + random.random())
    asr_text = "नमस्ते, यह एक डेमो है।"

    # Simulate LLM latency
    time.sleep(1 + random.random())
    llm_text = "नमस्ते उपयोगकर्ता, मैं आपकी कैसे सहायता कर सकती हूँ?"

    # Simulate TTS latency
    time.sleep(1 + random.random())
    tts_output = f"data/outputs/{session_id}_dummy.wav"

    result = {"session_id": session_id, "asr": asr_text, "llm": llm_text, "tts": tts_output}

    set_json(f"session:{session_id}", result)
    return result


