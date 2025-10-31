from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import io
import time
from dotenv import load_dotenv
from models.asr_model import ASRModel
from models.llm_model import LLMModel
from models.tts_model import TTSModel
from utils.audio_utils import save_uploaded_wav, load_audio_bytes, get_audio_duration, trim_silence
from utils.logger import log_request, latency_monitor
import logging
from typing import Dict, Any
import uuid
from celery.result import AsyncResult
from celery_app import celery_app
from tasks import process_pipeline_task
from utils.redis_client import get_json
import torch

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Speech Pipeline Latency Test",
    description="End-to-end speech processing pipeline with latency measurement",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
app.middleware("http")(log_request)

# Configuration from environment
HF_TOKEN = os.getenv("HF_ACCESS_TOKEN")
ASR_MOCK_MODE = os.getenv("ASR_MOCK_MODE", "false").lower() == "true"
LLM_ECHO_MODE = os.getenv("LLM_ECHO_MODE", "false").lower() == "true"
TTS_MOCK_MODE = os.getenv("TTS_MOCK_MODE", "false").lower() == "true"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

# Model names from environment
ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME", "ai4bharat/indic-conformer-600m-multilingual")
TTS_MODEL_NAME = os.getenv("TTS_MODEL_NAME", "parler-tts/parler-tts-mini-v1.1")

# Create output directory
os.makedirs("data/outputs", exist_ok=True)

# Initialize models
logger.info("Initializing models...")
asr = ASRModel(model_name=ASR_MODEL_NAME, hf_token=HF_TOKEN, mock_mode=ASR_MOCK_MODE)
llm = LLMModel(ollama_url=OLLAMA_URL, model_name=OLLAMA_MODEL, echo_mode=LLM_ECHO_MODE)
tts = TTSModel(model_name=TTS_MODEL_NAME, hf_token=HF_TOKEN, mock_mode=TTS_MOCK_MODE)
logger.info("Models initialized successfully")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    logger.info(f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

# Startup event to preload LLM model and initialize database
@app.on_event("startup")
async def startup_event():
    """Preload LLM model on startup and initialize database."""
    # Initialize database tables
    try:
        from utils.database import init_db
        logger.info("Initializing database...")
        init_db()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        logger.warning("Service will continue, but database features may not work")
    
    # Preload LLM model
    logger.info("Preloading LLM model...")
    success = await llm.preload_model()
    if success:
        logger.info("LLM model preloaded successfully")
    else:
        logger.warning("LLM model preload failed, but service will continue")
    
    # Log GPU memory usage after model loading
    if torch.cuda.is_available():
        logger.info(f"GPU memory after model loading:")
        logger.info(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        logger.info(f"  Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        logger.info(f"  Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1024**3:.2f} GB")

@app.post("/pipeline")
async def process_speech_pipeline(
    request: Request,
    file: UploadFile = File(...)
) -> StreamingResponse:
    """
    Main speech processing pipeline endpoint.
    
    Processes uploaded WAV audio through ASR -> LLM -> TTS pipeline
    and returns synthesized audio with latency measurements.
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    pipeline_start = time.time()
    
    logger.info(f"🚀 Starting pipeline processing for request: {request_id}")
    logger.info(f"🚀 File: {file.filename}, Content-Type: {file.content_type}")
    logger.info(f"🚀 File size: {file.size if hasattr(file, 'size') else 'unknown'} bytes")
    
    try:
        # Validate file type
        if not file.filename.lower().endswith('.wav'):
            logger.error(f"❌ Invalid file type: {file.filename}")
            raise HTTPException(status_code=400, detail="Only WAV files are supported")
        
        logger.info(f"✅ File validation passed: {file.filename}")
        
        # Save input audio with validation
        input_path = f"data/outputs/{request_id}_{file.filename}"
        logger.info(f"💾 Saving uploaded file to: {input_path}")
        save_uploaded_wav(file, input_path)
        
        # Get audio duration for logging
        audio_duration = get_audio_duration(input_path)
        logger.info(f"🎵 Audio file loaded: {audio_duration:.2f}s duration")
        logger.info(f"🎵 Input file size: {os.path.getsize(input_path) / 1024:.1f} KB")
        
        # --- ASR Stage (Hindi only) ---
        logger.info(f"🎤 Starting ASR stage for request: {request_id}")
        latency_monitor.start_stage(request_id, "asr")
        asr_start = time.time()
        transcribed_text = await asr.transcribe(input_path)
        asr_time = latency_monitor.end_stage(request_id, "asr")
        asr_elapsed = time.time() - asr_start
        
        logger.info(f"✅ ASR completed in {asr_elapsed:.3f}s")
        logger.info(f"🎤 Transcribed text length: {len(transcribed_text)} characters")
        logger.info(f"🎤 Transcribed text: '{transcribed_text[:150]}{'...' if len(transcribed_text) > 150 else ''}'")
        
        # --- LLM Stage with Database Context ---
        logger.info(f"🧠 Starting LLM stage for request: {request_id}")
        latency_monitor.start_stage(request_id, "llm")
        llm_start = time.time()
        
        # Query PostgreSQL for relevant context
        db_context_text = ""
        db_context = {"query_type": "none"}
        try:
            from utils.medical_db_query import MedicalDBQuery
            db_query = MedicalDBQuery() 
            db_context = db_query.query_by_user_question(transcribed_text)
            db_context_text = db_query.format_context_for_llm(db_context)
            db_query.close()
            logger.info(f"🔍 Database context retrieved: {db_context.get('query_type', 'unknown')}")
        except Exception as e:
            logger.warning(f"⚠️ Database query failed, proceeding without context: {e}")
            # Continue without database context
        
        # Generate response with database context
        response_text = await llm.generate_response_with_context(
            transcribed_text, 
            db_context=db_context_text
        )
        
        llm_time = latency_monitor.end_stage(request_id, "llm")
        llm_elapsed = time.time() - llm_start
        
        logger.info(f"✅ LLM completed in {llm_elapsed:.3f}s")
        logger.info(f"🧠 Response text length: {len(response_text)} characters")
        logger.info(f"🧠 Response text: '{response_text[:150]}{'...' if len(response_text) > 150 else ''}'")
        
        # --- TTS Stage (Hindi only) ---
        logger.info(f"🔊 Starting TTS stage for request: {request_id}")
        latency_monitor.start_stage(request_id, "tts")
        tts_start = time.time()
        output_path = f"data/outputs/{request_id}_response_{file.filename}"
        
        # Enhanced voice description for better quality
        description = (
            "A professional Indian female speaker with a warm, clear voice delivers speech at a moderate pace. "
            "Her voice is clear and reassuring, with excellent recording quality and minimal background noise. "
            "She speaks Hindi naturally with proper pronunciation and Indian intonation."
        )
        
        logger.info(f"🔊 Output path: {output_path}")
        logger.info(f"🔊 Voice description: '{description[:100]}{'...' if len(description) > 100 else ''}'")
        
        await tts.synthesize(response_text, output_path, description=description)
        tts_time = latency_monitor.end_stage(request_id, "tts")
        tts_elapsed = time.time() - tts_start
        
        logger.info(f"✅ TTS completed in {tts_elapsed:.3f}s")
        logger.info(f"🔊 Output file size: {os.path.getsize(output_path) / 1024:.1f} KB")
        
        # Get complete latency report
        latency_report = latency_monitor.get_report(request_id)
        total_pipeline_time = time.time() - pipeline_start
        
        # Prepare response audio stream
        logger.info(f"📦 Preparing response audio stream...")
        audio_bytes = load_audio_bytes(output_path)
        logger.info(f"📦 Audio stream size: {len(audio_bytes) / 1024:.1f} KB")
        
        # Prepare response headers with detailed metrics
        # Encode Unicode text safely for HTTP headers
        import base64
        transcribed_b64 = base64.b64encode(transcribed_text[:200].encode('utf-8')).decode('ascii')
        response_b64 = base64.b64encode(response_text[:200].encode('utf-8')).decode('ascii')
        
        headers = {
            "X-Request-ID": request_id,
            "X-ASR-Latency-Ms": f"{latency_report.get('asr_time_ms', 0):.2f}",
            "X-LLM-Latency-Ms": f"{latency_report.get('llm_time_ms', 0):.2f}",
            "X-TTS-Latency-Ms": f"{latency_report.get('tts_time_ms', 0):.2f}",
            "X-Total-Latency-Ms": f"{latency_report.get('total_time_ms', 0):.2f}",
            "X-Audio-Duration-S": f"{audio_duration:.2f}",
            "X-Detected-Language": "hindi",
            "X-Transcribed-Text-B64": transcribed_b64,
            "X-Response-Text-B64": response_b64,
            "Content-Disposition": f"attachment; filename=response_{file.filename}"
        }
        
        # Final pipeline summary
        logger.info(f"🎉 Pipeline completed successfully for request: {request_id}")
        logger.info(f"📊 Performance Summary:")
        logger.info(f"  🎤 ASR: {asr_elapsed:.3f}s ({latency_report.get('asr_time_ms', 0):.1f}ms)")
        logger.info(f"  🧠 LLM: {llm_elapsed:.3f}s ({latency_report.get('llm_time_ms', 0):.1f}ms)")
        logger.info(f"  🔊 TTS: {tts_elapsed:.3f}s ({latency_report.get('tts_time_ms', 0):.1f}ms)")
        logger.info(f"  ⏱️  Total: {total_pipeline_time:.3f}s ({latency_report.get('total_time_ms', 0):.1f}ms)")
        logger.info(f"  🎵 Audio: {audio_duration:.2f}s input → {len(audio_bytes) / 1024:.1f} KB output")
        
        # Calculate efficiency metrics
        real_time_factor = total_pipeline_time / audio_duration if audio_duration > 0 else 0
        logger.info(f"  ⚡ Real-time factor: {real_time_factor:.2f}x")
        logger.info(f"  🚀 Processing speed: {audio_duration/total_pipeline_time:.2f}x faster than real-time" if real_time_factor < 1 else f"  🐌 Processing speed: {real_time_factor:.2f}x slower than real-time")
        
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers=headers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Pipeline failed: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={
                "error": "Pipeline processing failed",
                "detail": str(e),
                "request_id": request_id
            }
        )

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint with model status.
    """
    try:
        # Check LLM service if not in echo mode
        llm_status = True
        if not llm.echo_mode:
            llm_status = await llm.health_check()
        
        return {
            "status": "healthy",
            "models": {
                "asr": {
                    "status": "ready",
                    "mock_mode": asr.mock_mode
                },
                "llm": {
                    "status": "ready" if llm_status else "unavailable",
                    "echo_mode": llm.echo_mode,
                    "ollama_url": llm.ollama_url if not llm.echo_mode else None
                },
                "tts": {
                    "status": "ready",
                    "mock_mode": tts.mock_mode
                }
            },
            "configuration": {
                "asr_mock_mode": ASR_MOCK_MODE,
                "llm_echo_mode": LLM_ECHO_MODE,
                "tts_mock_mode": TTS_MOCK_MODE
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )

@app.get("/metrics/{request_id}")
async def get_metrics(request_id: str) -> Dict[str, Any]:
    """
    Get detailed metrics for a specific request.
    """
    report = latency_monitor.get_report(request_id)
    if not report:
        raise HTTPException(status_code=404, detail="Request ID not found")
    
    return {
        "request_id": request_id,
        "latency_report": report,
        "timestamp": latency_monitor.timings.get(request_id, {})
    }


@app.post("/pipeline_async")
async def pipeline_async() -> Dict[str, Any]:
    """
    Enqueue the dummy ASR→LLM→TTS Celery task and return a task_id.
    """
    session_id = str(uuid.uuid4())
    task = process_pipeline_task.delay(session_id, {"note": "demo"})
    return {"task_id": task.id, "session_id": session_id}


@app.get("/status/{task_id}")
async def task_status(task_id: str) -> Dict[str, Any]:
    """
    Check Celery task status/result from Redis backend.
    Also returns any cached session record.
    """
    res = AsyncResult(task_id, app=celery_app)
    payload: Dict[str, Any] = {
        "task_id": task_id,
        "state": res.state,
        "ready": res.ready(),
    }
    if res.ready():
        try:
            payload["result"] = res.get(timeout=0)
        except Exception:
            pass
    # Attempt to pull any session record by scanning known key
    # The client should provide session_id ideally; for demo, include when enqueueing
    payload["cached_session_example"] = get_json(f"session:{payload.get('result', {}).get('session_id', '')}") if isinstance(payload.get("result"), dict) else None
    return payload

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    await llm.close()
    logger.info("Application shutdown complete")
