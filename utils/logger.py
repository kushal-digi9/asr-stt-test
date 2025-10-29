import time
import logging
import uuid
from typing import Dict, Optional
from fastapi import Request
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class LatencyMonitor:
    
    def __init__(self):
        self.timings: Dict[str, Dict[str, float]] = {}
        self.logger = logging.getLogger(__name__)
    
    def start_stage(self, request_id: str, stage_name: str) -> str:
        timing_id = f"{request_id}_{stage_name}"
        
        if request_id not in self.timings:
            self.timings[request_id] = {}
            
        self.timings[request_id][f"{stage_name}_start"] = time.perf_counter()
        return timing_id
    
    def end_stage(self, request_id: str, stage_name: str) -> float:
        end_time = time.perf_counter()
        
        if request_id not in self.timings:
            self.logger.warning(f"No timing data found for request {request_id}")
            return 0.0
            
        start_key = f"{stage_name}_start"
        if start_key not in self.timings[request_id]:
            self.logger.warning(f"No start time found for {stage_name} in request {request_id}")
            return 0.0
            
        start_time = self.timings[request_id][start_key]
        duration_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Store the duration
        self.timings[request_id][f"{stage_name}_duration"] = duration_ms
        
        self.logger.info(f"[{request_id}] {stage_name} completed in {duration_ms:.2f}ms")
        return duration_ms
    
    def get_report(self, request_id: str) -> Dict[str, float]:
        if request_id not in self.timings:
            return {}
            
        timings = self.timings[request_id]
        report = {}
        
        # Extract duration values
        for key, value in timings.items():
            if key.endswith('_duration'):
                stage_name = key.replace('_duration', '')
                report[f"{stage_name}_time_ms"] = round(value, 2)
        
        # Calculate total time
        total_time = sum(v for k, v in report.items() if k.endswith('_time_ms'))
        report['total_time_ms'] = round(total_time, 2)
        
        return report
    
    def cleanup_request(self, request_id: str):
        if request_id in self.timings:
            del self.timings[request_id]

# Global latency monitor instance
latency_monitor = LatencyMonitor()

def log_latency(stage: str, start_time: float) -> float:
    duration = time.perf_counter() - start_time
    logging.info(f"{stage} took {duration:.3f} sec")
    return duration

async def log_request(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()
    
    # Add request ID to request state
    request.state.request_id = request_id
    
    logger = logging.getLogger(__name__)
    logger.info(f"[{request_id}] {request.method} {request.url.path} - Started")
    
    try:
        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} - "
            f"Completed in {process_time:.3f}s (Status: {response.status_code})"
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response
        
    except Exception as e:
        process_time = time.perf_counter() - start_time
        logger.error(
            f"[{request_id}] {request.method} {request.url.path} - "
            f"Failed after {process_time:.3f}s: {str(e)}"
        )
        raise
    finally:
        latency_monitor.cleanup_request(request_id)
