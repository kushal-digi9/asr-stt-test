import os
from celery import Celery


def _broker_url() -> str:
    return os.getenv("REDIS_URL", "redis://redis:6379/0")


celery_app = Celery(
    "voice_ai",
    broker=_broker_url(),
    backend=_broker_url(),
)

# Reasonable defaults for demo
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    task_track_started=True,
    broker_connection_retry_on_startup=True,
)


