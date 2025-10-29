import os
import json
from typing import Any, Dict
import redis


def get_redis_client() -> redis.Redis:
    url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    return redis.from_url(url)


def set_json(key: str, value: Dict[str, Any], ex_seconds: int | None = 3600) -> None:
    r = get_redis_client()
    r.set(key, json.dumps(value), ex=ex_seconds)


def get_json(key: str) -> Dict[str, Any] | None:
    r = get_redis_client()
    raw = r.get(key)
    if not raw:
        return None
    return json.loads(raw)


