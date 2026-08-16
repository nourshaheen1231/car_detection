import os
import httpx

LARAVEL_CALLBACK_URL = os.getenv(
    "LARAVEL_CALLBACK_URL",
    "http://localhost:8000/api/internal/ai/video-result"
)

def send_callback(payload: dict) -> bool:
    try:
        response = httpx.post(LARAVEL_CALLBACK_URL, json=payload, timeout=60.0)
        print(f"[CALLBACK] Status: {response.status_code}")
        if response.status_code >= 400:
            print(f"[CALLBACK] Body: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"[CALLBACK ERROR] {e}")
        return False