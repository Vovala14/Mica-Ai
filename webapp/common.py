"""Shared helpers for the MICA test-site functions (api/flame.py, api/ember.py)."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_PROMPT = 1000


def sign(gen_id: str, model: str, mode: str, prompt: str, output: str) -> str:
    """HMAC that api/log.js checks, so only outputs this server made get logged."""
    key = os.environ.get("LOG_SECRET", "")
    if not key:
        return ""
    msg = "\n".join([gen_id, model, mode, prompt, output]).encode("utf-8")
    return hmac.new(key.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def clean_prompt(body: dict) -> str:
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Please type a prompt.")
    if len(prompt) > MAX_PROMPT:
        raise ValueError(f"Prompts are limited to {MAX_PROMPT} characters.")
    return prompt


def result(model: str, mode: str, prompt: str, output: str, t0: float, **extra) -> dict:
    gen_id = uuid.uuid4().hex
    return {"id": gen_id, "model": model, "mode": mode, "prompt": prompt,
            "output": output, "ms": round((time.perf_counter() - t0) * 1000),
            "sig": sign(gen_id, model, mode, prompt, output), **extra}


class JsonHandler(BaseHTTPRequestHandler):
    """Base for the Vercel Python functions: POST JSON to run(body) -> dict.

    Each api/*.py file must declare `class handler(JsonHandler)` literally,
    because Vercel only detects a Python function from that source line.
    """

    run = None  # set by the subclass as staticmethod(run)

    def _send(self, code: int, obj: dict) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        try:
            n = int(self.headers.get("content-length") or 0)
            if n > 16_384:
                raise ValueError("Request too large.")
            body = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("Bad request.")
            self._send(200, self.run(body))
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except Exception:
            traceback.print_exc()
            self._send(500, {"error": "The model failed on this prompt."})

    def do_GET(self):
        self._send(405, {"error": "Use POST."})
