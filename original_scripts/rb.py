"""Tiny RunBios client used for the study. The key is read from RUNBIOS_API_KEY and never printed."""
import json, os, subprocess, time, urllib.error, urllib.request

BASE = os.environ.get("RUNBIOS_API_BASE", "https://api.runbios.ai")
_KEY = None


def key():
    global _KEY
    if _KEY is None:
        _KEY = os.environ["RUNBIOS_API_KEY"]
    return _KEY


class ApiError(Exception):
    def __init__(self, status, payload):
        self.status, self.payload = status, payload
        err = (payload or {}).get("error") if isinstance(payload, dict) else None
        self.code = (err or {}).get("code") if isinstance(err, dict) else (payload or {}).get("code") if isinstance(payload, dict) else None
        super().__init__(f"HTTP {status}: {json.dumps(payload)[:600]}")


def call(method, path, body=None, headers=None, timeout=300, retries=6):
    data = None if body is None else json.dumps(body).encode()
    h = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "runbios-casestudy/1"}
    h.update(headers or {"X-API-Key": key()})
    for attempt in range(retries):
        req = urllib.request.Request(BASE + path, data=data, method=method, headers=h)
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                return r.status, (json.loads(raw) if raw else {}), time.time() - t0
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {"raw": raw[:300].decode(errors="replace")}
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(min(60, 3 * 2 ** attempt))
                continue
            raise ApiError(e.code, payload) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt < retries - 1:
                time.sleep(min(60, 3 * 2 ** attempt))
                continue
            raise


def get(path):
    return call("GET", path)[1]


def post(path, body):
    return call("POST", path, body)[1]


def chat(model, messages, max_tokens=None, timeout=300, **extra):
    body = {"model": model, "messages": messages, "temperature": 0}
    if max_tokens:
        body["max_tokens"] = max_tokens
    body.update(extra)
    status, out, secs = call("POST", "/v1/chat/completions", body,
                             headers={"Authorization": "Bearer " + key(), "X-Loop-Capture": "off"},
                             timeout=timeout, retries=4)
    return out, secs
