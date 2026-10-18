"""Minimal client for the public RunBios API (standard library only).

Configuration comes from the environment and is never written anywhere:
  RUNBIOS_API_KEY   your RunBios API key (required)
  RUNBIOS_API_BASE  API address, default https://api.runbios.ai
"""
import json, os, time, urllib.error, urllib.request

BASE = os.environ.get("RUNBIOS_API_BASE", "https://api.runbios.ai").rstrip("/")


def _key():
    k = os.environ.get("RUNBIOS_API_KEY", "").strip()
    if not k:
        raise SystemExit("Set RUNBIOS_API_KEY to your RunBios API key.")
    return k


class ApiError(Exception):
    def __init__(self, status, payload):
        self.status, self.payload = status, payload
        err = payload.get("error") if isinstance(payload, dict) else None
        self.code = err.get("code") if isinstance(err, dict) else None
        super().__init__(f"HTTP {status}: {json.dumps(payload)[:400]}")


def call(method, path, body=None, headers=None, timeout=300, retries=6, raw_body=None):
    h = {"Accept": "application/json", "User-Agent": "runbios-paper-reproduce/1.0"}
    # model calls (/v1) take a bearer token, platform calls (/api) take the X-API-Key header
    if path.startswith("/v1/"):
        h["Authorization"] = "Bearer " + _key()
    else:
        h["X-API-Key"] = _key()
    data = raw_body
    if body is not None:
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    h.update(headers or {})
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


def post(path, body, headers=None):
    return call("POST", path, body, headers=headers)[1]


def chat(model, messages, **extra):
    body = {"model": model, "messages": messages, "temperature": 0}
    body.update(extra)
    _, out, secs = call("POST", "/v1/chat/completions", body, retries=4)
    return out, secs


def upload_jsonl(name, lines):
    """Upload a JSONL training file (multipart form) and return the dataset record."""
    boundary = "----runbios-paper-" + str(int(time.time() * 1000))
    payload = "\n".join(lines).encode() + b"\n"
    parts = []
    ws = (get("/api/api-keys/introspect").get("workspace") or {}).get("id", "")
    for k, v in (("name", name), ("format", "jsonl"), ("workspace_id", ws)):
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}.jsonl\"\r\n"
                 f"Content-Type: application/jsonl\r\n\r\n".encode() + payload + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    _, out, _ = call("POST", "/api/datasets/upload", raw_body=b"".join(parts),
                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, timeout=1800, retries=3)
    return out
