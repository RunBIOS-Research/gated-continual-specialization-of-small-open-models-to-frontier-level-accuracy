#!/usr/bin/env python3
"""Fine-tune, deploy and score one use case end to end on RunBios.

  python ft_runner.py <uc> "<title>"

1. waits for the uploaded training dataset to be ready
2. creates a RunBios managed training job (Llama-3.1-8B-Instruct, LoRA SFT)
3. waits for it to finish, deploys its final checkpoint on RunBios
4. scores the deployed model on the same 500 frozen records as the frontier
   models (evaluate.py, label "ft"), and records the deployment's hourly
   price and the throughput measured while scoring
Everything it learns is written to work/<uc>/ft_state.json.
"""
import json, os, subprocess, sys, time, uuid
from pathlib import Path

import rb

ROOT = Path(__file__).resolve().parent
uc, title = sys.argv[1], sys.argv[2]
VER = sys.argv[3] if len(sys.argv) > 3 else ""
BASE = sys.argv[4] if len(sys.argv) > 4 else "meta-llama/Llama-3.1-8B-Instruct"
KW = {"chat_template_kwargs": {"enable_thinking": False}} if ("qwen" in BASE.lower() or "gemma" in BASE.lower()) else {}
SERVE = "merged" if "gemma" in BASE.lower() else "adapter"
d = ROOT / "work" / uc
sp = d / f"ft{VER}_state.json"
st = json.load(open(sp)) if sp.exists() else {}
save = lambda: json.dump(st, open(sp, "w"), indent=1)


def log(m):
    print(f"{time.strftime('%H:%M:%S')} [{uc}] {m}", flush=True)


EPOCHS = int(sys.argv[5]) if len(sys.argv) > 5 else 3
if VER.endswith("hy"):
    files = sorted(p.name for p in d.glob("dataset_h*.json")) + sorted(p.name for p in d.glob("dataset_y*.json"))
elif VER.endswith("Flong"):
    files = sorted((p.name for p in d.glob("dataset_F*.json")), key=lambda n: int(n[len("dataset_F"):-5]))
elif "short" in VER:
    os.environ["CS_SYSTEM"] = "system_prompt_short"
    files = sorted((p.name for p in d.glob("dataset_S*.json")), key=lambda n: int(n[len("dataset_S"):-5]))
elif VER.endswith("full"):
    pre = {"ledgar": "dataset_L", "finer": "dataset_F", "cve": None}[uc]
    if pre:
        files = sorted((p.name for p in d.glob(pre + "*.json")), key=lambda n: int(n[len(pre):-5]))
    else:
        files = sorted(p.name for p in d.glob("dataset_h*.json")) + sorted(p.name for p in d.glob("dataset_y*.json")) + \
            [f for f in ("dataset.json", "dataset_b.json", "dataset_c.json", "dataset_d.json") if (d / f).exists()]
elif VER.endswith("rec"):
    files = sorted(p.name for p in d.glob("dataset_h*.json")) + \
        [f for f in ("dataset.json", "dataset_b.json", "dataset_c.json", "dataset_d.json") if (d / f).exists()]
elif VER.endswith("all"):
    files = sorted(p.name for p in d.glob("dataset*.json") if '"id"' in p.read_text() and "detail" not in p.read_text())
elif VER.startswith("q") and VER.endswith("x"):
    files = [f for f in ("dataset.json", "dataset_b.json", "dataset_c.json", "dataset_d.json") if (d / f).exists()] \
        + [f"dataset_x{i}.json" for i in range(8)]
elif VER.startswith("q"):
    files = [f for f in ("dataset.json", "dataset_b.json", "dataset_c.json", "dataset_d.json") if (d / f).exists()]
else:
    files = ["dataset.json"] + (["dataset_b.json"] if VER in ("2", "3") else []) + (["dataset_c.json"] if VER == "3" else []) \
        + (["dataset_d.json"] if VER == "3" and (d / "sft5k_d.jsonl").exists() else [])
while not all((d / f).exists() for f in files):
    time.sleep(15)
ds_ids = [json.load(open(d / f))["id"] for f in files]
total = 0
for ds in ds_ids:
    while True:
        x = rb.get(f"/api/datasets/{ds}")
        if x.get("status") == "ready":
            total += x.get("num_samples") or 0
            break
        if x.get("status") == "failed":
            sys.exit(f"dataset failed: {x.get('error_message')}")
        time.sleep(20)
st["dataset"] = {"ids": ds_ids, "num_samples": total}
save()

if not st.get("job_id"):
    if not VER and (d / "train_job.json").exists() and json.load(open(d / "train_job.json")).get("job_id"):
        st["job_id"] = json.load(open(d / "train_job.json"))["job_id"]
    else:
        name = f"Case study - {title} v{VER or 1} ({BASE.split('/')[-1]})"
        body = {"model_id": BASE, "dataset_ids": ds_ids, "training_method": "sft", "train_type": "lora",
                "name": name, "queue_if_unavailable": True, "config": {"num_train_epochs": EPOCHS, "max_length": 2048, "precision": "bf16"}}
        pf = rb.post("/api/training/preflight", body)
        req = dict(pf["canonical_request"])
        _, jout, _ = rb.call("POST", "/api/training/jobs", req, headers={"X-API-Key": rb.key(), "Idempotency-Key": f"cs-{uc}-v{VER or 1}-{int(time.time())}"})
        st["job_id"] = jout.get("job_id") or next(x["id"] for x in (rb.get("/api/training/jobs?limit=20").get("jobs") or []) if x.get("name") == name)
        st["estimate"] = pf.get("estimate")
    save()
job_id = st["job_id"]
log(f"job {job_id}")

last = None
while True:
    j = rb.get(f"/api/training/jobs/{job_id}")
    j = j.get("job", j)
    s = j.get("status")
    if s != last:
        log(f"training status {s}")
        last = s
    if s in ("completed", "succeeded"):
        st["job"] = {k: j.get(k) for k in ("status", "created_at", "started_at", "completed_at", "finished_at",
                                         "total_cost_cents", "cost_cents", "billed_cents", "tokens_trained",
                                         "token_charge", "billing_mode", "progress")}
        save()
        break
    if s in ("failed", "cancelled", "stopped"):
        sys.exit(f"training ended {s}: {j.get('error_message')}")
    time.sleep(30)

cps = rb.get(f"/api/training/jobs/{job_id}/checkpoints")
cps = cps.get("checkpoints", cps) if isinstance(cps, dict) else cps
cp = sorted(cps, key=lambda c: (c.get("is_best") is True, c.get("step") or 0))[-1]
st["checkpoint"] = {k: cp.get(k) for k in ("id", "step", "is_best", "eval_loss", "kind", "label")}
save()
log(f"checkpoint {st['checkpoint']}")

if not st.get("deployment_id"):
    out = None
    while out is None:
        opts = rb.get(f"/api/inference/gpu-options?model={BASE}&serving_mode={SERVE}")["options"]
        for o in sorted((x for x in opts if x.get("selectable")), key=lambda x: x["market"]["minimum_configuration_hour_cents"]):
            body = {"name": f"cs-{uc}-ft{VER}-{job_id[:6]}-{int(time.time()) % 10000}", "source_type": "checkpoint", "source_job_id": job_id, "source_checkpoint_id": cp["id"],
                    "gpu_type": o["gpu_type"], "gpu_count": o["min_gpus"], "gpu_tier": "secure", "allow_capacity_queue": False}
            try:
                s, out, _ = rb.call("POST", "/api/inference", body, headers={"X-API-Key": rb.key(),
                                    "Idempotency-Key": f"cs4{VER}-{uc}-{cp['id']}-{o['gpu_type']}-{int(time.time())}"}, retries=1)
                log(f"deploy create on {o['gpu_type']} {s}: {json.dumps(out)[:300]}")
                handle = (out.get("booking") or {}).get("handle")
                while handle and not out.get("id") and not (out.get("deployment") or {}).get("id"):
                    time.sleep(10)
                    out = rb.get(f"/api/inference/bookings/{handle}")
                    if (out.get("booking") or {}).get("status") in ("failed", "expired"):
                        raise rb.ApiError(409, out)
                break
            except rb.ApiError as e:
                log(f"deploy on {o['gpu_type']} refused: {e.code} {str(e)[:160]}")
                out = None
        if out is None:
            time.sleep(60)
    dep = out.get("deployment") or out
    st["deployment_id"] = dep.get("id")
    st["deployment_create"] = {k: dep.get(k) for k in ("id", "name", "model", "gpu_type", "gpu_count", "price_per_hour_cents",
                                                       "total_price_per_hour_cents", "status")}
    save()
    log(f"deployment {st['deployment_id']}")

while True:
    dep = rb.get(f"/api/inference/{st['deployment_id']}")
    dep = dep.get("deployment", dep)
    if dep.get("status") in ("running", "ready", "active"):
        break
    if dep.get("status") in ("failed", "stopped", "deleted"):
        sys.exit(f"deployment {dep.get('status')}: {dep.get('error_message')}")
    time.sleep(20)
st["deployment"] = {k: v for k, v in dep.items() if isinstance(v, (str, int, float, bool)) or v is None}
save()
names = [dep.get(k) for k in ("model", "model_name", "served_model_name", "alias", "name") if dep.get(k)]
model = None
for n in names:
    try:
        rb.chat(n, [{"role": "user", "content": "ping"}], max_tokens=2, timeout=120, **KW)
        model = n
        break
    except Exception as e:
        log(f"model name {n!r} did not answer: {str(e)[:120]}")
if not model:
    sys.exit(f"no callable model name among {names}")
st["model"] = model
save()
t0 = time.time()
xtra = ["--no-think"] if KW else []
subprocess.run([sys.executable, str(ROOT / "evaluate.py"), uc, model, "--label", f"ft{VER}", "--workers", "8"] + xtra, check=False)
subprocess.run([sys.executable, str(ROOT / "evaluate.py"), uc, model, "--label", f"ft{VER}", "--workers", "8", "--set", "val"] + xtra, check=False)
wall = time.time() - t0
rows = [json.loads(l) for l in open(d / "results" / f"ft{VER}.jsonl")]
ok = [r for r in rows if r.get("answer") is not None]
hourly = next((dep.get(k) for k in ("cost_per_hour_cents", "total_price_per_hour_cents", "price_per_hour_cents", "hourly_price_cents") if dep.get(k)), None)
st["serving"] = {"hourly_cents": hourly, "gpu": f"{dep.get('gpu_count')}x {dep.get('gpu_type')}",
                 "throughput_rps": round(len(ok) / wall, 3), "wall_seconds": round(wall, 1), "concurrency": 8}
save()
log(f"DONE {st['serving']}")
