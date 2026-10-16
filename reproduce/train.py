#!/usr/bin/env python3
"""Retrain a selected model of the paper on RunBios, deploy it and evaluate it on the released test cases.

Example
  python reproduce/train.py --task banking-intent-routing

The configuration (base model, epochs, instruction format, training rows) is read from runs/run_log.csv and
data/<task>/train_selected_model.jsonl.gz, so the run repeats the selected configuration of that task. The script
uploads the rows in files of 5,000, creates a LoRA training job (bf16, maximum length 2,048), waits for it, deploys the
best checkpoint on the cheapest available GPU, runs reproduce/evaluate.py on the test set and stops the deployment.
RunBios bills the training job and the deployment hours to your account. Pass --keep to leave the deployment running.
"""
import argparse, csv, gzip, json, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runbios_client as rb

ROOT = Path(__file__).resolve().parent.parent


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--keep", action="store_true", help="leave the deployment running after evaluation")
    a = ap.parse_args()

    run = next(r for r in csv.DictReader(open(ROOT / "runs" / "run_log.csv")) if r["task"] == a.task and r["selected"] == "True")
    task = json.load(open(ROOT / "data" / a.task / "task.json"))
    short = run["instruction"] == "one sentence"
    system = task["instruction_one_sentence"] if short else task["instruction_with_label_list"]
    epochs = int(float(run["epochs"])) if run["epochs"] else 3
    rows = [json.loads(l) for l in gzip.open(ROOT / "data" / a.task / "train_selected_model.jsonl.gz", "rt", encoding="utf-8")]
    log(f"{a.task}: {run['base_model']}, {len(rows):,} rows, {epochs} epochs, instruction {'one sentence' if short else 'with label list'}")

    dataset_ids = []
    for i in range(0, len(rows), 5000):
        lines = [json.dumps({"messages": [{"role": "system", "content": system}, {"role": "user", "content": r["text"]},
                                          {"role": "assistant", "content": r["label"]}]}, ensure_ascii=False) for r in rows[i:i + 5000]]
        d = rb.upload_jsonl(f"{a.task}-reproduction-part{i // 5000}", lines)
        dataset_ids.append(d["id"])
        log(f"uploaded part {i // 5000} ({len(lines)} rows): {d['id']}")
    for did in dataset_ids:
        while True:
            d = rb.get(f"/api/datasets/{did}")
            d = d.get("dataset", d)
            if d.get("status") == "ready":
                break
            if d.get("status") in ("failed", "needs_mapping"):
                sys.exit(f"dataset {did} {d.get('status')}: {d.get('error') or d.get('validation')}")
            time.sleep(10)

    body = {"model_id": run["base_model"], "dataset_ids": dataset_ids, "training_method": "sft", "train_type": "lora",
            "name": f"{a.task} reproduction", "queue_if_unavailable": True,
            "config": {"num_train_epochs": epochs, "max_length": 2048, "precision": "bf16"}}
    pf = rb.post("/api/training/preflight", body)
    log(f"preflight valid={pf.get('valid')} estimate=${((pf.get('estimate') or {}).get('estimated_cents') or 0) / 100:.2f}")
    job = rb.post("/api/training/jobs", body, headers={"Idempotency-Key": f"repro-{a.task}-{int(time.time())}"})
    jid = job.get("job_id") or job.get("id") or (job.get("job") or {}).get("id")
    log(f"training job {jid}")
    last = None
    while True:
        j = rb.get(f"/api/training/jobs/{jid}")
        j = j.get("job", j)
        if j.get("status") != last:
            last = j.get("status")
            log(f"training {last}")
        if last in ("completed", "succeeded"):
            break
        if last in ("failed", "cancelled", "stopped"):
            sys.exit(f"training ended {last}: {j.get('error_message')}")
        time.sleep(30)
    cps = rb.get(f"/api/training/jobs/{jid}/checkpoints")
    cps = cps.get("checkpoints", cps)
    cp = sorted(cps, key=lambda c: (c.get("is_best") is True, c.get("step") or 0))[-1]
    log(f"checkpoint step {cp.get('step')} ({cp['id']})")

    dep = None
    while dep is None:
        opts = rb.get(f"/api/inference/gpu-options?model={run['base_model']}&serving_mode=adapter")["options"]
        for o in sorted((x for x in opts if x.get("selectable")), key=lambda x: x["market"]["minimum_configuration_hour_cents"]):
            try:
                out = rb.post("/api/inference", {"name": f"{a.task}-reproduction-{int(time.time()) % 100000}", "source_type": "checkpoint",
                                                 "source_job_id": jid, "source_checkpoint_id": cp["id"], "gpu_type": o["gpu_type"],
                                                 "gpu_count": o["min_gpus"], "gpu_tier": "secure", "allow_capacity_queue": False},
                              headers={"Idempotency-Key": f"repro-{jid}-{o['gpu_type']}-{int(time.time())}"})
                handle = (out.get("booking") or {}).get("handle")
                while handle and not (out.get("id") or (out.get("deployment") or {}).get("id")):
                    time.sleep(10)
                    out = rb.get(f"/api/inference/bookings/{handle}")
                dep = out.get("deployment") or out
                log(f"deployment {dep.get('id')} on {o['gpu_type']}")
                break
            except rb.ApiError as e:
                log(f"{o['gpu_type']} unavailable ({e.code}), trying the next GPU")
        if dep is None:
            time.sleep(60)
    did = dep["id"]
    while True:
        d = rb.get(f"/api/inference/{did}")
        d = d.get("deployment", d)
        if d.get("status") in ("running", "ready", "active"):
            break
        if d.get("status") in ("failed", "stopped", "deleted"):
            sys.exit(f"deployment {d.get('status')}")
        time.sleep(20)
    served = next(d.get(k) for k in ("served_model_name", "model_name", "model", "alias", "name") if d.get(k))
    log(f"serving as {served}")
    try:
        subprocess.run([sys.executable, str(Path(__file__).with_name("evaluate.py")), "--task", a.task, "--model", served,
                        "--instruction", "short" if short else "list"], check=False)
    finally:
        if not a.keep:
            rb.post(f"/api/inference/{did}/stop", {})
            log("deployment stopped")


if __name__ == "__main__":
    main()
