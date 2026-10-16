#!/usr/bin/env python3
"""Drive one Conscious Loop pipeline for one use case, version after version.

  python run_pipeline.py <uc> [--model meta-llama/Llama-3.1-8B-Instruct] [--versions 5]

1. imports the 200 frozen benchmark rows (own source, no pipeline tag, right
   answer attached) and pins them as an EXACT-match benchmark
2. creates the pipeline (promotion manual), attaches the benchmark as primary
3. per attempt: imports the next batch (log-growing), trains now, waits for the
   attempt's comparison and benchmark replay, then promotes it if and only if
   the candidate's exact benchmark score is higher than the live model's,
   otherwise rejects it. Every decision is logged with its numbers.

State (ids, decisions) is kept in work/<uc>/state.json so a restart resumes.
"""
import argparse, json, sys, time
from pathlib import Path

import rb

ROOT = Path(__file__).resolve().parent
BATCHES = [4000, 6000, 8000, 8000, 8000]
TERMINAL = {"promoted", "rejected", "failed", "budget_stopped", "expired", "cancelled", "superseded", "rolled_back"}
DECIDABLE = {"reported", "awaiting_review"}


def log(uc, msg):
    line = f"{time.strftime('%H:%M:%S')} [{uc}] {msg}"
    print(line, flush=True)
    with open(ROOT / "work" / uc / "run.log", "a") as f:
        f.write(line + "\n")


def rows(uc, name):
    return [json.loads(l) for l in open(ROOT / "work" / uc / f"{name}.jsonl")]


def convo(system, r, with_answer=True):
    m = [{"role": "system", "content": system}, {"role": "user", "content": r["text"]}]
    if with_answer:
        m.append({"role": "assistant", "content": r["label"]})
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("uc")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--versions", type=int, default=5)
    ap.add_argument("--poll", type=int, default=45)
    a = ap.parse_args()
    uc = a.uc
    d = ROOT / "work" / uc
    meta = json.load(open(d / "meta.json"))
    system = meta["system_prompt"]
    sp = d / "state.json"
    st = json.load(open(sp)) if sp.exists() else {"fed": 0, "decisions": []}
    save = lambda: json.dump(st, open(sp, "w"), indent=1)
    ws = rb.get("/api/api-keys/introspect")["workspace"]["id"]
    tag = f"cs-{uc}"

    if not st.get("benchmark_id"):
        bench = rows(uc, "bench")
        body = {"workspace_id": ws, "source": f"{tag}-benchmark",
                "rows": [{"messages": convo(system, r), "label": True} for r in bench],
                "row_ids": [f"{tag}-bench-{r['id']}" for r in bench]}
        out = rb.post("/api/loop/import", body)
        tids = out.get("trace_ids") or []
        log(uc, f"benchmark import: imported={out.get('imported')} present={out.get('already_present')} "
                f"reviewed={out.get('reviewed')} refused={out.get('refused')} traces={len(tids)}")
        if len(tids) != len(bench):
            sys.exit(f"expected {len(bench)} trace ids, got {len(tids)}: {json.dumps(out)[:800]}")
        b = rb.post("/api/loop/benchmarks", {"name": f"{meta['title']} (exact, 200 frozen real records, {time.strftime('%Y-%m-%d')})",
                                             "source": {"kind": "traces", "trace_ids": tids}, "check": "exact"})["benchmark"]
        st["benchmark_id"], st["benchmark_digest"] = b["id"], b.get("items_digest")
        save()
        log(uc, f"benchmark pinned {b['id']} items={b.get('item_count')} check={b.get('check')}")

    if not st.get("pipeline_id"):
        body = {"name": f"Case study: {meta['title']}", "use_case": meta["title"] + ". " + meta["provenance"][:300],
                "model": {"id": a.model}, "train_when": {"schedule": None},
                "promotion": {"policy": "manual"}, "max_versions": a.versions, "enabled": True}
        while True:
            try:
                p = rb.post("/api/loop/pipelines", body)["pipeline"]
                break
            except rb.ApiError as e:
                if e.code not in ("NO_GPU_FITS", "GPU_OPTIONS_DOWN", "MODEL_REVISION_UNAVAILABLE") and "cannot take turns" not in str(e) and "kinds of machine" not in str(e) and "kind of machine" not in str(e):
                    raise
                log(uc, f"pipeline save refused {e.code} (fix not deployed yet?); retrying in 60s")
                time.sleep(60)
        st["pipeline_id"], st["own_tag"], st["serving_name"] = p["rule_id"], p.get("own_tag"), p.get("serving_name")
        save()
        log(uc, f"pipeline created {p['rule_id']} tag={p.get('own_tag')} serving={p.get('serving_name')}")
        rb.post(f"/api/loop/pipelines/{p['rule_id']}/benchmarks", {"benchmark_id": st["benchmark_id"]})
        log(uc, "benchmark attached")
    pid = st["pipeline_id"]
    train = rows(uc, "train")

    def feed(n):
        batch = train[st["fed"]:st["fed"] + n]
        for i in range(0, len(batch), 150):
            chunk = batch[i:i + 150]
            body = {"rows": [{"messages": convo(system, r)} for r in chunk],
                    "row_ids": [f"{tag}-train-{r['id']}" for r in chunk], "default_feedback": "good"}
            for attempt in range(8):
                try:
                    import fcntl
                    with open(ROOT / "work" / ".import.lock", "w") as lk:
                        fcntl.flock(lk, fcntl.LOCK_EX)
                        out = rb.post(f"/api/loop/pipelines/{pid}/import", body)
                    break
                except (rb.ApiError, OSError) as e:
                    log(uc, f"import chunk failed ({str(e)[:120]}); retrying")
                    time.sleep(20 * (attempt + 1))
            else:
                raise RuntimeError("import kept failing")
            log(uc, f"fed {len(chunk)}: imported={out.get('imported')} present={out.get('already_present')} "
                    f"reviewed={out.get('reviewed')} refused={out.get('refused')}")
        st["fed"] += len(batch)
        save()
        return len(batch)

    while True:
        p = rb.get(f"/api/loop/pipelines/{pid}")["pipeline"]
        if p.get("status") == "paused" and p.get("paused_reason") == "model_not_trainable":
            try:
                rb.call("PUT", f"/api/loop/pipelines/{pid}", {"enabled": True})
                log(uc, "pipeline switched back on")
            except rb.ApiError as e:
                log(uc, f"switch back on refused: {str(e)[:200]}")
                time.sleep(120)
            continue
        made = int(p.get("versions_made") or 0)
        attempts = len(st["decisions"])
        if made >= a.versions or attempts >= len(BATCHES) or st["fed"] >= len(train):
            log(uc, f"done: {made} versions, {attempts} attempts")
            return
        runs = rb.get(f"/api/loop/training-runs?rule_id={pid}&limit=50").get("runs") or []
        open_runs = [r for r in runs if r.get("state") not in TERMINAL]
        if not open_runs:
            want = sum(BATCHES[:attempts + 1])
            if st["fed"] < want:
                feed(want - st["fed"])
            while True:
                try:
                    rb.post(f"/api/loop/pipelines/{pid}/run", {})
                    log(uc, f"attempt {attempts + 1} started (fed {st['fed']} rows)")
                    break
                except rb.ApiError as e:
                    if e.code == "RUN_ACTIVE":
                        break
                    if e.code == "NOT_ENOUGH_SAMPLES":
                        err = e.payload.get("error", e.payload)
                        short = int(err.get("need", 0) or 0) - int(err.get("have", 0) or 0)
                        log(uc, f"not enough samples ({json.dumps(err)[:200]}); topping up {max(short, 50) + 20}")
                        if not feed(max(short, 50) + 20):
                            log(uc, "training pool exhausted")
                            return
                        continue
                    raise
            time.sleep(a.poll)
            continue
        r = rb.get(f"/api/loop/training-runs/{open_runs[0]['id']}")["run"]
        state = r.get("state")
        key = f"{r['id']}:{state}"
        if st.get("last_state") != key:
            st["last_state"] = key
            save()
            log(uc, f"run seq={r.get('seq')} state={state} verdict={r.get('verdict')} reason={(r.get('last_reason') or '')[:160]}")
        if state in DECIDABLE:
            hist = rb.get(f"/api/loop/benchmarks/{st['benchmark_id']}/history?limit=50").get("points") or []
            pt = next((h for h in hist if h.get("run_seq") == r.get("seq")), None)
            if not pt or pt.get("status") != "done":
                log(uc, f"waiting for the benchmark replay of run {r.get('seq')}: {pt and pt.get('status')}")
                time.sleep(a.poll)
                continue
            cand, inc = pt.get("candidate_score"), pt.get("incumbent_score")
            better = cand is not None and inc is not None and cand > inc
            dec = {"seq": r.get("seq"), "run_id": r["id"], "fed": st["fed"], "candidate": cand, "incumbent": inc,
                   "loop_verdict": r.get("verdict"), "decision": "promote" if better else "reject", "at": time.time()}
            if better:
                rb.post(f"/api/loop/training-runs/{r['id']}/promote",
                        {"expected_revision": p.get("revision"), "force": r.get("verdict") != "better"})
            else:
                try:
                    rb.post(f"/api/loop/training-runs/{r['id']}/reject", {"reason": f"exact benchmark {cand} not above live {inc}"})
                except rb.ApiError as e:
                    log(uc, f"reject call failed: {e}")
            st["decisions"].append(dec)
            save()
            if better:
                import subprocess
                subprocess.Popen([sys.executable, str(ROOT / "eval_version.py"), uc, str((p.get("live_version") or 0) + 1)],
                                 stdout=open(ROOT / "logs" / f"evalv_{uc}_{r.get('seq')}.log", "a"), stderr=subprocess.STDOUT)
            log(uc, f"DECISION run {r.get('seq')}: benchmark candidate={cand} live={inc} loop_verdict={r.get('verdict')} -> {dec['decision']}")
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
