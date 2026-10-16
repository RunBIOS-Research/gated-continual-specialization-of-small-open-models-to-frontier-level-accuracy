#!/usr/bin/env python3
"""Run a Conscious Loop pipeline on RunBios for one task of the paper (Study B), version after version.

Example
  python reproduce/conscious_loop.py --task banking-intent-routing --model meta-llama/Llama-3.1-8B-Instruct

What it does, using only the public RunBios API:
  1. imports the 200 benchmark cases of the task (data/<task>/test.jsonl, in_benchmark_200) with their right answers and pins
     them as an exact-match benchmark
  2. creates a pipeline for the base model, with manual promotion, and attaches the benchmark
  3. for each attempt: feeds the next batch of labeled training rows to the pipeline as good examples, asks the pipeline to
     train, waits while the Loop trains a version, compares it with the live version on held-back rows and replays the
     benchmark, then promotes or rejects it (see --rule). The pipeline also starts attempts by itself when it holds enough
     new samples, for example right after a version goes live, and a later attempt usually builds on the live version.
     The rows each attempt actually trained on are recorded from the platform's own run record
  4. records every attempt and decision in reproduced/<task>/conscious_loop_state.json, so a restart resumes where it stopped

The Loop bills training, the comparison machines and the live version's deployment to your account. Each live version keeps
a dedicated deployment running until you pause or delete the pipeline in the RunBios console.
"""
import argparse, gzip, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runbios_client as rb

ROOT = Path(__file__).resolve().parent.parent
TERMINAL = {"promoted", "rejected", "failed", "budget_stopped", "expired", "cancelled", "superseded", "rolled_back"}
DECIDABLE = {"reported", "awaiting_review"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--batches", default="4000,6000,8000", help="cumulative training rows fed before each attempt, as increments")
    ap.add_argument("--rule", choices=["benchmark", "loop"], default="benchmark",
                    help="benchmark: promote when the new version beats the live one on the benchmark (the rule of Study B). "
                         "loop: promote only when the Loop's held-back comparison also finds it better")
    ap.add_argument("--name", default="", help="pipeline name (default: Conscious Loop reproduction: <task>)")
    ap.add_argument("--poll", type=int, default=45)
    a = ap.parse_args()

    task = json.load(open(ROOT / "data" / a.task / "task.json"))
    system = task["instruction_with_label_list"]
    test = [json.loads(l) for l in open(ROOT / "data" / a.task / "test.jsonl")]
    bench = [r for r in test if r["in_benchmark_200"]]
    train = [json.loads(l) for l in gzip.open(ROOT / "data" / a.task / "train_selected_model.jsonl.gz", "rt", encoding="utf-8")]
    batches = [int(x) for x in a.batches.split(",")]
    out_dir = ROOT / "reproduced" / a.task
    out_dir.mkdir(parents=True, exist_ok=True)
    sp = out_dir / "conscious_loop_state.json"
    st = json.load(open(sp)) if sp.exists() else {"fed": 0, "decisions": []}

    def save():
        sp.write_text(json.dumps(st, indent=1))

    def log(m):
        line = f"{time.strftime('%H:%M:%S')} {m}"
        print(line, flush=True)
        with open(out_dir / "conscious_loop.log", "a") as f:
            f.write(line + "\n")

    def convo(text, label):
        return [{"role": "system", "content": system}, {"role": "user", "content": text}, {"role": "assistant", "content": label}]

    ws = rb.get("/api/api-keys/introspect")["workspace"]["id"]
    tag = f"cl-{a.task}"[:40]

    if not st.get("benchmark_id"):
        out = rb.post("/api/loop/import", {"workspace_id": ws, "source": f"{tag}-benchmark",
                                           "rows": [{"messages": convo(r["text"], r["label"]), "label": True} for r in bench],
                                           "row_ids": [f"{tag}-bench-{r['id']}" for r in bench]})
        tids = out.get("trace_ids") or []
        if len(tids) != len(bench):
            sys.exit(f"benchmark import returned {len(tids)} of {len(bench)} conversations: {json.dumps(out)[:400]}")
        b = rb.post("/api/loop/benchmarks", {"name": f"{a.task} benchmark (exact, 200 cases) {time.strftime('%Y-%m-%d %H:%M')}",
                                             "source": {"kind": "traces", "trace_ids": tids}, "check": "exact"})["benchmark"]
        st["benchmark_id"] = b["id"]
        save()
        log(f"benchmark pinned {b['id']} ({b.get('item_count')} cases, exact match)")

    if not st.get("pipeline_id"):
        p = rb.post("/api/loop/pipelines", {"name": a.name or f"Conscious Loop reproduction: {a.task}", "use_case": task["title"],
                                            "model": {"id": a.model}, "train_when": {"schedule": None},
                                            "promotion": {"policy": "manual"}, "max_versions": len(batches), "enabled": True})["pipeline"]
        st["pipeline_id"], st["serving_name"] = p["rule_id"], p.get("serving_name")
        save()
        rb.post(f"/api/loop/pipelines/{p['rule_id']}/benchmarks", {"benchmark_id": st["benchmark_id"]})
        log(f"pipeline created {p['rule_id']} (serving name {p.get('serving_name')}), benchmark attached")
    pid = st["pipeline_id"]

    def feed(n):
        rows = train[st["fed"]:st["fed"] + n]
        for i in range(0, len(rows), 150):
            chunk = rows[i:i + 150]
            rb.post(f"/api/loop/pipelines/{pid}/import", {"rows": [{"messages": convo(r["text"], r["label"])} for r in chunk],
                                                          "row_ids": [f"{tag}-train-{st['fed'] + i + k}" for k in range(len(chunk))],
                                                          "default_feedback": "good"})
        st["fed"] += len(rows)
        save()
        log(f"fed {len(rows)} labeled rows (total {st['fed']})")
        return len(rows)

    while True:
        p = rb.get(f"/api/loop/pipelines/{pid}")["pipeline"]
        attempts = len(st["decisions"])
        if attempts >= len(batches):
            log(f"done: {attempts} attempts, {p.get('versions_made')} versions made")
            return
        runs = rb.get(f"/api/loop/training-runs?rule_id={pid}&limit=50").get("runs") or []
        open_runs = [r for r in runs if r.get("state") not in TERMINAL]
        if not open_runs:
            want = sum(batches[:attempts + 1])
            if st["fed"] < want and not feed(want - st["fed"]):
                log("training rows exhausted")
                return
            try:
                rb.post(f"/api/loop/pipelines/{pid}/run", {})
                log(f"attempt {attempts + 1} requested ({st['fed']} rows fed)")
            except rb.ApiError as e:
                if e.code != "RUN_ACTIVE":
                    log(f"run refused: {e.code} {str(e)[:200]}")
            time.sleep(a.poll)
            continue
        r = rb.get(f"/api/loop/training-runs/{open_runs[0]['id']}")["run"]
        key = f"{r['id']}:{r.get('state')}"
        if st.get("last_state") != key:
            st["last_state"] = key
            save()
            log(f"attempt {r.get('seq')}: {r.get('state')} ({(r.get('last_reason') or '')[:140]})")
        if r.get("state") in DECIDABLE:
            hist = rb.get(f"/api/loop/benchmarks/{st['benchmark_id']}/history?pipeline={pid}&limit=50").get("points") or []
            pt = next((h for h in hist if h.get("run_seq") == r.get("seq")), None)
            if not pt or pt.get("status") != "done":
                time.sleep(a.poll)
                continue
            cand, inc = pt.get("candidate_score"), pt.get("incumbent_score")
            better = cand is not None and inc is not None and cand > inc
            if a.rule == "loop":
                better = better and r.get("verdict") == "better"
            if better:
                rb.post(f"/api/loop/training-runs/{r['id']}/promote", {"expected_revision": p.get("revision"), "force": r.get("verdict") != "better"})
            else:
                rb.post(f"/api/loop/training-runs/{r['id']}/reject", {})
            lin = r.get("lineage") or {}
            st["decisions"].append({"attempt": r.get("seq"), "run_id": r["id"], "rows_fed_so_far": st["fed"],
                                    "train_rows": r.get("train_rows"), "trained_from": lin.get("trained_from"),
                                    "built_on_versions": lin.get("built_on"), "new_samples": lin.get("new_samples"),
                                    "replayed_samples": lin.get("replayed_samples"), "benchmark_new": cand,
                                    "benchmark_live": inc, "heldback_verdict": r.get("verdict"),
                                    "decision": "promote" if better else "reject", "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            save()
            log(f"attempt {r.get('seq')}: benchmark new {cand} vs live {inc}, held-back verdict {r.get('verdict')} -> "
                f"{'promoted' if better else 'rejected'}")
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
