#!/usr/bin/env python3
"""Score one model on one use case's frozen test set, through RunBios.

  python evaluate.py <uc> <model> [--label NAME] [--workers 8] [--set bench|all]

Every call is POST /v1/chat/completions on RunBios with the use case's system
prompt and the real record. The answer is checked with the platform's own exact
rule (lower case, spacing collapsed: loop-service reference_match.go
normalisedAnswer), and also leniently (surrounding quotes, backticks, a trailing
period or a 'Label:' prefix forgiven) so no model loses on formatting alone.
Cost is the cost RunBios billed for each call (usage.cost); latency is measured
wall-clock per call. Results are resumable: work/<uc>/results/<label>.jsonl.
"""
import argparse, json, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import rb

ROOT = Path(__file__).resolve().parent


def strict(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def lenient(s):
    s = strict(s)
    s = re.sub(r"^(label|answer|queue|category|tag|cwe id|weakness)\s*:\s*", "", s)
    s = s.strip().strip("`'\"*").strip().rstrip(".").strip().strip("`'\"*").strip()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("uc")
    ap.add_argument("model")
    ap.add_argument("--label")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--set", default="all", choices=["bench", "all", "val"])
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--no-think", action="store_true")
    a = ap.parse_args()
    d = ROOT / "work" / a.uc
    meta = json.load(open(d / "meta.json"))
    rows = [json.loads(l) for l in open(d / "bench.jsonl")] if a.set != "val" else [json.loads(l) for l in open(d / "val.jsonl")]
    if a.set == "all":
        rows += [json.loads(l) for l in open(d / "test_ext.jsonl")]
    label = (a.label or a.model.replace("/", "_")) + ("_val" if a.set == "val" else "")
    out_dir = d / "results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{label}.jsonl"
    done = set()
    if path.exists():
        for l in open(path):
            r = json.loads(l)
            if r.get("answer") is not None or r.get("error"):
                done.add(r["id"])
    todo = [r for r in rows if r["id"] not in done]
    lock = threading.Lock()
    labels = {strict(x) for x in meta["labels"]}
    print(f"[{a.uc}/{label}] {len(todo)} to score ({len(done)} already)", flush=True)

    def one(r):
        msgs = [{"role": "system", "content": meta[os.environ.get("CS_SYSTEM", "system_prompt")]}, {"role": "user", "content": r["text"]}]
        rec = {"id": r["id"], "label": r["label"], "model": a.model, "in_bench": None}
        try:
            out, secs = rb.chat(a.model, msgs, max_tokens=a.max_tokens or None,
                                **({"chat_template_kwargs": {"enable_thinking": False}} if a.no_think else {}))
            ans = (out["choices"][0]["message"].get("content") or "").strip()
            u = out.get("usage") or {}
            rec.update(answer=ans, secs=round(secs, 3), prompt_tokens=u.get("prompt_tokens"),
                       completion_tokens=u.get("completion_tokens"), cost=u.get("cost"),
                       exact=strict(ans) == strict(r["label"]), lenient=lenient(ans) == strict(r["label"]),
                       in_label_set=lenient(ans) in labels)
        except Exception as e:
            rec.update(answer=None, error=str(e)[:300])
        with lock:
            with open(path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, todo))
    res = [json.loads(l) for l in open(path)]
    ok = [r for r in res if r.get("answer") is not None]
    bench_ids = {json.loads(l)["id"] for l in open(d / "bench.jsonl")}
    acc = lambda xs, k: round(sum(r[k] for r in xs) / len(xs), 4) if xs else None
    b = [r for r in ok if r["id"] in bench_ids]
    print(f"[{a.uc}/{label}] scored {len(ok)}/{len(res)} errors={len(res) - len(ok)} "
          f"exact={acc(ok, 'exact')} lenient={acc(ok, 'lenient')} bench200_exact={acc(b, 'exact')} "
          f"cost=${sum(r.get('cost') or 0 for r in ok):.3f} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
