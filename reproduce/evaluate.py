#!/usr/bin/env python3
"""Ask a model the released test or validation cases of a task and score it by exact match.

Examples
  python reproduce/evaluate.py --task contract-clause-classification --model anthropic/claude-opus-5-5
  python reproduce/evaluate.py --task contract-clause-classification --model <served name of your deployment> --instruction short

The scoring rule is the paper's: an answer is correct only if it equals the reference label after lower-casing and
collapsing whitespace. Results go to reproduced/<task>/<model>.<split>.jsonl together with a summary.
"""
import argparse, json, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runbios_client as rb

ROOT = Path(__file__).resolve().parent.parent


def norm(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", choices=["test", "validation"], default="test")
    ap.add_argument("--instruction", choices=["list", "short"], default="list",
                    help="list: instruction with the full label list (frontier baselines). short: one-sentence instruction")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N cases (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    task = json.load(open(ROOT / "data" / a.task / "task.json"))
    system = task["instruction_with_label_list"] if a.instruction == "list" else task["instruction_one_sentence"]
    cases = [json.loads(l) for l in open(ROOT / "data" / a.task / f"{a.split}.jsonl")]
    if a.limit:
        cases = cases[:a.limit]
    out_dir = ROOT / "reproduced" / a.task
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{a.model.replace('/', '_')}.{a.split}.jsonl"
    done = {json.loads(l)["id"] for l in open(out_path)} if out_path.exists() else set()
    lock = threading.Lock()
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if "/" not in a.model else {}

    def one(c):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": c["text"]}]
        try:
            out, secs = rb.chat(a.model, msgs, **extra)
            ans = (out["choices"][0]["message"].get("content") or "").strip()
            u = out.get("usage") or {}
            row = {"id": c["id"], "label": c["label"], "answer": ans, "exact": norm(ans) == norm(c["label"]),
                   "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                   "billed_cost_usd": u.get("cost"), "seconds": round(secs, 3)}
        except Exception as e:
            row = {"id": c["id"], "label": c["label"], "answer": None, "exact": None, "error": str(e)[:300]}
        with lock, open(out_path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, [c for c in cases if c["id"] not in done]))
    rows = [json.loads(l) for l in open(out_path)]
    ok = [r for r in rows if r.get("answer") is not None]
    acc = sum(r["exact"] for r in ok) / len(ok) if ok else 0
    cost = sum(r.get("billed_cost_usd") or 0 for r in ok)
    summary = {"task": a.task, "model": a.model, "split": a.split, "instruction": a.instruction, "answered": len(ok),
               "errors": len(rows) - len(ok), "exact_match": round(acc, 4), "billed_total_usd": round(cost, 4),
               "usd_per_1000": round(cost / len(ok) * 1000, 4) if ok else None, "seconds": round(time.time() - t0, 1)}
    (out_dir / f"{a.model.replace('/', '_')}.{a.split}.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
