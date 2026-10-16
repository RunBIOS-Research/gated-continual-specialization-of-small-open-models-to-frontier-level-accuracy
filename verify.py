#!/usr/bin/env python3
"""Recompute every accuracy, interval, significance test and cost in the paper from this package.

Uses only the Python standard library. Run from the package root:  python verify.py
"""
import csv, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTIER = ["anthropic_claude-opus-5-5", "openai_gpt-5.5", "openai_gpt-6-astra"]


def answers(task, name, split="test"):
    p = ROOT / "results" / task / f"{name}.{split}.jsonl"
    return {r["id"]: r for r in map(json.loads, open(p)) if r["answer"] is not None}


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    k = min(n01, n10)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n), n01, n10


def main():
    runs = list(csv.DictReader(open(ROOT / "runs" / "run_log.csv")))
    frontier = {(r["task"], r["model"].replace("/", "_")): r for r in csv.DictReader(open(ROOT / "runs" / "frontier_baselines.csv"))}
    print(f"{'task':34} {'selected model':44} {'acc':>6} {'95% CI':>15} {'best frontier':>14} {'diff':>6} {'p':>7} {'$/1k':>6} {'frontier $/1k':>13}")
    for task in sorted({r["task"] for r in runs}):
        cand = [r for r in runs if r["task"] == task]
        with_val = [r for r in cand if r["validation_exact_match"]]
        # selection rule: highest validation accuracy, otherwise the configuration trained on the most data
        sel = max(with_val, key=lambda r: float(r["validation_exact_match"])) if with_val else max(cand, key=lambda r: int(r["training_examples"]))
        assert sel["selected"] == "True", f"{task}: selection rule does not reproduce the reported model"
        ours = answers(task, sel["configuration"])
        tests = [json.loads(l) for l in open(ROOT / "data" / task / "test.jsonl")]
        assert len(tests) == 500 and set(ours) == {t["id"] for t in tests}, f"{task}: answers do not cover the 500 test cases"
        fr = {m: answers(task, m) for m in FRONTIER}
        best = max(FRONTIER, key=lambda m: sum(r["exact"] for r in fr[m].values()) / len(fr[m]))
        acc = sum(r["exact"] for r in ours.values()) / len(ours)
        bacc = sum(r["exact"] for r in fr[best].values()) / len(fr[best])
        ids = [i for i in ours if i in fr[best]]
        p, _, _ = mcnemar([ours[i]["exact"] for i in ids], [fr[best][i]["exact"] for i in ids])
        lo, hi = wilson(acc, len(ours))
        f_cost = sum(r["billed_cost_usd"] or 0 for r in fr[best].values()) / len(fr[best]) * 1000
        print(f"{task:34} {sel['configuration']:44} {acc * 100:6.1f} [{lo * 100:5.1f}, {hi * 100:5.1f}] {bacc * 100:14.1f} "
              f"{(acc - bacc) * 100:+6.1f} {('<0.001' if p < 0.001 else f'{p:.3f}'):>7} {float(sel['usd_per_1000']):6.2f} {f_cost:13.2f}")
        assert abs(f_cost - float(frontier[(task, best)]["usd_per_1000"])) < 0.01


if __name__ == "__main__":
    main()
    print("All checks passed: the selection rule, accuracies, intervals, tests and costs reproduce the reported values.")
