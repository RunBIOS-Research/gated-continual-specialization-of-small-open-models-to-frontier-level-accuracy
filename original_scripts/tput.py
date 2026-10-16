#!/usr/bin/env python3
"""Measure a deployed fine-tuned model's throughput on its own, so the cost per
1,000 answers is not distorted by other tests running at the same time.

  python tput.py <uc> [ver]

Sends the 200 benchmark records at 8 concurrent requests through RunBios and
writes serving.throughput_rps (and the measurement) into work/<uc>/ft<ver>_state.json.
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import rb

ROOT = Path(__file__).resolve().parent
uc = sys.argv[1]
ver = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] not in ("-", "1") else ""
VER = ver
d = ROOT / "work" / uc
sp = d / f"ft{ver}_state.json"
st = json.load(open(sp))
meta = json.load(open(d / "meta.json"))
rows = [json.loads(l) for l in open(d / "bench.jsonl")] + [json.loads(l) for l in open(d / "test_ext.jsonl")][:100]
CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 8
dep = rb.get(f"/api/inference/{st['deployment_id']}")
dep = dep.get("deployment", dep)


def one(r):
    out, secs = rb.chat(st["model"], [{"role": "system", "content": meta[os.environ.get("CS_SYSTEM", "system_prompt")]}, {"role": "user", "content": r["text"]}],
                        **({"chat_template_kwargs": {"enable_thinking": False}} if VER.startswith("q") else {}))
    return secs


t0 = time.time()
with ThreadPoolExecutor(CONC) as ex:
    lat = list(ex.map(one, rows))
wall = time.time() - t0
st["serving"].update(hourly_cents=dep.get("cost_per_hour_cents"), throughput_rps=round(len(rows) / wall, 3),
                     throughput_note=f"isolated run: {len(rows)} requests, {CONC} concurrent, {wall:.1f}s, {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
                     latency_p50_isolated=sorted(lat)[len(lat) // 2], concurrency=CONC)
json.dump(st, open(sp, "w"), indent=1)
print(uc, ver, st["serving"])
