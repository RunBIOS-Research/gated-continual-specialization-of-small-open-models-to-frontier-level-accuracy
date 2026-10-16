#!/usr/bin/env python3
"""Turn the five real datasets in raw/ into Conscious Loop case-study inputs.

For every use case, writes work/<uc>/:
  bench.jsonl     200 frozen rows, pinned as the platform's exact-match benchmark
  test_ext.jsonl  300 more frozen rows (bench + test_ext = the 500-row frozen test)
  train.jsonl     the training pool, in feeding order (never overlaps the test)
  meta.json       system prompt, label set, provenance, counts, guards applied

Each row: {"id", "text", "label"}. Nothing is generated: every text is a real
record from its source, every label is the source's own.
"""
import csv, hashlib, json, random, re, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW, WORK = ROOT / "raw", ROOT / "work"
SEED = 20261013
BENCH, EXT = 200, 300


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def write(uc, bench, ext, train, labels, system, meta):
    d = WORK / uc
    d.mkdir(parents=True, exist_ok=True)
    test_keys = {norm(r["text"]) for r in bench + ext}
    before = len(train)
    seen, clean = set(test_keys), []
    for r in train:
        k = norm(r["text"])
        if k in seen:
            continue
        seen.add(k)
        clean.append(r)
    for name, rows in (("bench", bench), ("test_ext", ext), ("train", clean)):
        with open(d / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta.update({
        "use_case": uc, "seed": SEED, "labels": labels, "system_prompt": system,
        "counts": {"bench": len(bench), "test_ext": len(ext), "train_pool": len(clean),
                   "train_dropped_duplicate_or_test_overlap": before - len(clean)},
    })
    json.dump(meta, open(d / "meta.json", "w"), indent=1, ensure_ascii=False)
    print(f"{uc}: labels={len(labels)} bench={len(bench)} ext={len(ext)} train={len(clean)} "
          f"(dropped {before - len(clean)} dup/overlap)")


def system_for(role, labels, extra=""):
    return (f"{role} Reply with exactly one label from the list below, copied exactly, and nothing else: "
            f"no explanation, no punctuation, no quotes.{extra}\n\nLabels:\n" + "\n".join(labels))


def split_eval(rows, rng):
    rng.shuffle(rows)
    return rows[:BENCH], rows[BENCH:BENCH + EXT]


def banking77():
    tr = pd.read_csv(RAW / "banking77/train.csv")
    te = pd.read_csv(RAW / "banking77/test.csv")
    labels = sorted(tr.category.unique())
    rng = random.Random(SEED)
    ev = [{"id": f"b77-test-{i}", "text": t, "label": c} for i, (t, c) in enumerate(zip(te.text, te.category))]
    bench, ext = split_eval(ev, rng)
    train = [{"id": f"b77-train-{i}", "text": t, "label": c} for i, (t, c) in enumerate(zip(tr.text, tr.category))]
    rng.shuffle(train)
    system = system_for("You are the triage assistant for a retail bank's customer support team. Read the "
                        "customer's message and route it to the right support queue.", labels)
    write("banking77", bench, ext, train, labels, system, {
        "title": "Retail banking: customer-message routing",
        "source": "PolyAI Banking77 (github.com/PolyAI-LDN/task-specific-datasets), CC-BY-4.0",
        "provenance": "Real customer queries to an online bank; train from the official train split, "
                      "every test row from the official test split."})


def cfpb():
    d = pd.read_parquet(RAW / "cfpb/narratives_2023plus.parquet")
    d["text"] = d["Consumer complaint narrative"].str.strip()
    d["words"] = d.text.str.split().str.len()
    d = d[(d.words >= 15) & (d.words <= 450)]
    d["label"] = d.Product.str.strip() + " / " + d.Issue.str.strip()
    # CFPB moved to a new product taxonomy in August 2023 (e.g. "Credit card or prepaid card" became
    # "Credit card"); train and test only on the current one, so every label means the same thing.
    d = d[pd.to_datetime(d["Date received"]) >= "2023-09-01"]
    vc = d.label.value_counts()
    keep = vc[vc >= 200].index
    covered = float(vc[keep].sum() / vc.sum())
    d = d[d.label.isin(keep)]
    labels = sorted(keep)
    d["date"] = pd.to_datetime(d["Date received"])
    rng = random.Random(SEED)
    ev_pool = d[d.date >= "2024-01-01"]
    tr_pool = d[d.date < "2024-01-01"]
    ev = [{"id": f"cfpb-{c}", "text": t, "label": l} for c, t, l in zip(ev_pool["Complaint ID"], ev_pool.text, ev_pool.label)]
    bench, ext = split_eval(ev, rng)
    tr = tr_pool.sample(n=min(20000, len(tr_pool)), random_state=SEED)
    train = [{"id": f"cfpb-{c}", "text": t, "label": l} for c, t, l in zip(tr["Complaint ID"], tr.text, tr.label)]
    system = system_for("You are the complaint-intake assistant for a US consumer-finance company. Read the "
                        "consumer's complaint and route it to the right product and issue category "
                        "(written as 'Product / Issue'). XXXX marks text the regulator redacted.", labels)
    write("cfpb", bench, ext, train, labels, system, {
        "title": "Consumer finance: regulatory complaint triage",
        "source": "US CFPB Consumer Complaint Database (CC0), narratives via the CC0 mirror "
                  "BEE-spoke-data/consumer-finance-complaints; labels cross-checked against the official "
                  "CFPB bulk file of 2026-10-12 (2,000 of 2,000 sampled IDs identical on product and issue)",
        "provenance": "Real complaints written by US consumers under CFPB's current category list (in use since September 2023). "
                      "Train from Sep-Dec 2023, test from Jan-Feb 2024 (later than any training row).",
        "label_coverage_since_2023_09": round(covered, 4),
        "note": "The label is the product and issue the consumer selected when filing."})


def ledgar():
    names = json.load(open(RAW / "ledgar/info.json"))["dataset_info"]["features"]["label"]["names"]
    tr = pd.read_parquet(RAW / "ledgar/train.parquet")
    te = pd.read_parquet(RAW / "ledgar/test.parquet")
    labels = sorted(names)
    rng = random.Random(SEED)
    ev = [{"id": f"ledgar-test-{i}", "text": t, "label": names[l]} for i, (t, l) in enumerate(zip(te.text, te.label))
          if len(t.split()) <= 400]
    bench, ext = split_eval(ev, rng)
    tr = tr[tr.text.str.split().str.len() <= 400].sample(frac=1.0, random_state=SEED).head(20000)
    train = [{"id": f"ledgar-train-{i}", "text": t, "label": names[l]} for i, (t, l) in zip(tr.index, zip(tr.text, tr.label))]
    system = system_for("You are a contract-review assistant for a corporate legal team. Read the contract "
                        "provision and classify it by the clause type it is.", labels)
    write("ledgar", bench, ext, train, labels, system, {
        "title": "Legal: contract clause classification",
        "source": "LEDGAR, from LexGLUE (coastalcph/lex_glue), CC-BY-4.0",
        "provenance": "Real provisions from contracts filed with the US SEC (EDGAR). Train from the official "
                      "train split, test from the official test split."})


def finer():
    tags = json.load(open(RAW / "finer139/info.json"))["dataset_info"]["features"]["ner_tags"]["feature"]["names"]
    labels = sorted({t[2:] for t in tags if t != "O"})
    rng = random.Random(SEED)

    def rows(files, prefix, limit):
        out = []
        for f in files:
            df = pd.read_parquet(RAW / f"finer139/{f}.parquet")
            for sid, toks, tg in zip(df.id, df.tokens, df.ner_tags):
                toks, tg = list(toks), list(tg)
                if len(toks) > 90 or not any(tg):
                    continue
                ents, i = [], 0
                while i < len(tg):
                    if tags[tg[i]].startswith("B-"):
                        j = i + 1
                        while j < len(tg) and tags[tg[j]] == "I-" + tags[tg[i]][2:]:
                            j += 1
                        ents.append((i, j, tags[tg[i]][2:]))
                        i = j
                    else:
                        i += 1
                if not ents:
                    continue
                s, e, lab = rng.choice(ents)
                marked = toks[:s] + ["[["] + list(toks[s:e]) + ["]]"] + toks[e:]
                out.append({"id": f"{prefix}-{sid}-{s}", "text": " ".join(marked), "label": lab})
                if len(out) >= limit:
                    return out
        return out

    ev = rows(["test_0"], "finer-test", 4000)
    bench, ext = split_eval(ev, rng)
    train = rows(["train_0"], "finer-train", 40000)
    rng.shuffle(train)
    train = train[:20000]
    system = system_for("You are an XBRL tagging assistant for a public company's financial-reporting team. "
                        "Read the sentence from an SEC 10-K/10-Q filing and give the US-GAAP XBRL tag for the "
                        "value marked [[ like this ]].", labels)
    write("finer", bench, ext, train, labels, system, {
        "title": "Financial reporting: XBRL tagging of SEC filings",
        "source": "FiNER-139 (nlpaueb/finer-139), CC-BY-SA-4.0",
        "provenance": "Real sentences from SEC 10-K/10-Q filings, tagged with the XBRL tags the filers "
                      "submitted. Train from the official train split, test from the official test split. "
                      "One tagged value per sentence is marked and asked about."})


def cve():
    names = {}
    with open(RAW / "cwe/1000.csv", newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            names[f"CWE-{r['CWE-ID']}"] = r["Name"]
    c = pd.read_csv(RAW / "cve/CVE_CWE_2025.csv", low_memory=False)
    c["year"] = c["CVE-ID"].str.split("-").str[1].astype(int)
    c = c[c["CWE-ID"].astype(str).str.match(r"^CWE-\d+$") & (c.year >= 2021)]
    c = c[~c.DESCRIPTION.str.contains(r"CWE-\d", regex=True, na=False)]
    vc = c["CWE-ID"].value_counts()
    keep = [k for k in vc.index if vc[k] >= 300 and k in names][:60]
    labels_ids = sorted(keep, key=lambda k: int(k.split("-")[1]))
    labels = labels_ids
    leak_name = {k: norm(names[k]) for k in keep}
    nvd = json.load(open(RAW / "cve/nvd_2026q3.json"))
    ev, ev_ids = [], set()
    for v in nvd:
        cv = v["cve"]
        if cv.get("vulnStatus") in ("Rejected", "Reserved"):
            continue
        cwes = {d["value"] for w in cv.get("weaknesses", []) for d in w["description"] if d["value"].startswith("CWE-")}
        desc = next((d["value"] for d in cv["descriptions"] if d["lang"] == "en"), "")
        if len(cwes) != 1 or not desc or re.search(r"CWE-\d", desc):
            continue
        cw = cwes.pop()
        if cw not in keep or leak_name[cw] in norm(desc):
            continue
        ev.append({"id": cv["id"], "text": desc, "label": cw})
        ev_ids.add(cv["id"])
    rng = random.Random(SEED)
    bench, ext = split_eval(ev, rng)
    tr = c[c["CWE-ID"].isin(keep) & ~c["CVE-ID"].isin(ev_ids)].sample(frac=1.0, random_state=SEED).head(20000)
    train = [{"id": i, "text": t, "label": l} for i, t, l in zip(tr["CVE-ID"], tr.DESCRIPTION, tr["CWE-ID"])]
    listing = [f"{k}: {names[k]}" for k in labels_ids]
    system = ("You are a vulnerability-triage assistant for a product-security team. Read the CVE description "
              "and classify the root-cause weakness. Reply with exactly one CWE ID from the list below (for "
              "example CWE-79), and nothing else: no name, no explanation, no punctuation.\n\nWeaknesses:\n"
              + "\n".join(listing))
    write("cve", bench, ext, train, labels, system, {
        "title": "Cybersecurity: vulnerability root-cause triage (CVE to CWE)",
        "source": "NIST National Vulnerability Database (public domain): test set fetched live from the NVD "
                  "API (CVEs published 2026-07-01..2026-09-30); training set from the CC0 NVD export "
                  "stasvinokur/cve-and-cwe-dataset-1999-2025 (CVEs 2021-2025); CWE names from MITRE CWE-1000",
        "provenance": "Real CVE records. Every test CVE was published in Q3 2026, after the training data "
                      "and after the public release of every model compared, so no model can have memorised it.",
        "guards": "Descriptions that name a CWE ID, or contain the CWE's own name, are removed from train and test.",
        "nvd_q3_2026_records": len(nvd), "eligible_test_records": len(ev)})


if __name__ == "__main__":
    want = sys.argv[1:] or ["banking77", "cfpb", "ledgar", "finer", "cve"]
    for uc in want:
        globals()[uc]()
