import requests, rb, json, sys
uc, part = sys.argv[1], sys.argv[2]
ws = rb.get("/api/api-keys/introspect")["workspace"]["id"]
with open(f"work/{uc}/sft5k_{part}.jsonl", "rb") as f:
    r = requests.post(rb.BASE + "/api/datasets/upload", headers={"X-API-Key": rb.key()},
                      files={"file": (f"{uc}_train_{part}.jsonl", f, "application/jsonl")},
                      data={"name": f"Case study - {uc} train {part} (real)", "workspace_id": ws, "format": "jsonl"}, timeout=1800)
print(uc, part, r.status_code, r.text[:200], flush=True)
j = r.json()
if r.status_code == 200 and j.get("id"):
    json.dump(j, open(f"work/{uc}/dataset_{part}.json", "w"))
