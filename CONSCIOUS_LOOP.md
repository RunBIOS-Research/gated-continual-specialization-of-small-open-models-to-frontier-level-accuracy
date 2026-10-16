# Running your own Conscious Loop

The Conscious Loop is the part of the RunBios platform that turns labeled examples into successive versions of a fine-tuned
model. It runs on RunBios itself. You do not install or host anything: you create a pipeline, give it labeled examples, and
the platform trains, tests and deploys each version. You can drive it from the RunBios console or, as in this repository,
from the public API.

This guide explains the concepts, the two ways to give the Loop data, and every API call that
`reproduce/conscious_loop.py` makes. Section 2 of the paper describes the method in detail.

## Requirements

- A RunBios account with access to training, deployments and the Conscious Loop.
- An API key with the scopes `loop:read`, `loop:write`, `serverless`, `deployments:read` and `training:read`.
- Credit in the workspace wallet. The Loop bills training, the machines used to compare versions, and the deployment that
  serves the live version.

```
export RUNBIOS_API_KEY=<your key>
```

The key stays on your machine. Scripts read it from the environment and never write it anywhere.

## Concepts

| Term | Meaning |
| --- | --- |
| Pipeline | One task the Loop improves: a base model, the examples it learns from, its benchmark and its promotion rule |
| Sample | One conversation the Loop can learn from: instruction, input and the right answer |
| Feedback | What makes a sample usable for training. An answer marked good is learned as it is. A correction is learned in its place |
| Held-back set | Samples set aside by the pipeline (5 percent by default) to compare a new version with the live one |
| Benchmark | A fixed set of 20 to 200 conversations with right answers, pinned once and replayed for every version |
| Attempt | One training run: build the set, train, deploy beside the live version, compare, decide |
| Live version | The version that answers requests sent to the pipeline's serving name (`loop-<tag>`) |
| Promotion | Replacing the live version with a new one. Manual (a person decides) or automatic under the pipeline's policy |

## Two ways to give the Loop data

**Import labeled examples.** This is what the paper did, and what `reproduce/conscious_loop.py` does. Each row is a
conversation whose last turn is the right answer, sent with `default_feedback: "good"` so the Loop learns it as a correct
example.

**Capture production traffic.** Send your application's requests to a model through RunBios with the header
`X-Loop-Pipeline: <pipeline tag>`. Each exchange is recorded as a sample for that pipeline. Your team then marks answers as
good or gives a correction, in the console or through the API. The Loop trains on the samples that have feedback. Sending
`X-Loop-Capture: off` on a call records nothing.

Both routes produce the same kind of sample, and a pipeline can use both.

## The cycle, step by step (API)

All paths are relative to `https://api.runbios.ai`. Platform calls (`/api/...`) carry the key in the `X-API-Key` header and
model calls (`/v1/...`) carry it as `Authorization: Bearer <key>`.

**1. Pin a benchmark.** Import the benchmark conversations with their right answers, then pin them.

```
POST /api/loop/import
{"workspace_id": "<from GET /api/api-keys/introspect>", "source": "my-benchmark",
 "rows": [{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."},
                        {"role": "assistant", "content": "<right answer>"}], "label": true}],
 "row_ids": ["bench-1"]}
-> {"trace_ids": [...]}

POST /api/loop/benchmarks
{"name": "My benchmark", "source": {"kind": "traces", "trace_ids": [...]}, "check": "exact"}
-> {"benchmark": {"id": "..."}}
```

`check: "exact"` scores an answer as right only when it matches the right answer, with no model involved.

**2. Create the pipeline and attach the benchmark.**

```
POST /api/loop/pipelines
{"name": "My pipeline", "use_case": "What the model does", "model": {"id": "meta-llama/Llama-3.1-8B-Instruct"},
 "train_when": {"schedule": null}, "promotion": {"policy": "manual"}, "max_versions": 5, "enabled": true}
-> {"pipeline": {"rule_id": "...", "serving_name": "loop-<tag>"}}

POST /api/loop/pipelines/<rule_id>/benchmarks
{"benchmark_id": "..."}
```

`GET /api/loop/models` lists the base models the Loop can train.

**3. Feed labeled examples.** Send them in batches of up to a few hundred rows.

```
POST /api/loop/pipelines/<rule_id>/import
{"rows": [{"messages": [...]}], "row_ids": ["train-1"], "default_feedback": "good"}
```

**4. Train.** Ask for an attempt now, or let the pipeline's schedule start one.

```
POST /api/loop/pipelines/<rule_id>/run
```

**5. Follow the attempt.** It moves through dataset building, training, deployment of the new version, the held-back
comparison and the benchmark replay.

```
GET /api/loop/training-runs?rule_id=<rule_id>
GET /api/loop/training-runs/<run_id>          (state, verdict, training summary, scores)
GET /api/loop/benchmarks/<benchmark_id>/history?pipeline=<rule_id>
```

**6. Decide.** Under manual promotion an attempt waits in `awaiting_review`. Promote or reject it.

```
POST /api/loop/training-runs/<run_id>/promote   {"expected_revision": <pipeline revision>, "force": false}
POST /api/loop/training-runs/<run_id>/reject    {}
```

`force: true` promotes a version whose held-back comparison did not find it better. In Study B of the paper a version was
promoted when its benchmark score exceeded the live version's, which sometimes required `force`. Run the script with
`--rule loop` to promote only versions the held-back comparison also found better.

**7. Use the live version.** Call the pipeline's serving name like any model.

```
POST /v1/chat/completions
{"model": "loop-<tag>", "messages": [...]}
```

Repeat steps 3 to 6 as new examples arrive. The pipeline also starts attempts by itself whenever it holds enough new
samples, for example right after a version goes live. A later attempt usually builds on the live version: it trains on the
samples that version has not learned and replays part of what it learned, so it does not forget it. Each run record
(`GET /api/loop/training-runs/<run_id>`) states exactly what it trained on in `lineage` and `training_summary`. Every attempt, score and decision stays visible in the RunBios console under the
pipeline.

## Reproducing Study B

```
python reproduce/conscious_loop.py --task banking-intent-routing --model meta-llama/Llama-3.1-8B-Instruct
```

The paper's Study B used Llama-3.1-8B for banking and complaints, Qwen3.6-27B for contracts and vulnerabilities, and
Qwen3.5-9B for XBRL tagging, with batches of 4,000, then 6,000, then 8,000 rows. The complete records of the original
pipelines are in `logs/conscious_loop/`. A reproduction run made on the public API is in `reproduction/`.

## Costs and cleanup

Each attempt is billed for its training job, for the time the comparison machines run and for any benchmark scoring. Once
a version is live, its deployment is billed by the hour until you pause or delete the pipeline in the RunBios console or
with `PUT /api/loop/pipelines/<rule_id>` and `{"enabled": false}`.
