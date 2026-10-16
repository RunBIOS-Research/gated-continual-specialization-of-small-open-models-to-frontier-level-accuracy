# Reproducing the results

There are three levels of reproduction. Level 1 needs nothing but Python. Levels 2 and 3 use the public RunBios API and
bill your own RunBios account.

## Level 1. Recompute every reported number (no account needed)

```
python verify.py
shasum -a 256 -c MANIFEST.sha256
```

`verify.py` reapplies the selection rule to `runs/run_log.csv`, then recomputes for each task the accuracy of the selected
model, its 95% Wilson interval, the strongest frontier model, the difference, the exact McNemar p value and the cost per
1,000 requests, all from the released answers in `results/`. The training curves, checkpoints and evaluations of every
training job are in `logs/training/`, and every Conscious Loop attempt with its held-back comparison and benchmark replay,
answer by answer, is in `logs/conscious_loop/`.

## Level 2. Ask the frontier models the same questions

Requirements: Python 3.10 or later and a RunBios API key with serverless access.

```
export RUNBIOS_API_KEY=<your key>
python reproduce/evaluate.py --task contract-clause-classification --model anthropic/claude-opus-5-5
python reproduce/evaluate.py --task contract-clause-classification --model openai/gpt-5.5
python reproduce/evaluate.py --task contract-clause-classification --model openai/gpt-6-astra
```

Each run asks the 500 test cases with the instruction and label list in `data/<task>/task.json`, at temperature 0, and
writes every answer and a summary to `reproduced/<task>/`. A full run of the three models on one task costs between about
$4 and $30 at current RunBios prices, depending on the task. Add `--limit 20` for a quick check.

## Level 3. Retrain and test a selected model

Requirements: a RunBios API key with training and deployment access, and credit for the training job and about one hour
of a dedicated GPU.

```
export RUNBIOS_API_KEY=<your key>
python reproduce/train.py --task banking-intent-routing
```

`train.py` reads the selected configuration of the task from `runs/run_log.csv` (base model, epochs and instruction
format) and the exact training rows from `data/<task>/train_selected_model.jsonl.gz`. It uploads the rows, creates a LoRA
training job (bf16, maximum length 2,048), deploys the best checkpoint on the cheapest available GPU, evaluates it on the
500 test cases with `reproduce/evaluate.py` and stops the deployment.

| Task | Base model | Training rows | Training estimate |
| --- | --- | --- | --- |
| banking-intent-routing | Llama-3.1-8B-Instruct | 9,968 | about $8 to $11 |
| consumer-complaint-triage | Llama-3.1-8B-Instruct | 15,000 | about $18 |
| contract-clause-classification | Qwen3.6-27B | 66,737 | about $73 |
| xbrl-numeric-tagging | Qwen3.5-9B | 124,454 | about $193 |
| vulnerability-root-cause-triage | Qwen3.6-27B | 34,906 | about $303 |

The script prints the RunBios estimate before training starts.

## What to expect

Reproduction is statistical, not bit-exact. Fine-tuning is not deterministic across GPUs and runs, and providers update
frontier models over time. A faithful reproduction should fall within the 95% intervals reported in the paper. The
`reproduction/` folder, when present, records an independent re-run made on the public RunBios API after the study.

## Scope

`original_scripts/` contains the scripts as they were run for the study, with the API address and key taken from
environment variables. They expect the working-folder layout of the study and are provided for inspection. Use
`reproduce/` to re-run. The Conscious Loop pipelines of Study B depend on platform changes that are not yet available on
the public RunBios API, so Study B is documented by its complete logs in `logs/conscious_loop/` rather than by a re-run
script.
