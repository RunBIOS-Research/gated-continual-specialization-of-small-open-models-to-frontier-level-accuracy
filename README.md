# Gated Continual Specialization of Small Open Models to Frontier-Level Accuracy: data, logs and code (version 1.0.0)

This package accompanies the technical report *The Conscious Loop: Gated Continual Specialization of Small Open Models to
Frontier-Level Accuracy at a Fraction of the Cost* (Arpit Sharma, Tathagat Sharma, Bhavesh Soni, Mohit Kumar, Suraj Sharma, Anubhav Tripathi, Pranjal Bawa, Ankit Sharma and the UltraSafe AI Team, UltraSafe AI, October 2026), included here as `paper.pdf`.

It contains everything needed to check the numbers in the report without access to RunBios: the exact test and validation
cases, the training rows of every selected model, every answer from every model with its billed cost, the complete run log
and the scripts used.

## What the paper claims, and where the evidence is

| Claim | Evidence in this repository |
| --- | --- |
| On five enterprise classification tasks built from real public records, an open model of 8 to 27 billion parameters fine-tuned on RunBios was more accurate than Claude Opus 5.5, GPT-5.5 and GPT-6 Astra on all five, with a statistically significant margin on three (banking +6.6, complaints +8.6, contracts +4.2 points) and parity within sampling error on two (XBRL tagging +1.0, vulnerability triage +0.4) | `results/`, `runs/run_log.csv`, recomputed by `verify.py` |
| The reported model of each task was chosen on separate validation cases, never on the test cases | `data/<task>/validation.jsonl`, the validation columns of `runs/run_log.csv`, and the selection check in `verify.py` |
| No test case was used for training | `data/<task>/test.jsonl` and `data/<task>/train_selected_model.jsonl.gz` |
| Serving cost per request was 4.1 to 19.0 times lower than the strongest frontier model | Billed cost of every frontier answer in `results/`, serving price and measured throughput in `runs/run_log.csv` and `logs/training/*/deployment.json` |
| Accuracy rose with the volume and recency of real training data | The development paths in `runs/run_log.csv` and the training curves in `logs/training/` |
| The Conscious Loop trained, tested and promoted successive versions on its own (Study B) | `loop/pipelines.json` and every attempt in `logs/conscious_loop/` |
| The results can be reproduced on the public RunBios API | `reproduce/`, `REPRODUCE.md`, `CONSCIOUS_LOOP.md` and the re-runs in `reproduction/` |

The paper does not claim that a fine-tuned model is better than frontier models in general. Its claims are limited to
these tasks and these test cases.

## Verify the results

```
python verify.py
```

`verify.py` uses only the Python standard library. It reapplies the selection rule (highest validation accuracy) to the run
log, then recomputes for each task the exact-match accuracy of the selected model, its 95% Wilson interval, the strongest
frontier model, the difference, the exact McNemar p value and the cost per 1,000 requests. It stops with an error if the
selection rule does not reproduce the reported model. Run `shasum -a 256 -c MANIFEST.sha256` to confirm that no file has
changed since publication.

## Independent re-run

After the study, the selected banking-intent-routing configuration was retrained on the public RunBios API with
`reproduce/train.py` and the released training rows. It scored 94.0% on the 500 test cases, against the reported 95.0%
(95% interval 92.7% to 96.6%), for $7.02 of training. The run log, training records and every answer are in
`reproduction/banking-intent-routing/`.

A Conscious Loop pipeline for the same task was then created and run on the public RunBios API with
`reproduce/conscious_loop.py`. Its benchmark rose from 47.5% (untrained) to 83.5% with the first version, promoted on the
Loop's own held-back verdict, and to 89.0% with a second version that the pipeline started by itself and built on the first
(promoted under the Study B rule, since its held-back comparison was not significant). Every record is in
`reproduction/conscious-loop-banking-intent-routing/`.

### Live read-only previews

The runs made on the production RunBios platform can also be opened live, without an account. Each page shows the
settings, training curve, validation results, checkpoints, comparisons, benchmark answers and costs, and allows no
download or deployment.

| Task | Run | What it shows | Live preview |
|---|---|---|---|
| Banking intent routing | Retraining of the selected model from `data/banking-intent-routing/` | 94.0% on the 500-case test set, $7.02 | https://platform.runbios.ai/public/training/XulG3ywp1CfnzUBezlY9QvNL |
| Banking intent routing | Conscious Loop pipeline, two versions | 47.5% → 83.5% → 89.0% on its 200-case benchmark | https://platform.runbios.ai/public/loop/C0Ga-gNRE_ueMs0SbomYtQuS |

The five-task study itself ran in the RunBios pre-production environment, which is not reachable from the internet,
so it has no live pages. This repository is the permanent record of every result, and nothing in it depends on the
pages above staying online. Further production runs will be added to this table as they are made.

## Contents

| Path | Contents |
| --- | --- |
| `paper.pdf` | The technical report |
| `data/<task>/task.json` | Source, licence, label set and both instruction formats |
| `data/<task>/test.jsonl` | The 500 test cases (`in_benchmark_200` marks the Conscious Loop benchmark subset) |
| `data/<task>/validation.jsonl` | The validation cases used for model selection |
| `data/<task>/train_selected_model.jsonl.gz` | The exact training rows of the selected model |
| `results/<task>/<model>.<split>.jsonl` | One line per answer: case id, reference label, answer, exact match, tokens, billed cost |
| `runs/run_log.csv` | Every trained configuration with scores, costs, serving configuration and platform record IDs |
| `runs/frontier_baselines.csv` | Frontier model scores and billed cost per task |
| `loop/pipelines.json` | Conscious Loop pipelines (Study B): every attempt, its benchmark scores, verdict and outcome |
| `logs/training/` | For each of the 41 training jobs: the job record and configuration, per-step training and validation metrics, checkpoints, evaluations and, where it was served, the deployment and throughput measurement. `logs/training/index.csv` lists them |
| `logs/conscious_loop/` | For each pipeline: its settings, benchmark and benchmark history, and for every attempt the run record, the training job logs, the held-back comparison and the benchmark replay, answer by answer |
| `reproduce/` | Standalone scripts that use only the public RunBios API: `evaluate.py` re-asks any model the test cases, `train.py` retrains and tests a selected model, `conscious_loop.py` runs a Conscious Loop pipeline (see `REPRODUCE.md`) |
| `CONSCIOUS_LOOP.md` | How the Conscious Loop works and how to run your own pipeline, with every API call |
| `reproduction/` | An independent re-run on the public RunBios API made after the study |
| `original_scripts/` | The data preparation, training, evaluation and throughput scripts as they were run |
| `verify.py` | Independent recomputation of the reported results |
| `REPRODUCE.md` | How to reproduce the results at four levels |
| `MANIFEST.sha256` | SHA-256 checksum of every file |

The platform does not expose raw trainer console output, so training logs are the structured records RunBios keeps for
each job: metrics at every logged step, checkpoints and evaluations. Account, storage and network identifiers were
removed from every record before publication.

Tasks: `banking-intent-routing`, `consumer-complaint-triage`, `contract-clause-classification`, `xbrl-numeric-tagging`,
`vulnerability-root-cause-triage`.

Model files are named `runbios-ft_<base model>_<training rows>_<instruction>`, where the instruction is `list` (the
instruction with the full label list, as given to the frontier models) or `short` (a one-sentence instruction).

## How the data were built

`original_scripts/prepare.py` draws the test and benchmark cases from each source and builds the first training sets. The
validation sets and the larger training sets of later rounds (more records from the same official training splits, NVD
records published from January 2025 to June 2026, and the one-sentence instruction variant) were built with the same
rules: the same label set, the same filters and deduplication against every validation and test case by normalized text.
Because some of those steps were run as one-off commands rather than as a script, the package includes the exact rows
instead: `data/<task>/validation.jsonl` and `data/<task>/train_selected_model.jsonl.gz` are the files that were used.

Checks run before publication:

- no training row of any configuration has the same normalized text as a test case (for XBRL tagging also at sentence
  level with the markers removed);
- every test file contains exactly 500 cases, and every model's answers cover all of them;
- for complaint triage and vulnerability triage, every test case is later than every training record.

## Platform records

Every training job, deployment, dataset and Conscious Loop pipeline is identified in `runs/run_log.csv` and
`loop/pipelines.json` by its RunBios record ID. The records are kept in the RunBios pre-production environment. Reviewers
and auditors may request supervised access to them, under a confidentiality agreement, from UltraSafe AI.

## Data sources and licences

See `LICENSE.md`. In brief: Banking77 and LEDGAR are CC-BY-4.0, FiNER-139 is CC-BY-SA-4.0, the CFPB Consumer Complaint
Database is CC0 and the National Vulnerability Database is in the public domain. Because the package contains material from
FiNER-139, the package as a whole is released under CC-BY-SA-4.0. The scripts are released under the MIT License.

## Citation

See `CITATION.cff`.
