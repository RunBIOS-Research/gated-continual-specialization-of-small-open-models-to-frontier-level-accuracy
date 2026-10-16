# The Conscious Loop: evidence package (version 1.0.0)

This package accompanies the technical report *The Conscious Loop: Gated Continual Specialization of Small Open Models to
Frontier-Level Accuracy at a Fraction of the Cost* (UltraSafe AI Research, October 2026), included here as `paper.pdf`.

It contains everything needed to check the numbers in the report without access to RunBios: the exact test and validation
cases, the training rows of every selected model, every answer from every model with its billed cost, the complete run log
and the scripts used.

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
| `reproduce/` | Standalone scripts that use only the public RunBios API to re-ask the frontier models and to retrain and test a selected model (see `REPRODUCE.md`) |
| `reproduction/` | An independent re-run on the public RunBios API made after the study |
| `original_scripts/` | The data preparation, training, evaluation and throughput scripts as they were run |
| `verify.py` | Independent recomputation of the reported results |
| `REPRODUCE.md` | How to reproduce the results at three levels |
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
