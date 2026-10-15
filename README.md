# Structural Entanglement in Continual Machine Unlearning Streams

A compute-conscious research scaffold for testing whether sequential unlearning updates produce non-monotonic changes in representation geometry, compared with no-update and matched-update controls.

> **Research caution:** this repository is an experimental scaffold, not evidence that "structural entanglement inversion" is novel or real. Freeze the operational definition before inspecting the main results. Report negative and inconclusive results honestly.

## Study status

| Stage | Protocol | Status |
|---|---|---|
| Pilot v1 (GA, no fine-tuning, 5 orders) | [docs/protocol.md](docs/protocol.md) | Complete — [results](docs/results_pilot_v1.md): rule verdict "supported" but the model collapsed and the verdict was not robust |
| **Study v2** (fine-tune first, calibration, NPO + GA, 10 orders, spill-over) | [docs/protocol_v2.md](docs/protocol_v2.md) | Frozen; run [`notebooks/kaggle_study_v2.ipynb`](notebooks/kaggle_study_v2.ipynb) on Kaggle |

v2 pipeline: `00` generate data (main + calibration) → `01b` fine-tune with knowledge
gate → `03` calibration runs → `09` frozen hyperparameter selection → `02`/`03` main runs
→ `06` paired effects → `07 --protocol v2` inversion rule → `10` spill-over → `08`
figures. `tests/test_pipeline_v2.py` runs the whole chain on a tiny CPU model.

## Scope

- Hugging Face causal language models, fine-tuned on synthetic fictional facts
- A small, explicit JSONL dataset format (synthetic main and calibration sets)
- Gradient ascent and NPO (Zhang et al., 2024) unlearning
- Sequential forget requests with evaluation after each request
- No-update, control-update, and retain-update conditions from one shared base checkpoint
- Token losses (aggregate and per example), forget–control and forget–retain cosine
  separation at several layers
- Restartable runs, provenance, machine-readable metrics, tests, and statistical summaries

RMU and other methods are not implemented; add them only after verifying the source paper.

The runner records the dataset's SHA-256 and the model's resolved revision.

## Safe Windows checks (no model download or training)

Python 3.10 or 3.11 is recommended. From the repository root, these commands install
the project, disable CUDA for this shell, run the CPU tests, and validate the bundled
synthetic data. They do not load a model:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
$env:CUDA_VISIBLE_DEVICES = "-1"
.\.venv\Scripts\python.exe -m pytest -q
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\01_prepare_data.py --input data\demo.jsonl --output data\processed
```

`pip install -e ".[dev]"` may install Python packages, but does not download a model.
Do not run scripts `02` or `03` on a PC when model downloads/training are not wanted:
the baseline loads the configured model, and the sequential runner additionally updates
it. The bundled `demo` config points to `sshleifer/tiny-gpt2`; this still requires a
model download if it is not cached and is not suitable for scientific conclusions.

`02_run_baseline.py` writes the shared `base_model` checkpoint and evaluates the
**no-update** condition. Each invocation of `03_run_sequential.py` runs a condition
and order from that exact saved checkpoint, writes one metric row per completed
request to both CSV and JSON, and can be rerun to resume from the latest complete
request checkpoint. The control and retain requests are greedily paired to the
selected forget requests by nearest tokenized prompt length, without replacement;
the same examples remain paired to each forget request across original/shuffled
orders. Their lengths are recorded, but this does not guarantee exact FLOP equality.

The fixed primary metric is `cosine_separation`; loss columns are exploratory. The
analysis command reports paired difference-in-change estimates against no-update,
control GA, and retain GA, standardized effect sizes, and seed-bootstrap intervals.
One seed cannot support a meaningful uncertainty interval or an inversion claim.

```powershell
.\.venv\Scripts\python.exe scripts\04_evaluate.py --config configs\pilot.yaml
.\.venv\Scripts\python.exe scripts\05_make_figures.py --input results\runs\sequential_pilot\condition_forget_original\metrics.csv --output results\figures\forget_original.png
.\.venv\Scripts\python.exe scripts\06_analyze.py --input results\runs\sequential_pilot --output results\analysis
```

Those post-run commands operate on existing results and do not load a model. Do not
run scripts `02` or `03` on a PC when model downloads or training are not wanted.
Use Kaggle for experiments, and substitute the Kaggle dataset/output paths in the
config before running.

## Dataset format

Each JSONL row is an object:

```json
{"id":"forget_001","text":"A fictional profile says the sample code is ALPHA.","split":"forget","request_id":"request_1"}
{"id":"forget_update_001","text":"A separate fictional update profile says code FOXTROT.","split":"forget_update","request_id":"request_1"}
{"id":"retain_001","text":"A fictional profile says the sample code is BETA.","split":"retain","request_id":"retain"}
{"id":"control_001","text":"A fictional profile says the sample code is GAMMA.","split":"control","request_id":"control"}
{"id":"control_update_001","text":"An unrelated fictional text says code DELTA.","split":"control_update"}
{"id":"retain_update_001","text":"A separate fictional retained text says code EPSILON.","split":"retain_update"}
```

Allowed `split` values: `forget`, `retain`, `control`, `forget_update`,
`control_update`, and `retain_update`. `forget`/`retain`/`control` are held-out
evaluation examples. All three `_update` splits supply training requests and cannot
exactly overlap normalized evaluation text or each other. Use fictional or otherwise
permitted data. Do not put personal or confidential data in the repository.

## Kaggle pilot commands

After filling the model revision, dataset version/path, and frozen protocol in
`configs/pilot.yaml`, run from the repository root in Kaggle. This runs all conditions
for one configured seed; repeat with distinct seed/output-directory configs for
independent replicates.

```bash
python scripts/01_prepare_data.py --input data/fictional_authors_v1.jsonl --output data/processed
python scripts/02_run_baseline.py --config configs/pilot.yaml
python scripts/03_run_sequential.py --config configs/pilot.yaml --condition forget --order original
python scripts/03_run_sequential.py --config configs/pilot.yaml --condition forget --order shuffled
python scripts/03_run_sequential.py --config configs/pilot.yaml --condition control --order original
python scripts/03_run_sequential.py --config configs/pilot.yaml --condition control --order shuffled
python scripts/03_run_sequential.py --config configs/pilot.yaml --condition retain --order original
python scripts/03_run_sequential.py --config configs/pilot.yaml --condition retain --order shuffled
python scripts/04_evaluate.py --config configs/pilot.yaml
python scripts/06_analyze.py --input results/runs/sequential_pilot --output results/analysis
python scripts/05_make_figures.py --input results/runs/sequential_pilot/condition_forget_original/metrics.csv --output results/figures/forget_original.png
```

To resume after a Kaggle interruption, rerun the interrupted `03` command with the
same config. Completed request checkpoints and paired metrics are retained; the runner
rejects changed configs, data, evaluation prompts, or base checkpoints in an existing
run directory. Each update request is reseeded from `experiment.seed + step`.

For independent replicates, create one config per seed with the same dataset,
model/revision, and settings, but distinct `experiment.seed` and `experiment.output_dir`
(for example `results/runs/pilot_seed_17`). Run the full command block for each config,
then point `06_analyze.py --input` at their common parent directory. The analysis
requires exactly one matching no-update, control, and retain run per forget run.

## Kaggle notebook workflow

The gated, section-by-section workflow is in
[`notebooks/kaggle_pilot.ipynb`](notebooks/kaggle_pilot.ipynb); its exact package pins
are in [`requirements-kaggle.lock`](requirements-kaggle.lock). Before importing the
notebook into Kaggle, publish a commit containing both files, then replace
`REPOSITORY_COMMIT` near the top of the notebook with that commit's full 40-character
SHA and publish the pin update. This two-commit sequence avoids a self-referential
commit SHA: the selected pinned commit must already contain both files. The notebook
intentionally will not clone an older commit that lacks its lock.

In Kaggle, create a GPU notebook, upload/import the notebook, attach the permitted
JSONL dataset as an input, and run sections 1–5 in order. Fill in the model commit SHA,
dataset path/version/source/license, and the two human-reviewed rights confirmations
in section 3. The license metadata/access request is only a check; it does not grant
permission or replace reviewing the terms. Section 4 runs tests and validates the
dataset. Section 5 downloads/loads the selected model for a tiny baseline plus one
request each of forget, control, and retain updates. It writes observed wall time,
CUDA peak allocation, and a clearly labeled linear projection derived from those
measurements. The notebook does not estimate a Kaggle quota or runtime allowance.

Do not set `START_MAIN_EXPERIMENT = True` in section 6 until the smoke run and
permissions have been reviewed. Save a Kaggle Notebook Version after smoke and after
each condition. For a new session, attach the prior saved output and set
`RECOVERY_ARTIFACTS` in the optional recovery cell to the attached
`/kaggle/input/<attached-version>/pilot-artifacts` directory before continuing.
Artifacts, configuration, logs, metrics, and checkpoints are stored under
`/kaggle/working/pilot-artifacts`; recovery requires the prior Kaggle output version
to have been saved and attached.

## Run workflow

1. Freeze the protocol and validate/version the dataset.
2. Save and evaluate the shared base checkpoint (`no_update`).
3. Run forget, control, and retain update streams in original and shuffled orders.
4. Inspect each condition's `metrics.csv`, `metrics.json`, and `run.json`.
5. Analyze paired effects across independent seeds; review matching-token counts.

## Important limitations

- The included synthetic demo tests code flow, not real unlearning quality.
- A lower answer loss / changed answer is not by itself proof of deletion.
- The cosine-separation metric is a simple representation-geometry proxy, not a literal knowledge graph or a validated universal entanglement measure.
- One seed and a tiny model are exploratory only.
- Evaluation and unlearning apply the same configured text template to each JSONL `text`; there is no generation-based deletion test or separate question/answer schema.
- `generation_max_new_tokens` is currently unused because no generation-based evaluation is implemented. Gradient accumulation is not implemented; values other than `1` now fail explicitly.
- `cosine_separation` is the sole primary metric; per-split token losses are exploratory.
- The baseline/no-update and all update conditions share a saved initial checkpoint. Checkpoints are request-boundary model/tokenizer states; the per-request optimizer is recreated for each request.
- All update rows (`forget_update`, `control_update`, `retain_update`) are separate from held-out evaluation rows; exact case-insensitive, whitespace-trimmed text duplication anywhere across loaded splits is rejected.
- Control requests match request and optimizer-step counts and greedily minimize token-length differences. Exact compute matching is not guaranteed.
- Utility/task-performance and generation-based checks, semantic leakage/contamination audits, a prespecified inversion threshold/rule, and adequate independent seeds remain necessary before making causal claims.
