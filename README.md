# Structural Entanglement in Continual Machine Unlearning Streams

A compute-conscious research pilot for testing whether sequential unlearning updates produce non-monotonic changes in representation geometry, compared with one-shot and matched controls.

> **Research caution:** this repository is an experimental scaffold, not evidence that "structural entanglement inversion" is novel or real. Freeze the operational definition before inspecting the main results. Report negative and inconclusive results honestly.

## Scope

- Hugging Face causal language models
- A small, explicit JSONL dataset format (included synthetic demo)
- Gradient-ascent unlearning baseline
- Sequential forget requests with checkpoint/evaluation after each request
- Forget/retain/control metrics and a representation cosine-separation metric
- Repeatable configs, seeds, logs, tests, and plots

The starter implementation intentionally does **not** claim to reproduce every paper's GA/NPO/RMU method. GA is implemented as a simple baseline. NPO/RMU require method-specific objective choices and should be added only after verifying the source paper and implementation details.

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

On Windows, after a successful sequential run, print its recorded evaluation metrics
and plot them with:

```powershell
python scripts\04_evaluate.py --config configs\demo.yaml
python scripts\05_make_figures.py --input results\runs\demo_smoke_test\sequential_original\sequential_metrics.csv --output results\figures\sequential_metrics.png
```

For a GPU experiment, use Kaggle rather than this local CPU-only check. First provide
the permitted dataset, update `configs\pilot.yaml` with the exact model revision,
dataset path, and frozen protocol settings, and verify the model license and dataset
terms. Run scripts `02` and `03` in the Kaggle notebook/session using the corresponding
Kaggle paths; do not copy the Windows commands' local paths unchanged.

## Dataset format

Each JSONL row is an object:

```json
{"id":"forget_001","text":"A fictional profile says the sample code is ALPHA.","split":"forget","request_id":"request_1"}
{"id":"retain_001","text":"A fictional profile says the sample code is BETA.","split":"retain","request_id":"retain"}
{"id":"control_001","text":"A fictional profile says the sample code is GAMMA.","split":"control","request_id":"control"}
```

Allowed `split` values: `forget`, `retain`, `control`. Use fictional or otherwise permitted data. Do not put personal or confidential data in the repository.

## Run workflow

1. Prepare and validate JSONL.
2. Evaluate the unmodified model and cache baseline results.
3. Run one-shot GA or sequential GA updates.
4. Evaluate each checkpoint on the same fixed split rows and raw text strings.
5. Compare structural-score changes with matched controls.
6. Inspect `results/runs/*/sequential_*/sequential_metrics.csv` and
   `results/runs/*/run.json`.

## Important limitations

- The included synthetic demo tests code flow, not real unlearning quality.
- A lower answer loss / changed answer is not by itself proof of deletion.
- The cosine-separation metric is a simple representation-geometry proxy, not a literal knowledge graph or a validated universal entanglement measure.
- One seed and a tiny model are exploratory only.
- Evaluation uses each row's raw `text` as a causal-LM sequence; the scaffold has no separate question/answer prompt template or generation-based deletion test.
- `generation_max_new_tokens` is currently unused because no generation-based evaluation is implemented. Gradient accumulation is not implemented; values other than `1` now fail explicitly.
- `primary_structure_metric` documents the intended metric, but only `cosine_separation` is implemented.
- Checkpoints save model and tokenizer files, but not optimizer state; resuming the unlearning optimizer is not implemented.
- `control` rows are used as the structural comparison group and are not used for unlearning updates. The scaffold does not yet implement a matched control-update arm, so this alone cannot separate a structural effect from general model degradation.
- Add compute-matched retain-only/irrelevant-data updates, an unmodified-model baseline, utility and task-performance checks, multiple seeds/orders, and uncertainty estimates before making causal claims.
- Sequential and control runs must be matched for update count and compute before making causal claims.
