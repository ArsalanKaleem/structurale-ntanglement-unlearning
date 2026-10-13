# Pilot protocol v1.0 — frozen

**Status: FROZEN on 2026-10-13, before any main-run metrics were produced.** The commit
that introduces this version is the freeze point. Changes after this point are allowed
only as dated amendments at the end of this file, each stating whether any main-run
result had been seen when it was written. Smoke-test outputs may be inspected; they are
not main results.

## Research question
When a language model receives a stream of gradient-ascent "forget" requests, does the
representation geometry of the targeted material change **non-monotonically** — i.e. does
its separation from other material reverse direction partway through the stream — beyond
what equally sized gradient-ascent updates on non-target text produce, and how much does
that trajectory depend on the order of the requests?

## Positioning and what would be novel

Literature checked on 2026-10-13 (details in `docs/related_work.csv`):

- **Sequential unlearning degrades utility cumulatively.** MUSE (sequential folds),
  continual-unlearning work (Gao et al., arXiv 2407.10223), and FIT (Xu et al., arXiv
  2601.21682, up to 300 requests) show utility and forget accuracy decline as requests
  accumulate. FIT argues overlapping requests accumulate gradients along shared
  directions. None tracks representation geometry step by step, and FIT leaves request
  order to future work.
- **Representation diagnostics show suppression, not deletion.** "Unlearning Isn't
  Deletion" (Xu et al., ICML 2026, arXiv 2505.16831) uses PCA similarity/shift, CKA,
  and Fisher information on continual unlearning (6–100 requests), comparing original,
  unlearned, and relearned models. It shuffles request order only to average it out
  (4 seeds), and its controls are size-matched *relearning* sets, not matched
  *unlearning-update* streams.
- **Entanglement predicts collateral damage.** Wybitul et al. (arXiv 2609.02285)
  measure forget–retain entanglement on final-layer, mean-pooled, unit-normalized
  hidden states — the same representation used here — but only **before** single-shot
  unlearning, never along a request stream. CLReg (arXiv 2601.22028) and KUDA
  (arXiv 2602.19275) shape forget/retain representation overlap as an objective.

**Candidate contributions of this pilot** (claimed only if the data support them):

1. **Step-resolved entanglement trajectories.** Geometry is measured after every one of
   16 sequential requests, not only at endpoints.
2. **Matched-update contrast.** Forget-GA trajectories are compared against GA streams
   with the same request count, optimizer steps, and nearest-token-length pairing on
   (a) unrelated text and (b) same-domain retain text. This separates geometry changes
   specific to the targeted material from generic gradient-ascent drift — a control
   absent from the work above.
3. **Pre-registered non-monotonicity ("inversion") test**, defined below, instead of
   post-hoc description of curves.
4. **Order sensitivity as an estimand.** GA here is deterministic given the request
   order, so variation across seeds *is* variation across request permutations. The
   between-permutation spread is reported as a result, not averaged away.
5. **Entity-level spill-over.** Held-out facts about the same fictional entities show
   whether geometry changes generalize beyond the exact updated sentences.

**Scope limit that bounds every claim.** The base model has never learned the fictional
facts. This pilot therefore studies how sequential GA streams reshape representations of
targeted vs. non-targeted text; it does **not** test removal of learned knowledge.
A follow-up stage that first fine-tunes on the fictional facts (TOFU-style) is required
before any claim about unlearning knowledge, and would be a separate protocol.

## Hypotheses
- **H1 (primary):** the forget-GA trajectory of `cosine_separation`, relative to the
  matched control-GA trajectory, shows an inversion (rule below) within 16 requests.
- **H0:** paired differences are single-signed, monotone, or indistinguishable from 0
  given the spread across request orders.
- **Exploratory, no confirmatory claim:** (E1) the same rule applied to
  `cosine_separation_forget_retain`; (E2) order sensitivity at step 16; (E3) token-loss
  trajectories for forget, retain, and control rows.

## Metrics
- **Primary:** `cosine_separation` = mean pairwise cosine similarity among held-out forget
  representations (self-pairs excluded) minus mean cosine similarity between forget and
  control representations. A proxy, not a validated measure of entanglement.
- **Secondary (exploratory):** `cosine_separation_forget_retain`, the same quantity with
  retain rows in place of control rows (same-domain entanglement, the pairing used by
  Wybitul et al.).
- **Exploratory:** per-split mean token loss (`forget_`, `retain_`, `control_mean_token_loss`).

## Frozen settings
- Model and exact revision: `Qwen/Qwen2.5-0.5B-Instruct` at
  `7ae557604adf67be50417f59c2c2f167def9a775`, loaded in the pinned environment
  (`requirements-kaggle.lock`).
- Dataset version / source / license: `fictional-authors-v1`
  (`data/fictional_authors_v1.jsonl`, SHA-256
  `5644e12fbc7e304c67903efb945220505219eefb4cd9264e36a6353dde315103`), generated by
  `scripts/00_generate_fictional_dataset.py` with seed 20261013; MIT. See
  `docs/dataset_card.md`.
- Split construction: 16 fictional authors to forget, 16 different fictional authors to
  retain, 16 fictional islands as controls. Per entity, 4 held-out evaluation facts and 4
  different update-only facts (64 rows per split). All 64 rows of each evaluation split
  are evaluated at every step.
- Prompt template: `{text}` for both updates and evaluation.
- Representation: final hidden layer (`fixed_layer: -1`), mean over non-padding tokens,
  `max_length: 256`.
- Analysis horizon: **16 sequential requests** (`sequence_length: 16`), one per forget
  author; steps 0–16 are analyzed and no later steps are added.
- Unlearning: gradient ascent, AdamW, learning rate 1e-5, weight decay 0, 10 optimizer
  steps per request, batch size 1, gradient-norm clipping 1.0, fresh optimizer per request.
- Checkpoints: `checkpoint_retention: resume_only` (only the checkpoint needed to resume
  is kept; metrics are recorded at every step). This does not change any computation.

## Conditions and replication
- Per seed: no-update baseline (step 0), forget GA, control GA on `control_update`, and
  retain GA on `retain_update`, all from that seed's saved base checkpoint.
- Control and retain requests are paired to forget requests by nearest tokenized length
  without replacement and stay attached to the same forget request in every order.
  Residual token-count mismatch is recorded and reported.
- **Seeds: 17, 23, 29, 31, 37** (replacements, only for excluded seeds and in this
  order: 41, 43).
- **Primary analysis uses the `shuffled` order for all 5 seeds** (5 request
  permutations). The `original` (file) order is run for seed 17 only and reported as a
  single descriptive trajectory without an interval. Rationale: with batch size 1 and one
  example per request, GA is deterministic up to GPU numerical noise, so `original`-order
  runs at different seeds would be pseudo-replicates.

## Inversion rule (primary, applied to `cosine_separation`)
- For seed *s* and step *t*, with reference condition ref:
  D_ref(s, t) = [S_forget(s, t) − S_forget(s, 0)] − [S_ref(s, t) − S_ref(s, 0)],
  as computed by `scripts/06_analyze.py`. **Control GA is the primary reference**;
  retain GA and no-update are secondary references.
- Practical tolerance: τ = 0.10 × |S(0)|, with S(0) the base checkpoint's
  `cosine_separation` (shared by all conditions; a baseline, not an outcome). If
  |S(0)| < 0.01, τ = 0.001.
- Step *t* is **reliably positive** (negative) when the mean of D_control(·, t) over the
  5 shuffled-order seeds is ≥ +τ (≤ −τ) **and** its seed-bootstrap 95% percentile
  interval (5000 resamples) excludes 0.
- **Inversion is declared** when steps 1–16 contain a reliably positive step followed
  later by a reliably negative step, or a reliably negative step followed later by a
  reliably positive step.
- Reporting categories:
  - **Supported:** inversion against control GA, and the same inversion (same directions,
    any steps) against retain GA.
  - **Partial:** inversion against control GA only — reported as "inversion relative to
    unrelated updates only".
  - **Not detected:** otherwise (monotone, single-signed, or inconclusive). All per-step
    estimates are shown regardless.
- With 5 permutations the bootstrap interval is coarse and is reported as descriptive
  uncertainty, not a calibrated test. No multiplicity correction is applied; requiring two
  opposite-signed reliable steps is the guard against chance sign flips, and this is
  disclosed.

## Order sensitivity (exploratory, E2)
Report the standard deviation and range of D_control(·, 16) across the 5 shuffled seeds,
plus the seed-17 original-order value. Because GA is deterministic given order, this
spread estimates order dependence. No threshold or claim is pre-registered.

## Exclusion criteria (technical only)
- A run that cannot complete after resuming (`status: failed`), or any step producing a
  non-finite loss or metric.
- Any fingerprint mismatch (dataset, model revision, base checkpoint, evaluation
  settings) rejected by the analysis script.
- An excluded seed is replaced by the next listed replacement seed for **all** of its
  conditions and orders; every exclusion and its reason is reported.
- Never excluded on metric values or direction. Model collapse (e.g. very high retain
  loss) is reported as a utility cost, not excluded.

## Compute ceiling
At most 25 GPU-hours on Kaggle (18 condition runs × 16 requests plus 5 baselines). If the
smoke-test projection exceeds this, the only permitted reductions, decided and recorded
as an amendment **before** any main run, are, in order: seeds 5 → 3 (17, 23, 29), then
horizon 16 → 8.

## Report
Report all runs, seeds, failures, exclusions, deviations, residual token-count mismatch,
uncertainty, and utility costs (token losses). State the scope limit (no fine-tune-first
stage) next to every result. Do not present exploratory analyses or post-hoc metric
choices as pre-registered.

## Amendments
None.
