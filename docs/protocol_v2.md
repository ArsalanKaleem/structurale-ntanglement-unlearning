# Study protocol v2.0 — frozen

**Status: FROZEN on 2026-10-16, before any v2 data existed.** The commit introducing this
file is the freeze point. Later changes are dated amendments at the end, each stating
whether any v2 calibration or main-run result had been seen. v1 (`docs/protocol.md`,
results in `docs/results_pilot_v1.md`) is complete and is not modified.

## What v1 taught us and what changes

v1 used gradient ascent (GA) with learning rate 1e-5 and 10 steps per request on a model
that had never learned the facts. Every stream collapsed the model (held-out loss
6 → ~150 nats) and the formal "supported" verdict did not survive leaving out any one of
the 5 seeds. v2 fixes each cause:

| Problem in v1 | v2 design |
|---|---|
| Facts never learned, so nothing to unlearn | Fine-tune on all facts first; a knowledge gate must pass |
| Hyperparameters chosen by hand, model collapsed | Pre-registered calibration on separate entities selects them |
| Collapsed steps counted as evidence | Collapse guard: steps where any stream lost > 20% of its non-target knowledge are excluded |
| One seed could carry the verdict | 10 request orders; each reliable step must survive leave-one-out |
| Only GA (unbounded objective) | NPO (bounded) is primary; GA replicates |
| Forget-vs-islands metric is not what "entanglement" means in the literature | Primary metric is forget-vs-retain (same-domain) separation |

## Research questions

- **RQ1.** As a model processes a stream of unlearning requests, does the same-domain
  entanglement between targeted and retained knowledge change non-monotonically, beyond
  what a matched stream of unlearning *unrelated* knowledge produces?
- **RQ2.** Does unlearning spill over to same-domain knowledge that is never targeted, and
  is it local to the targeted entities?
- **RQ3 (descriptive).** How much do these trajectories depend on request order, and at
  which depth in the network do they occur?

## Positioning and novelty

Closest work (details in `docs/related_work.csv`): "Unlearning Isn't Deletion" (ICML 2026)
compares representations at endpoints of continual unlearning and averages over request
order; FIT (2026) studies up to 300 requests with output metrics and leaves order to
future work; Wybitul et al. (2026) measure forget–retain entanglement on the same
representation as this study, but only before single-shot unlearning. MUSE and TOFU
establish fictional-knowledge and sequential settings without geometry trajectories.

**Contributions this study can make** (each claimed only if supported):
1. Step-resolved forget–retain entanglement trajectories over a 16-request stream, in a
   regime verified *not* to have collapsed (calibration + collapse guard).
2. Matched-update contrasts: the same unlearning operation applied to unrelated learned
   knowledge (control stream) and to same-domain learned knowledge (retain stream), with
   equal request counts, optimizer steps, and nearest-token-length pairing.
3. Entity-level spill-over: same-domain never-targeted authors (H2a) and held-out facts of
   targeted vs not-yet-targeted authors within the stream (H2b).
4. Order dependence as an estimand: unlearning updates are deterministic given the order,
   so spread across the 10 shuffled seeds measures order effects directly.
5. Layer profile (layers 6, 12, 18, final) and a bounded (NPO) vs unbounded (GA)
   comparison of the same trajectories.

**Scope limits:** one 0.5B model, synthetic templated facts, two unlearning methods.
Results describe this setting; generalization to larger models and real data is untested.

## Fixed materials

- Model: `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae557604adf67be50417f59c2c2f167def9a775`, pinned
  packages (`requirements-kaggle.lock`), Kaggle GPU (T4 or P100).
- Main data: `data/fictional_authors_v1.jsonl` (SHA-256 `5644e12f…315103`): 16 forget
  authors, 16 retain authors, 16 control islands; per entity 4 evaluation and 4 update
  facts (64 rows per split).
- Calibration data: `data/fictional_authors_calibration_v1.jsonl` (SHA-256
  `79bcfdb4…f51d46d8e`): same design, generator seed 20261016, no names or texts shared
  with the main data (enforced by a test). Used only in calibration.
- Prompt template `{text}`; `max_length` 256; representations are mask-aware mean-pooled
  hidden states at the final layer (primary) and layers 6, 12, 18 (exploratory).

## Stage 1 — fine-tune (knowledge acquisition)

- Full fine-tuning on all 768 facts (main + calibration, all six splits), AdamW, learning
  rate 2e-5, 5 epochs, batch 16, seed 0, one shared checkpoint for every later run.
- **Knowledge gate:** for every split of both datasets, mean token loss after fine-tuning
  ≤ 0.3 × before. If it fails, the only permitted change (as an amendment, before any
  calibration) is more epochs: 10, then 20.
- Knowledge fraction of a split at step t: K(t) = (L_pre − L_t) / (L_pre − L_0), with L_pre
  the pre-fine-tuning loss and L_0 the fine-tuned (step-0) loss. K = 1 fully known, 0 back
  to the pre-fine-tuning level.

## Stage 2 — calibration (hyperparameter selection)

- Per method (NPO β = 0.1; GA), grid: learning rate {1e-6, 3e-6, 1e-5} × steps per request
  {2, 5}; calibration entities; forget stream; original order; 16 requests; seed 1.
- **Selection rule** (`scripts/09_select_hyperparameters.py`): admissible if
  K(retain, t) ≥ 0.8 and K(control, t) ≥ 0.8 at every step **and** efficacy
  1 − K(targeted update facts, step 16) ≥ 0.5. Choose the admissible configuration with
  the highest efficacy (ties: lower learning rate, then fewer steps).
- If a method has no admissible configuration, its main runs do not start; any grid
  extension is an amendment written before main runs.

## Stage 3 — main experiment

- Methods: **NPO (primary)**, GA (replication), each with its selected hyperparameters;
  batch size 1, fresh AdamW per request, gradient clipping 1.0.
- Streams per seed: forget (unlearn forget-author facts), control (unlearn island facts),
  retain (unlearn retain-author facts), each 16 requests in a seed-specific shuffled
  order, plus a no-update evaluation. Control and retain requests are paired to forget
  requests by nearest token length.
- **Seeds (request orders): 17, 23, 29, 31, 37, 41, 43, 47, 53, 59.** Replacements, only for
  technically excluded seeds, in order: 61, 67.
- Every step records aggregate metrics, per-example losses, and geometry at all layers.

## Hypotheses and decision rules

### H1 (primary): non-monotonic entanglement trajectory — NPO
- Metric S = `cosine_separation_forget_retain` (final layer): mean within-forget cosine
  similarity minus mean forget–retain cosine similarity.
- Paired effect per seed s, step t, reference ref ∈ {control, retain, no-update}:
  D_ref(s, t) = [S_forget(s, t) − S_forget(s, 0)] − [S_ref(s, t) − S_ref(s, 0)].
  **Control stream is the primary reference.**
- τ = 0.10 × |S(0)| (S(0) = fine-tuned base value; τ = 0.001 if |S(0)| < 0.01).
- **Collapse guard:** step t is *valid* only if, averaged over seeds, every stream keeps
  K ≥ 0.8 on each of its non-target splits (forget stream: retain, control; control
  stream: forget, retain; retain stream: forget, control).
- Step t is **reliably positive (negative)** if it is valid, the mean of D over the 10
  seeds is ≥ +τ (≤ −τ), its seed-bootstrap 95% percentile interval (5000 resamples)
  excludes 0, and the mean of every leave-one-seed-out subset is also ≥ +τ (≤ −τ).
- **Inversion:** a reliable step followed by a later reliable step of opposite sign.
- Verdict (`scripts/07_apply_protocol.py --protocol v2`): **supported** if the same
  inversion pattern holds against the control and the retain streams; **partial** if
  against the control stream only; otherwise **not detected**.
- **Replication:** the same rule applied to GA. "Replicated" only if GA also gives
  "supported" with the same pattern.

### H2 (secondary, NPO; GA reported alongside) — spill-over (`scripts/10_spillover.py`)
- **H2a, same-domain spill-over:** at step 16, the retain-split loss increase under the
  forget stream minus that under the control stream, per seed. Supported if the mean is
  > 0 and its bootstrap 95% interval excludes 0.
- **H2b, entity locality:** at step 8 of the forget stream, the loss increase on held-out
  facts of already-targeted authors minus not-yet-targeted authors, per seed. Supported
  if the mean is > 0 and the interval excludes 0.

### H3 and exploratory analyses (no confirmatory claims)
- Order sensitivity: SD and range of D_control(·, 16) across the 10 seeds.
- Layer profile: H1's statistics at layers 6, 12, 18.
- Forget-vs-control separation (`cosine_separation`, the v1 metric) under the v2 rule.
- Token-loss and knowledge-fraction trajectories for every split and stream.

**Multiplicity:** H1 (NPO) is the single primary test. H2a and H2b are secondary and
reported with 95% intervals without correction; this is disclosed. With 10 orders the
percentile bootstrap remains approximate; the leave-one-out requirement guards H1
against single-order effects.

## Exclusions

Technical only: a run that cannot complete after resuming, non-finite losses or metrics,
or a fingerprint mismatch. An excluded seed is replaced by the next replacement seed for
all its streams and both methods. Never excluded for metric values, direction, or loss of
utility. Collapse is handled by the guard, not by exclusion, and is reported.

## Compute ceiling

At most 30 GPU-hours for stages 1–3. If the smoke projection exceeds it, the only allowed
reductions, written as an amendment before calibration, are, in order: seeds 10 → 7
(17–43), then GA main runs dropped (GA calibration still reported).

## Reporting

Report every stage's gate, all calibration candidates, all runs, failures, exclusions,
deviations, per-step estimates with intervals, guard status per step, utility
(knowledge fractions), and the scope limits. Report null and partial outcomes in full.
Label every post-hoc analysis as such.

## Amendments

None.
