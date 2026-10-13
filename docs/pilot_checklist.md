# Scientific pilot readiness checklist

This checklist records the gaps identified in the starter experiment before changing
its behavior. Items are prioritized by what is required to interpret an order effect.

## P0 — needed for interpretable condition comparisons

- [x] Establish one saved, immutable base-model checkpoint and require every condition
  and order to load those exact weights.
- [x] Add an evaluated no-update baseline.
- [x] Add a request-count-matched control-update stream using prespecified
  non-forget split, alongside sequential forget-request updates.
- [x] Keep the exact evaluation row IDs, rendered prompts, truncation settings, and
  representation layer fixed across every condition and checkpoint.
- [x] Separate forget-update requests from held-out forget evaluation rows and reject
  exact normalized text overlap across dataset splits.
- [x] Freeze `cosine_separation` as the sole primary structural metric before main
  runs; label loss and any added measures as exploratory.
- [x] Record explicit dataset version/fingerprint, model requested and resolved
  revision, seed, hyperparameters, device, software versions, prompt template, and
  completed update count for every condition.
- [x] Persist every checkpoint's metrics in both CSV and JSON, with enough state to
  resume after interruption without repeating or skipping an update.
- [x] Add a multi-seed analysis that reports paired effect sizes and uncertainty;
  do not infer an effect from a single run.

The control requests are greedily paired without replacement to forget requests by
closest tokenized prompt length after truncation. Request and optimizer-step counts
match, and the same paired examples stay attached to each forget request across
original/shuffled orders; exact token/FLOP equality is not guaranteed and paired
lengths are recorded.

## P1 — required tests and reproducibility checks

- [x] Test restored checkpoint weights and tokenizer produce the same outputs.
- [x] Test deterministic split selection and stable evaluation IDs/prompts.
- [x] Test no-update and matched-control update counts, including resume behavior.
- [x] Test metric calculations against hand-computable vectors and test the analysis
  effect-size/interval calculations on known synthetic runs.
- [x] Pin an exact model revision and dataset version/hash before the main pilot.
- [x] Record failed/incomplete run status and error details; validate all comparisons use
  identical evaluation data and settings.

## P2 — scientific limitations to resolve or explicitly retain

- [ ] Define and freeze the inversion rule, analysis horizon, exclusions, seed count,
  and uncertainty procedure in the protocol before looking at main results.
- [x] Include separate never-taught control-update and retain-update streams; they
  address different comparison questions and are analyzed separately.
- [ ] Add utility/task-performance and generation-based evaluations if claims concern
  behavior rather than only token loss and hidden-state geometry.
- [ ] Audit semantic duplicates, prompt/data contamination, and dataset licensing.
- [x] Treat the cosine-separation score as a proxy, not a validated causal or universal
  entanglement measure; a changed score alone is not evidence of unlearning.

The pilot config intentionally contains a model-revision placeholder and cannot start
a remote model run until it is replaced by a full commit SHA. Dataset version and
content hash are recorded, but the dataset's scientific provenance, leakage audit,
and licensing still need to be supplied by the researcher.

## Implementation status

The checklist is a readiness record, not a claim that every scientific limitation is
solved. Update each checkbox only when code, tests, or a frozen protocol supplies the
corresponding evidence.
