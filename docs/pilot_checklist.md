# Scientific pilot readiness checklist

This checklist records the gaps identified in the starter experiment before changing
its behavior. Items are prioritized by what is required to interpret an order effect.

## P0 — needed for interpretable condition comparisons

- [ ] Establish one saved, immutable base-model checkpoint and require every condition
  and order to load those exact weights.
- [ ] Add an evaluated no-update baseline.
- [ ] Add a compute/update-matched control-update stream using a prespecified
  non-forget split, alongside sequential forget-request updates.
- [ ] Keep the exact evaluation row IDs, rendered prompts, truncation settings, and
  representation layer fixed across every condition and checkpoint.
- [ ] Freeze `cosine_separation` as the sole primary structural metric before main
  runs; label loss and any added measures as exploratory.
- [ ] Record explicit dataset version/fingerprint, model requested and resolved
  revision, seed, hyperparameters, device, software versions, prompt template, and
  completed update count for every condition.
- [ ] Persist every checkpoint's metrics in both CSV and JSON, with enough state to
  resume after interruption without repeating or skipping an update.
- [ ] Add a multi-seed analysis that reports paired effect sizes and uncertainty;
  do not infer an effect from a single run.

## P1 — required tests and reproducibility checks

- [ ] Test restored checkpoint weights and tokenizer produce the same outputs.
- [ ] Test deterministic split selection and stable evaluation IDs/prompts.
- [ ] Test no-update and matched-control update counts, including resume behavior.
- [ ] Test metric calculations against hand-computable vectors and test the analysis
  effect-size/interval calculations on known synthetic runs.
- [ ] Pin an exact model revision and dataset version/hash before the main pilot.
- [ ] Record failed/incomplete runs and deviations; validate all comparisons use
  identical evaluation data and settings.

## P2 — scientific limitations to resolve or explicitly retain

- [ ] Define and freeze the inversion rule, analysis horizon, exclusions, seed count,
  and uncertainty procedure in the protocol before looking at main results.
- [ ] Decide whether a control update should use never-taught control examples,
  retain examples, or both; these answer different questions and should not be
  conflated.
- [ ] Add utility/task-performance and generation-based evaluations if claims concern
  behavior rather than only token loss and hidden-state geometry.
- [ ] Audit semantic duplicates, prompt/data contamination, and dataset licensing.
- [ ] Treat the cosine-separation score as a proxy, not a validated causal or universal
  entanglement measure; a changed score alone is not evidence of unlearning.

## Implementation status

The checklist is a readiness record, not a claim that every scientific limitation is
solved. Update each checkbox only when code, tests, or a frozen protocol supplies the
corresponding evidence.
