# Protocol template — complete before main experiment

## Research question
Does sequential unlearning produce non-monotonic representation-geometry changes that exceed matched-control variation?

## Primary hypothesis
H1: the pre-defined structural score exhibits a sign change or rank reversal across sequential steps, beyond control variation.

## Null hypothesis
H0: changes are consistent with random seed, order, and general utility drift.

## Primary metric
Cosine separation = mean pairwise cosine similarity among forget representations (excluding self-pairs) minus mean cosine similarity between forget and control representations.

The implementation freezes `cosine_separation` as the primary structural metric.
Per-split token losses are exploratory. The `text` field is wrapped in the configured
`evaluation.prompt_template` for both updates and evaluation; retain one template,
row order, truncation limit, and representation layer for the whole pilot.

## Conditions implemented
- **No update:** evaluate and save the shared initial checkpoint without optimizer steps.
- **Forget GA:** sequential GA updates on separate `forget_update` requests; the
  `forget` evaluation rows are held out from updates.
- **Control GA:** the same number of GA requests/optimizer steps on separate
  `control_update` examples (not the `control` evaluation rows),
  paired greedily to forget requests by closest tokenized prompt length without
  replacement.
- **Retain GA:** same request-count and nearest-token-length matching on separate
  `retain_update` examples (not the `retain` evaluation rows).
- Each condition/order starts from the exact same saved base-checkpoint directory.
  The runner records its checkpoint SHA-256 and refuses mismatched run artifacts.
- Original and shuffled orders use the same selected forget request set. Per-request
  randomness is reseeded from the configured seed plus the one-based update index.
- Control/retain examples are paired against the original forget-request order once,
  then reordered by paired forget request ID for shuffled runs. This keeps the same
  matched examples attached to each forget request across order conditions.

Nearest-length pairing is a practical compute control, not proof of exact FLOP matching.
Review the recorded per-request token counts and disclose residual mismatch.

## Prespecified analysis requirement
Before main results, fill in the operational inversion rule, practical score tolerance,
analysis horizon, number of independent seeds, and exclusions in this protocol. The
analysis script reports paired difference-in-change estimates (forget GA minus the
no-update, control GA, and retain GA references) per step/order, standardized paired
effect sizes, and seed-bootstrap intervals. Those summaries do not by themselves test
the inversion event or prove deletion.

## Important caveat
This is an exploratory proxy. It is not a validated universal measure of entanglement and does not represent a literal knowledge graph.

## Freeze before main run
- Model and exact revision:
- Dataset version / source / license:
- Forget / retain / control split construction:
- Prompt templates:
- Representation layer:
- Sequence order(s):
- Unlearning hyperparameters:
- Number of seeds:
- Primary metric and inversion rule:
- Exclusion criteria:
- Compute ceiling:
- Planned controls:

## Report
Report all primary runs, control runs, seeds, failures, deviations, uncertainty, and utility trade-offs. Do not present post-hoc metric choices as pre-registered.
