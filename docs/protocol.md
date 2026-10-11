# Protocol template — complete before main experiment

## Research question
Does sequential unlearning produce non-monotonic representation-geometry changes that exceed matched-control variation?

## Primary hypothesis
H1: the pre-defined structural score exhibits a sign change or rank reversal across sequential steps, beyond control variation.

## Null hypothesis
H0: changes are consistent with random seed, order, and general utility drift.

## Primary metric
Cosine separation = mean pairwise cosine similarity among forget representations (excluding self-pairs) minus mean cosine similarity between forget and control representations.

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
