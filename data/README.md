Place permitted JSONL datasets here. Do not commit private/confidential data. The
included demo dataset is synthetic and intended only to test code flow.

Use the `forget`, `retain`, and `control` splits only for evaluation. Supply separate
`forget_update`, `retain_update`, and `control_update` rows for update requests; these
update-only texts must not duplicate normalized evaluation text. The loader rejects
exact case-insensitive, whitespace-trimmed overlaps across any loaded split.
Include stable unique IDs and record the dataset version, source,
license, and preprocessing in the dataset card.
