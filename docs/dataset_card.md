# Dataset card — fictional authors v1

- Dataset name and version: `fictional-authors-v1`
  (file `data/fictional_authors_v1.jsonl`, SHA-256
  `5644e12fbc7e304c67903efb945220505219eefb4cd9264e36a6353dde315103`; checked out with
  LF line endings via `.gitattributes`)
- Source URL or DOI: none; generated in this repository by
  `scripts/00_generate_fictional_dataset.py` with seed `20261013`. Regenerate with
  `python scripts/00_generate_fictional_dataset.py`; a test checks the committed file
  matches the generator output.
- License / terms: MIT (repository license). All text is invented from small seeded
  word lists; no third-party or copyrighted content.
- Intended research use: pilot runs of sequential gradient-ascent unlearning and the
  `cosine_separation` representation-geometry proxy.
- Data collection and preprocessing: three entity groups of 16 entities, each entity
  with 8 templated one-sentence facts. For every entity, 4 randomly chosen facts are
  held out for evaluation and the other 4 are update-only. Update rows are interleaved
  across entities, so update request *k* targets entity *k* mod 16; the first 16
  requests each touch a different entity.
  - Forget group: 16 fictional authors (`forget`, `forget_update`).
  - Retain group: 16 different fictional authors in the same domain/templates
    (`retain`, `retain_update`).
  - Control group: 16 fictional islands, an unrelated domain (`control`,
    `control_update`).
- Number of held-out forget evaluation examples: 64 (16 authors × 4 facts)
- Number of forget-update examples (not reused for evaluation): 64
- Number of retain examples: 64
- Number of retain-update examples (separate from evaluation retain rows): 64
- Number of never-taught/control examples: 64
- Number of control-update examples (separate from evaluation controls): 64
- Each row also carries an `entity` field (ignored by the loader) for grouping analyses.
- Possible overlap/leakage: no exact text overlap (enforced by the loader). Evaluation
  and update rows for the same entity share the entity name and templates, which is
  intentional: held-out forget facts measure spill-over to unseen facts about the same
  entity. Forget and retain authors share sentence templates by design. Invented names
  are syllable combinations and were not checked against every real name; any
  coincidental match is unintended.
- Privacy considerations: no real people, personal data, or confidential content.
- Known limitations:
  - **The base model has never learned these facts.** Gradient ascent here pushes down
    the likelihood of novel text rather than removing prior knowledge (benchmarks such
    as TOFU first fine-tune on the fictional facts). Results speak to how sequential
    GA reshapes representations, not to removal of memorized knowledge. A
    fine-tune-first stage would be needed for that claim.
  - Templated sentences are short and repetitive; template structure may dominate the
    representation geometry.
  - The control group differs in both entity type and templates, so forget-vs-control
    separation starts high; interpret changes relative to the no-update baseline.
