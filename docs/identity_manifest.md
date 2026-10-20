# Verified identity manifest

The hosted SEP-28k schema recorded by the validated baseline contains event counts and audio, but no verified speaker, episode, or recording identifier. The default v2 split is therefore exact-duplicate-aware and explicitly marked `NOT_VERIFIED` for identity independence.

To request a speaker-, episode-, or recording-disjoint split, provide a CSV with exactly one row per unique selected `sample_id`:

```csv
sample_id,group_id,group_kind,identity_source,identity_revision,identity_verified
sha256:...,speaker-001,speaker,curated-mapping.csv,9f2c1e7,true
```

Requirements:

- `sample_id` must match the run's source identifier or derived audio-content key exactly.
- `group_kind` must be consistently `speaker`, `episode`, or `recording`.
- `group_id` must come from a legitimate source; do not derive it from row order or filenames.
- `identity_source` must identify the mapping source.
- `identity_revision` must pin the mapping version or checksum.
- every row must explicitly set `identity_verified=true`.
- the manifest must cover the selected sample exactly, with no missing, duplicate, or extra rows.

Set `dataset.metadata_manifest` to the CSV path. The runner fails rather than silently reverting to an unverified split.

For a legitimate identifier already present in a future dataset schema, set `verified_group_kind`, `verified_group_column`, `identity_source`, and `identity_revision` together. The last two fields must document the external evidence used to verify the field's semantics. A column name alone is not treated as verification.
