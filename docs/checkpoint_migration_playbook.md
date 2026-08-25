# Checkpoint Migration Playbook

This playbook describes how to migrate historical EngiOpt checkpoints from Weights & Biases (W&B) to the Hugging Face Hub (HF) without breaking backward compatibility.

This is guidance only for phase 1 of the HF checkpoint backend rollout.

What this phase does not do:
- delete historical W&B artifacts
- mutate historical W&B artifacts in place
- assume every historical run should be migrated

## Goals

1. Stop creating new long-lived checkpoint pressure in W&B by making HF the default backend for new saved models.
2. Keep historical W&B checkpoints loadable while HF-backed packages roll out.
3. Create a safe path to reclaim W&B storage later, after validation.

## Recommended Scope

Start with the official or release subset of checkpoints first, not the full historical backlog.

Good early candidates:
- checkpoints referenced in papers, benchmarks, or public notebooks
- checkpoints linked from README examples
- checkpoints used by downstream evaluation scripts or case studies

## Migration Package Format

Each migrated HF checkpoint package should be self-contained and include:
- the original model weight file or files
- `run_config.json`
- `metadata.json`

`metadata.json` should record at least:
- `problem_id`
- `algo`
- `seed`
- original W&B project, run id, and artifact names
- HF repo id and package path
- primary file list

## Recommended Procedure

1. Inventory the target artifacts.
   Create a manifest with artifact name, aliases, run id, problem, algorithm, seed, expected files, and whether the artifact is part of the official/release subset.

2. Freeze the migration manifest before uploads.
   Use the manifest as the source of truth so the migration is reproducible and reviewable.

3. Download the original W&B artifacts.
   For each target artifact, download the stored files and extract the associated W&B run config needed to rebuild the model outside W&B.

4. Build the HF package.
   Upload the weights together with `run_config.json` and `metadata.json` into the per-model-family HF repo under the deterministic package path:
   `problem_id[/extra_parts]/seed_<seed>`

5. Record the mapping.
   For every migrated checkpoint, store a durable mapping from the W&B artifact alias to the HF repo id, revision, and package path.

6. Validate restores before cleanup.
   Test a representative sample end-to-end with EngiOpt’s evaluation or surrogate-model loading path and confirm the HF package reproduces the expected model restore behavior.

7. Add pointers back into W&B metadata.
   Once validated, update run metadata, summaries, or notes so the WandB run points to the canonical HF checkpoint location.

8. Keep an overlap period.
   Retain both HF and W&B copies long enough to verify that downstream users and scripts are not relying on the old blob storage path.

9. Only then define deletion policy.
   Deletion or retention changes should happen in a separate maintenance pass, using the manifest as the authoritative record.

## Validation Checklist

Before any cleanup is considered for a migrated checkpoint:

1. The HF package contains all required weight files.
2. `run_config.json` is present and sufficient to rebuild the model.
3. `metadata.json` is present and points back to the original W&B lineage.
4. A real EngiOpt load path has succeeded against the HF package.
5. The corresponding W&B run contains the HF pointer or mapping information.

## Cleanup Guidance for Later

When the project is ready to reclaim W&B storage:

1. Start with official checkpoints only.
2. Delete in small batches, not all at once.
3. Confirm the manifest entry is complete before each deletion.
4. Re-run a small restore audit after each batch.
5. Keep at least one validated overlap window where both copies coexist.

## Notes

- W&B remains part of the lineage story even after HF becomes the canonical checkpoint host.
- HF should be treated as the long-lived storage backend for public or durable checkpoints.
- Backward compatibility matters more than immediate cleanup.

## Post-training Top-K Selection

The 2D flow-matching comparison uses a two-stage validation procedure. Training
keeps the five checkpoints with the lowest validation MMD locally. Evaluation
then compares those five candidates by downstream validation COG and selects one
checkpoint before touching the test split.

For durable publication, archive only that selected checkpoint together with:
- `validation_metrics.json`, containing the training-time validation history
- `selection_results.json` and `selection_results.csv`, containing all five candidate scores
- `run_config.json` and `metadata.json`
- the selected checkpoint checksum, source revisions, seeds, and W&B run links

The selected checkpoint is sufficient to reproduce final test evaluation. The
candidate score files preserve the selection evidence without multiplying model
storage by five. Evaluators can either upload directly to HF for a smoke test or
stage bundles locally for a serialized upload job. Full campaigns should use the
staged path so concurrent GPU jobs never commit to the same HF repository.
