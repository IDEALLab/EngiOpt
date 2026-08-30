# Conditional Flow Matching Reproduction v1.0.0

This release freezes the EngiOpt implementation and public reproduction materials for the conditional
flow-matching engineering inverse-design study.

It provides:

- the conditional 2D flow-matching training and evaluation implementation;
- corrected conditional Diffusion and sigmoid-cGAN baselines;
- validation-only Top-5 MMD shortlisting followed by validation COG selection;
- local and Hugging Face selected-checkpoint bundles with selection evidence;
- deterministic CSV aggregation, qualitative export, and synchronized generation timing;
- a sanitized 220-experiment manifest and self-contained Slurm launchers.

The model weights are hosted in the linked Hugging Face collection rather than attached to this GitHub
release. W&B contains the corresponding run configurations, metrics, and checkpoint references. See
`reproducibility/conditional_flow_matching/ARTIFACTS.md` for immutable artifact revisions.

The published training runs used EngiOpt commit `b2e61a5571e4b99355e46b7c842ec3f1e89226f7`
and EngiBench commit `217394232bd628484a72b72ce624b1bdd955245a`. Final evaluation and qualitative
reproduction fixes are included through EngiOpt commit `9e3e521b722427bd85aad279110504ae58cfd6ef`.
