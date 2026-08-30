# Public experiment artifacts

The experiment records and selected evaluation checkpoints are published separately from Git.

## Weights & Biases

- [Experiment report](https://wandb.ai/smassoudi-eth-z-rich/engiopt-flow-matching/reports/Conditional-Flow-Matching-for-ML-Based-Inverse-Design-Problems---Experiments-and-Reproducibility--VmlldzoxNzgzMTUyMQ==)
- Project: `smassoudi-eth-z-rich/engiopt-flow-matching`
- Campaign group: `engopt2026-reproduction-v1`

The report links the training and selected-checkpoint evaluation runs. It records model configuration,
checkpoint selection, test metrics, and the corresponding Hugging Face checkpoint reference.

## Hugging Face

- [Conditional Flow Matching for Engineering Inverse Design collection](https://huggingface.co/collections/IDEALLab/conditional-flow-matching-for-engineering-inverse-design-6a9328bf1c39d74d4460ba46)
- `IDEALLab/engiopt-engopt2026-flow-matching-2d-cond` at revision `58ae97c4aa59784fbe22ab232b5d196ff088e4c7`
- `IDEALLab/engiopt-engopt2026-diffusion-2d-cond` at revision `ba6e9a191157a5068d9168c56f34130f3372c6d8`
- `IDEALLab/engiopt-engopt2026-cgan-cnn-2d` at revision `638ec26ff3ef1d3d75c4775f6712140aad7c9172`

Each published package contains the selected checkpoint, run configuration, validation shortlist,
selection results, metadata, and checksums needed for evaluation. The intermediate Top-5 candidates
are not duplicated in the public repositories; the evidence used to select the released checkpoint is
included with the package.

## Datasets

- `IDEALLab/beams_2d_50_100_v0` at revision `ccb64d4f09cf66a5ed9ba2081438d695b9e2438d`
- `IDEALLab/heat_conduction_2d_v0` at revision `9f07c1e70d244f783e34a71c96db621173871dc1`

## Audit

The final audit completed without errors or warnings. It verified 220 evaluation CSV files, 220
selected checkpoint bundles, 220 Hugging Face packages, and the linked training and evaluation W&B
runs. The original manifest SHA-256 is
`df65d492a7f5c333c74ed6c29c10c181fa21b08b127d5de4e5b09e4d6bdaca26`.
