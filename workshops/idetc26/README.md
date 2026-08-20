# IDETC-CIE 2026: Find the Best Model

This workshop asks participants to compare generative models for an engineering
design problem. The models are evaluated under the same conditions, but the
metrics do not necessarily agree. Participants inspect the designs, select the
evidence they consider relevant, and explain which model they would use.

## Status

The participant notebook, ten-model Beams2D bank, controls, cached designs, and
published physics results are implemented. The notebook runs in Google Colab and
uses one participant path from setup to the final discussion.

The main operational dependency is network access. A cold Colab runtime installs
EngiOpt and EngiBench and downloads the EngiBench dataset and latent encoder.
Testing measured approximately 510 MB of downloads after package installation.

## Participant notebook

[Open the workshop in Google Colab](https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/idetc26-workshop-participant-rework/workshops/idetc26/notebooks/01_find_the_best_model.ipynb)

The notebook opens read-only from GitHub. Participants should select
**File > Save a copy in Drive** before making changes.

To run it from a local checkout:

```bash
jupyter lab workshops/idetc26/notebooks/01_find_the_best_model.ipynb
```

### Colab setup

The setup cell installs:

- EngiOpt from the branch defined by `BRANCH` in `tools/build_notebook.py`.
- EngiBench from the Git reference defined by `ENGIBENCH_REF`.

The installation is guarded by `/content/.idetc26_setup_v1`. The first run
installs and verifies the required EngiBench problem version. Repeated executions
skip installation, so **Run all** does not trigger a restart loop. A factory reset
clears both the installed packages and the marker.

The notebook requests a T4 runtime. The normal participant path reads cached
designs and published physics, but a GPU prevents a long wait if a design cache
miss requires sampling from a checkpoint.

## Workshop sequence

The notebook contains six sections:

1. **The dataset**: inspect the training and held-out test designs as the problem
   conditions change.
2. **The suspects**: inspect the ten candidate models and compare generated
   designs under the same conditions.
3. **The interrogation**: evaluate cost, similarity, novelty, diversity,
   constraint satisfaction, and optimization performance.
4. **Representation spaces**: compare distance-based measurements in pixels, a
   matched PCA subspace, and a learned latent space.
5. **The board**: assemble a ranking from selected metrics.
6. **The accusation**: choose a model, explain the evidence, and identify what
   else should be checked before using it in practice.

The murder-mystery terminology is the workshop theme. The model names and metric
values are not anonymized.

## Model bank

The Beams2D bank contains ten candidates evaluated on the same frozen EngiBench
specification:

- k-nearest-neighbour retrieval
- supervised deconvolutional regression
- VQGAN
- conditional diffusion
- conditional GAN
- unconditional GAN
- constrained least-volume autoencoder
- three constructed candidates designed to expose weaknesses in individual
  evaluation criteria

The constructed candidates participate in the ranking. Their construction is
not printed by the participant-facing API. The facilitator must disclose them
after participants present their recommendations; there is no passphrase or
automated reveal in the notebook.

Three controls provide reference values and are never ranked with the candidate
models:

- `collapsed`: one averaged design repeated for every condition
- `noise_doped`: held-out optimized designs with added Gaussian noise
- `volume_only`: random designs satisfying only the volume constraint

## Evaluation interface

Participants work with one `Case` object:

```python
from engiopt.workshops.idetc26 import Case

case = Case.open("beams2d")
case.models()
case.metrics()
case.explain("diffusion")
case.show("diffusion")
case.evaluate("mmd")
```

`case.evaluate()` returns numerical results. `case.show()` handles design
galleries, comparisons, condition checks, nearest-training examples,
representation-space maps, and metric boards. `case.help()` prints the full
participant reference.

The workshop groups metrics by the question they answer:

| Group | Question |
|---|---|
| `cost` | What did training and generation require? |
| `similarity` | How closely does the generated set match held-out designs? |
| `memorization` | Are generated designs distinct from the training set? |
| `diversity` | How much does the generated set vary? |
| `obedience` | Do designs satisfy the requested conditions? |
| `performance` | How do designs perform under physics-based optimization? |
| `latent_space` | Is the fitted representation suitable for measurement? |

The `latent_space` group contains diagnostics for the representation and is not
used to rank models. Expensive performance results are read from published
physics records when available. Passing `fresh=True` explicitly resamples a
model or recomputes results on the current machine.

## Data and runtime behavior

Pre-sampled candidate designs are packaged under
`engiopt/workshops/idetc26/cache/`. The normal participant path therefore does
not download candidate checkpoints or run model sampling.

`Case.open()` still resolves the frozen EngiBench dataset and the latent-space
encoder. These assets are downloaded from Hugging Face on a cold runtime and
then cached by Colab. Internet access is therefore required during setup.

Published physics values are also read from the Hub. The notebook evaluates only
a small subset live; a complete simulator sweep is not expected during the
session.

## Repository layout

```text
workshops/idetc26/
  notebooks/01_find_the_best_model.ipynb  generated participant notebook
  tools/build_notebook.py                 canonical notebook source
  tools/build_design_cache.py             rebuild pre-sampled designs
  tools/search_pool.py                    inspect and select candidate banks
  tools/verify_notebook.py                execute the notebook end to end

engiopt/workshops/idetc26/
  case.py                                 participant API
  bank.py                                 candidate and control loading
  config.py                               per-problem configuration
  designs.py                              cached design access
  families.py                             participant metric groups
  views.py                                notebook visualizations
  problems/                               workshop problem configurations
  cache/                                  packaged generated designs
```

## Editing the notebook

The `.ipynb` file is generated. Edit `tools/build_notebook.py`, then rebuild it:

```bash
python workshops/idetc26/tools/build_notebook.py
```

The builder assigns stable cell identifiers and produces deterministic JSON.
Commit the source and generated notebook together.

After changing the notebook or participant API, run:

```bash
pytest tests/test_idetc26.py -m "not network"
python workshops/idetc26/tools/verify_notebook.py
```

The second command executes the complete notebook and requires network access.

## Adding a problem

Add `engiopt/workshops/idetc26/problems/<problem_id>.json` with the frozen
EngiBench spec, display name, metrics, candidate bank, and controls:

```json
{
  "problem_id": "example2d",
  "spec": "example2d/v1",
  "display_name": "Example2D",
  "metrics": ["mmd", "pixel_vendi", "viol", "iog"],
  "bank": [
    {
      "kind": "pretrained",
      "algo": "knn_retrieval",
      "seed": 1,
      "config_fingerprint": "..."
    }
  ],
  "controls": [
    {"kind": "reference", "algo": "collapsed"},
    {"kind": "reference", "algo": "noise_doped"}
  ]
}
```

Every configured metric must be registered in EngiOpt and supported by the
frozen spec. Every candidate must have a resolvable package or construction.
Before using the problem in a workshop, build its design cache, publish or
verify its physics results, and run the notebook verification against it.
