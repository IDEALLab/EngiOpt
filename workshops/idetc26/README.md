# IDETC'26 — Find the Best Model

A hands-on workshop about the evidence needed to compare generative models for
engineering design.

Teams work with ten candidate models for one topology-optimization problem: two
baselines, five trained generative models, and three constructed candidates. The
models are named, evaluated under the same conditions, and compared using cost,
distribution, constraint, and physics-based metrics. The resulting rankings
depend on both the selected metric and the representation space.

There is no model that is best independently of the intended use. Participants
must decide which evidence matters for their application and state what remains
uncertain.

**Status:** the Beams2D participant notebook, candidate bank, controls, cached
designs, and published physics results are implemented and runnable in Colab.
The main operational dependency is a cold-runtime download of approximately
510 MB after package installation.

---

## Workshop motivation

> Habibi et al., **[When Is it Actually Worth Learning Inverse Design?](https://asmedigitalcollection.asme.org/mechanicaldesign/article/148/6/061704/1229197/When-is-it-Actually-Worth-Learning-Inverse-Design)**
> *J. Mech. Des.* 148(6):061704 (2026), first presented at IDETC-CIE 2023.

That paper compared inverse-design model families on a topology-optimization
warm-start task and found that **k-nearest neighbours and random forests beat
deconvolutional networks when training data is limited**, once the cost of
generating that data is counted. It is an IDEAL Lab paper, on this task, first
given at this conference.

The workshop revisits that question with a broader model bank and a larger set
of evaluation criteria. Both sides of the paper's comparison are included as
full candidates: `knn_retrieval` at k=1 and `deconv_regression`.

On `beams2d`, the kNN baseline outperforms the conditional GAN on condition
adherence at 1/1600th of VQGAN's sampling cost, while kNN and VQGAN are within
5% of each other on MMD. This is why the baselines are ranked alongside the
trained models.

## Run it

[Open the participant notebook in Google Colab](https://colab.research.google.com/github/IDEALLab/EngiOpt/blob/codex/idetc26-workshop-participant-rework/workshops/idetc26/notebooks/01_find_the_best_model.ipynb)

The GitHub copy is read-only. Participants should select **File > Save a copy in
Drive** before editing.

```bash
jupyter lab workshops/idetc26/notebooks/01_find_the_best_model.ipynb
```

Colab installs EngiOpt from the branch named by the single `BRANCH` constant in
`tools/build_notebook.py` and EngiBench from `ENGIBENCH_REF`. The setup writes a
versioned marker under `/content` after verifying the installation. Repeated
executions skip installation and do not restart the runtime.

The notebook requests a T4. A GPU speeds up sampling after a cache miss; it does
**not** speed up the physics, which is CPU-bound topology optimization. A cold
runtime still downloads the EngiBench dataset and latent encoder from Hugging
Face, so network access is required during setup.

## Notebook structure

Ten suspects are evaluated on the same 50 design conditions. The notebook has
six sections:

**1 · the data** → **2 · the suspects** → **3 · the metric families** →
**4 · pixel, PCA, and learned latent spaces** → **5 · a selected metric board** →
**6 · the participant's accusation**. The final recommendation is discussed in
the room rather than recorded by the notebook.

The sequence moves from visual inspection to numerical evaluation and then to
the participant's own comparison. The notebook keeps API explanation brief;
the detailed command reference is available through `case.help()`.

**There are two commands.** `case.evaluate` for numbers, `case.show` for
pictures:

```python
from engiopt.workshops.idetc26 import Case

case = Case.open("beams2d")

case.help()                         # the cheat sheet: five calls, then every knob
case.models()                       # who is in the line-up
case.metrics()                      # what may be asked, by line of questioning

case.evaluate("diversity")          # all diversity metrics
case.evaluate("mmd", models="knn")  # one question, one suspect
answers = case.evaluate(["mmd", "novelty_ratio", "pixel_vendi"], controls=True)

case.show("diffusion")              # its designs
case.show("knn", "diffusion")       # two suspects on the same brief
case.show(answers)                  # a table of answers, drawn as ranks
case.show(answers, case.physics())  # selected metrics with published physics
case.show("knn", how="nearest_training")   # each design beside its closest training design
case.show("vqgan", "test", how="space_map")   # where its designs sit in a fitted space
```

**Neither command has a blanket form.** Calling `case.evaluate()` or
`case.show()` without specifying what to evaluate or display raises an error.
This requires participants to choose the metrics and models relevant to their
argument instead of generating a complete board by default. The errors include
an example of a valid call.

**`case.help()` is the command reference.** It starts with the five calls used in
the notebook — `models`, `metrics`, `explain`, `evaluate`, and `show` — followed
by the available views, controls, and optional arguments. Keeping this reference
in the API prevents notebook instructions from drifting when an argument
changes.

An earlier API exposed separate methods for computing, scoring, ranking, and
running physics. The current interface consolidates numerical evaluation under
`case.evaluate()` and expresses the differences as arguments.

The short form, `evaluate("mmd")`, uses the frozen evaluation specification.
Optional arguments such as `sigma`, `n_samples`, and `random_conditions` allow
participants to test how evaluation choices affect the result.

**Suspects are named by model family** from the first cell — `knn_retrieval`,
`diffusion_2d_cond`, and `cgan_cnn_2d`, for example — and can be addressed by an
unambiguous fragment such as `"diffusion"` or `"knn"`. Naming the models allows
participants to connect metric behavior with architecture and training method.

**No two suspects differ only by training seed.** The bank compares model
families and configurations rather than treating repeated training runs as
separate candidates.

There is no `seeds=` argument in the participant API. The specification freezes
the 50 design conditions, and changing the sampling seed on one checkpoint does
not measure variation across training runs. A training-seed study requires
separate checkpoints; those are outside the current participant workflow.

### Metric groups

Metrics are grouped by the question they answer. Participants can evaluate a
whole group or select individual metrics:

| Group | Question | Limitation or failure mode |
|---|---|---|
| `cost` | What did training and generation require? | Does not measure design quality |
| `similarity` | Do generated designs resemble held-out designs? | Returning training designs can score well |
| `memorization` | Are generated designs distinct from the training set? | Random noise can appear novel |
| `diversity` | How much does the generated set vary? | Corruption can increase diversity scores |
| `obedience` | Do designs satisfy the requested conditions? | Meeting a volume budget does not imply structural quality |
| `performance` | How do designs perform under optimization? | Requires the simulator |

The seventh group, `latent_space`, contains `lv_dual_gap`, `lv_active_dims`, and
`pca_dims`. These values describe the fitted representation rather than a
candidate model, so they are not ranked. `case.latent_space()` identifies the
autoencoder, its active width, and the matched PCA width.

Each metric group has a known failure mode, so no group is sufficient on its
own.

These workshop groups differ from the metric registry's implementation
families. The registry places `novelty` and `mmd` under `distribution`, but they
answer different questions: copying the training set can improve MMD, while
novelty is intended to detect that behavior. The workshop therefore separates
memorization. Similarly, each `lv_*` metric is grouped by the question it
answers, while its learned-latent dependency appears in the `space` column:

| the question | pixels | PCA subspace | learned latent |
|---|---|---|---|
| does this look like the real data? | `mmd` | `pca_mmd` | `lv_mmd` |
| is it copying the training set? | `novelty_ratio` | — | `lv_novelty` |
| did it cover the real modes? | — | `pca_coverage` | `lv_coverage` |
| how many distinct designs? | `pixel_vendi`, `dpp_geometric` | `pca_vendi` | `lv_vendi` |
| did it answer the brief? | `pixel_paired_distance` | — | `lv_paired_distance` |

Differences between these columns can come from the representation space rather
than the candidate model.

`case.show()` accepts a model name, two model names, a metric board, or selected
columns from a board. The `how=` argument provides the `designs`, `compare`,
`conditions`, `nearest_training`, and `space_map` views.

`space_map` displays a candidate's 50 designs in the pinned latent space or the
matched PCA subspace, with the training set as a reference. It exposes collapse,
poor coverage, and isolated samples that are difficult to interpret from a
single MMD value. Comparing `space="lv"` with `space="pca"` shows how the chosen
representation affects a distribution metric.

`case.evaluate()` uses the same evaluator, frozen specification, and argument
names as `python -m engiopt.evaluate`. It prints the equivalent CLI command so
results can be reproduced outside the notebook.

Physics-based metrics use the same evaluation interface as the cheap metrics.
The published board, available through `case.physics()`, avoids running the
simulator over all 50 designs during the session. A live request prints its
estimated cost before simulation begins.

## Cached designs and network requirements

Candidate designs for the specification's 50 conditions are precomputed and
shipped under `engiopt/workshops/idetc26/cache/`. The normal participant path
therefore does not sample checkpoints or download model weights.

```bash
# rebuild after the bank changes -- a cluster/workstation job, not a laptop one
python workshops/idetc26/tools/build_design_cache.py --problem-id beams2d --seeds 1 2 3
```

`case.evaluate(fresh=True)` explicitly resamples checkpoints and records the
runtime. Replayed `gen_seconds` values retain the machine on which they were
measured because wall-clock comparisons are only meaningful on comparable
hardware.

A cached candidate does not download its checkpoint at `Case.open()`, avoiding a
1.8 GB VQGAN download in the participant path. A cold runtime still downloads
the EngiBench dataset and latent encoder from Hugging Face. The measured transfer
was approximately 510 MB after package installation, so the workshop requires
reliable network access during setup.

## Repository structure

```
workshops/idetc26/
  notebooks/01_find_the_best_model.ipynb   the session (a build product)
  tools/build_notebook.py                  notebook SOURCE -- edit here, not the .ipynb
  tools/build_design_cache.py              pre-sample the bank so the session never waits
  tools/search_pool.py                     score the pool, then search it for a bank
  tools/condition_probe.py                 does a model respond to its condition at all?
  tools/verify_notebook.py                 execute every cell, fail on the first error

engiopt/workshops/idetc26/                 participant API, bank, config, and views
  problems/beams2d.json                    per-problem configuration
  cache/                                   packaged, pre-sampled designs
engiopt/baselines/                         dataset-fitted models (see below)
engiopt/generators/knn_retrieval/          kNN as a first-class generator package
engiopt/generators/linear_regression/      ridge regression, likewise
tests/test_idetc26.py                      harness verification
```

Colab installs the `engiopt` package rather than the repository's `workshops/`
tree. Problem configurations and design caches therefore live inside the package
and are included through `[tool.setuptools.package-data]`. A source checkout is
also supported; the package location is checked first.

## The model bank and controls

The Beams2D bank contains ten ranked candidates. Seven are pretrained entries:
`knn_retrieval`, `deconv_regression`, `vqgan`, `diffusion_2d_cond`,
`cgan_cnn_2d`, `gan_cnn_2d`, and `constrained_plvae_2d`. The other three are
constructed candidates designed to expose weaknesses in individual evaluation
criteria.

The participant-facing API describes what every model does but does not disclose
which candidates were constructed for the workshop. The facilitator must make
that disclosure after participants present their recommendations. The notebook
does not contain a passphrase or an automated reveal.

The bank also provides three controls: `collapsed`, `noise_doped`, and
`volume_only`. Controls provide reference values for interpreting metrics and
are never included in candidate rankings.

## Published baselines

`knn_retrieval` and `deconv_regression` are full generator packages with
training scripts that write checkpoints through `save_checkpoint_package`. They
are not notebook-specific implementations. `linear_regression` remains available
in the repository but is not part of the workshop bank; the deconvolutional
network is used because it is the comparator from the cited paper.

| | `knn_retrieval` | `deconv_regression` | `diffusion_2d_cond` | `vqgan` |
|---|---|---|---|---|
| checkpoint | 38.9 MB | **14.8 MB** | ~70 MB | ~1.8 GB |
| parameters | its training set | 3.7 M | 17.5 M | 102 M |
| 50 samples | 0.054 s | one forward pass | 452 s | 85 s |

Each package reports its parameter count, allowing the stored training data used
by kNN and the learned weights used by neural models to be compared on the same
cost axis.

## Measured costs

Development measurements on a laptop CPU gave the following ranges. The full
board used for these measurements contained eight candidates; the current bank
contains ten.

| | cost |
|---|---|
| the cheap board, 8 models × 50 samples | seconds |
| the physics board, per model × 50 samples | ~90–125 s |
| a whole 8-model physics board | **~15 min** |

`beams2d` optimization is CPU-bound, so a GPU does not reduce the physics cost.
Participants can evaluate a small subset live, while the complete physics board
is read from published results.

## Known model behavior

`tools/condition_probe.py` sweeps the requested volume fraction at fixed sampling
noise and regresses the realized fraction on the request. The current diffusion
checkpoint has a negative conditioning response:

| model | slope | r² | reading |
|---|---|---|---|
| `diffusion_2d_cond` | **−0.220** | 0.504 | ignores, and *anti-correlates with*, the request |
| `vqgan` | 0.902 | 0.993 | follows the request |

VQGAN follows the requested volume fraction through the same evaluation path,
which argues against a loading or measurement error. A current hypothesis is
that the UNet receives an unnormalized length-four conditioning token in which
`volfrac` ranges from 0.15 to 0.45 while `rmin` ranges from 1 to 3. Testing that
hypothesis requires retraining the diffusion model.

Current checkpoint packages include `design_min` and `design_max`. Legacy
NeurIPS checkpoints do not, which provides a direct way to distinguish the two
formats.

## Adding a problem

Add `engiopt/workshops/idetc26/problems/<problem_id>.json` with the frozen
EngiBench specification, display name, metric list, model bank, and controls:

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

`unavailable_metrics()` derives unsupported metrics from the frozen spec. For
example, `photonics2d` has no volume-fraction budget, so `viol` and related
condition metrics report themselves unavailable with a reason.

Every candidate must have a resolvable checkpoint package or construction.
Before using a new problem in a workshop, build its design cache, publish or
verify its physics results, and execute the notebook against the new
configuration.

## Editing and verification

The notebook is generated from `tools/build_notebook.py`. Edit the builder and
commit the source and generated notebook together:

```bash
python workshops/idetc26/tools/build_notebook.py
```

After changing the notebook or participant API, run:

```bash
pytest tests/test_idetc26.py -m "not network"
python workshops/idetc26/tools/verify_notebook.py
```

The notebook verification executes every code cell and requires network access.
