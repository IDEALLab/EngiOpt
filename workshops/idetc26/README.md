# IDETC'26 — Find the Best Model

A hands-on workshop about what you would have to measure before you were
entitled to say one generative model is better than another.

Teams get an anonymized bank of models for one engineering design problem, the
three metrics papers actually report, and a commitment cell. They pick a winner
and defend it. Then the ranking comes apart — under a change of random seed,
under the cheap columns they were not given, and under the physics.

**The answer is meant to be unknown.** Not withheld, not rigged — genuinely
undetermined by the metrics the field reports. The bank is chosen by searching
the published checkpoint pool for the subset whose metrics disagree most, and
disagreement is measured rather than hoped for.

**Status: harness working end to end; bank selection in progress.** See
[Open questions](#open-questions).

---

## The argument

> Habibi et al., **[When Is it Actually Worth Learning Inverse Design?](https://asmedigitalcollection.asme.org/mechanicaldesign/article/148/6/061704/1229197/When-is-it-Actually-Worth-Learning-Inverse-Design)**
> *J. Mech. Des.* 148(6):061704 (2026), first presented at IDETC-CIE 2023.

That paper compared inverse-design model families on a topology-optimization
warm-start task and found that **k-nearest neighbours and random forests beat
deconvolutional networks when training data is limited**, once the cost of
generating that data is counted. It is an IDEAL Lab paper, on this task, first
given at this conference.

So the session is the natural sequel: *you* try to answer that question for a
bank of real models, using the metrics you can afford, and discover that you
cannot. kNN and ridge regression sit in the bank as **competitive entries**, not
as jokes. On `beams2d` the kNN baseline currently beats the conditional GAN on
MMD, feasibility *and* condition adherence, at 1/1600th of VQGAN's sampling
cost. A leaderboard topped by retrieval is a result, not a broken leaderboard.

## Run it

```bash
jupyter lab workshops/idetc26/notebooks/01_find_the_best_model.ipynb
```

Colab installs EngiOpt from the branch named by the single `BRANCH` constant in
`tools/build_notebook.py`. A GPU speeds up sampling from the trained models; it
does **not** speed up the physics, which is CPU-bound topology optimization.

## What is in the box

```
workshops/idetc26/
  notebooks/01_find_the_best_model.ipynb   the session (a build product)
  problems/beams2d.json                    the parameterization unit
  sealed/                                  the preregistered physics board
  tools/build_notebook.py                  notebook SOURCE -- edit here, not the .ipynb
  tools/search_pool.py                     score the pool, then search it for a bank
  tools/condition_probe.py                 does a model respond to its condition at all?
  tools/build_sealed_board.py              compute and seal the physics board
  tools/verify_notebook.py                 execute every cell, fail on the first error

engiopt/workshops/idetc26/                 bank, challenge, config, sealing
engiopt/baselines/                         dataset-fitted models (see below)
engiopt/generators/knn_retrieval/          kNN as a first-class generator package
engiopt/generators/linear_regression/      ridge regression, likewise
tests/test_idetc26.py                      harness verification
```

The directory is `tools/`, not `build/`, because the root `.gitignore` ignores
`build/` and silently dropped every script in it from a commit.

## The bank, and what is not in it

Bank members are **plausibly-valid models somebody might actually use**: trained
checkpoints from the Euler sweep (`cgan_cnn_2d`, `gan_cnn_2d`, `vqgan`,
`diffusion_2d_cond`, `constrained_plvae_2d`) plus the two published baselines.
Selection is by measured disagreement, not by hand.

`engiopt/baselines/references.py` holds four deliberately-flawed models —
`collapsed`, `volume_only`, `noise_doped`, `checkerboard`. Every one carries
`bank_eligible = False` and `ModelBank` refuses to rank them. They are
**calibration standards**: what does a diversity metric read on one repeated
design, or on real designs plus noise at known severity? That belongs on a
reference row beneath a board, the way a scale bar belongs on a micrograph.
Putting them in the bank would be a trick; putting them beside it is the
measurement.

## Two baselines, published like any other model

`knn_retrieval` and `linear_regression` are full generator packages with
training scripts that write checkpoints through `save_checkpoint_package`. They
are not notebook-only conveniences.

| | `linear_regression` | `knn_retrieval` | `diffusion_2d_cond` | `vqgan` |
|---|---|---|---|---|
| checkpoint | **0.18 MB** | 38.9 MB | ~70 MB | ~1.8 GB |
| parameters | 45,000 | its training set | 17.5 M | 102 M |
| 50 samples | **0.005 s** | 0.054 s | 452 s | 85 s |

Both report real parameter counts, so a kNN's data cost and a network's weight
cost land on the same axis. Three checkpoint sizes side by side are a better
answer to *what is a model?* than any definition.

## Measured costs

From actual runs on a laptop CPU, not estimates:

| | cost |
|---|---|
| the cheap board, 8 models × 50 samples | seconds |
| the physics board, per model × 50 samples | ~90–125 s |
| a whole 8-model physics board | **~15 min** |

The project's working assumption was ~16 GPU-hours. That came from SLURM
*requested* walltime and had never been measured; the real figure is two orders
of magnitude smaller. `beams2d`'s optimizer is CPU-bound, so **a GPU does not
help the physics at all.** The practical consequence is that teams could compute
part of the expensive board live, which is a different session from the one
where it has to be sealed.

## Findings that changed the plan

**The diffusion model's conditioning is wired backwards.** `tools/condition_probe.py`
sweeps the requested volume fraction at fixed sampling noise and regresses
realized on requested:

| model | slope | r² | reading |
|---|---|---|---|
| `diffusion_2d_cond` | **−0.220** | 0.504 | ignores, and *anti-correlates with*, the request |
| `vqgan` | 0.902 | 0.993 | follows the request |

Ask for more material, get less. VQGAN goes through identical plumbing at slope
0.90, so this is not a loading bug. The post-NeurIPS diffusion fixes addressed
design-range normalization and the noise schedule; none touched conditioning.
Hypothesis worth testing: conditions reach the UNet as a single length-4
cross-attention token, **unnormalized**, with `volfrac` ∈ [0.15, 0.45] beside
`rmin` ∈ [1, 3]. Normalizing before `encoder_hid_proj` is the first thing to try,
and it needs a retrain.

**The checkpoints are the current Euler runs.** Repos created 2026-08-06, still
being written. The new packages carry `design_min`/`design_max`; the legacy
NeurIPS checkpoints have no such keys, which is how you can tell them apart.
Euler's `ops/euler-sweeps` contains every diffusion and evaluation fix commit.

**Two bugs of my own, both silent.** The three LVAE adapters dropped `problem`
when building, so all 18 PLVAE packages were unloadable. And the pool scorer
appended per-row frames with per-row columns to one CSV, shifting metric columns
for any family whose config keys differed from the first row's — which read as
NaN for exactly `vqgan` and `diffusion_2d_cond`, got them dropped by `dropna`,
and let the selector report six-way disagreement over a board containing only
GANs. Both fixed; the corrupted board was discarded rather than patched.

## Adding a problem

Write `problems/<problem_id>.json`. Nothing else changes:

```json
{
  "problem_id": "photonics2d",
  "spec": "photonics2d/v1",
  "opening_metrics": ["mmd", "dpp", "viol"],
  "withheld_metrics": ["novelty", "cond_err", "pixel_vendi", "pca_mmd", "gen_seconds"],
  "expensive_metrics": ["iog", "cog", "fog"],
  "bank": [{"kind": "baseline", "algo": "knn_retrieval", "seed": 1}]
}
```

`unavailable_metrics()` is **derived from the frozen spec**, not declared — so on
photonics2d, whose spec sets `volume_condition` to null, the feasibility and
condition columns report themselves unavailable with a reason. A team
discovering mid-session that the column their neighbours are arguing about does
not exist for their problem is content, not a bug.

## Open questions

- **Bank not yet selected.** The pool rescore is running after the CSV fix;
  `tools/search_pool.py select` then ranks candidate banks by distinct rank-1
  winners, mean Kendall τ, and rank spread. The sealed board must be rebuilt
  afterwards — the committed one is for a superseded bank.
- **The slow families are under-sampled.** `vqgan` and `diffusion_2d_cond` are
  capped at 4 packages each because sampling costs 85–540 s per package, against
  46 apiece for the GANs. A bank chosen from that pool over-represents GANs
  unless selection is stratified, which it now is — but the underlying coverage
  is still uneven and `select` reports `families` per candidate so it cannot be
  missed.
- **kNN and linear regression are not on the Hub yet**, and not in the Euler
  manifest. Both train in seconds on CPU and need no GPU arm.
- **Flow matching**: `codex/flow-matching-model` has a complete `core.py`; the
  adapter is ~80 lines but there are no checkpoints, so it needs a training run.
- **No test covers `engiopt/baselines/`** since the old adversarial tests were
  removed with the models they asserted.
- **`tests/test_idetc26.py` takes 49 minutes** because opening a challenge
  downloads checkpoints. It needs the `network` marker the repo already declares.
