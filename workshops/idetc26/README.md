# IDETC'26 — Find the Best Model

A hands-on workshop about what you would have to measure before you were
entitled to say one generative model is better than another.

Teams get a bank of eight anonymized models for one engineering design problem,
three cheap metrics, and a commitment cell. They pick a winner and defend it.
Then the ranking comes apart — first under a change of random seed, then under
the cheap columns they were not given, then under the physics, and finally under
the disclosure that most of the bank was never trained at all.

**Status: working end to end on `beams2d` with a locally-staged bank.** See
[Open questions](#open-questions) for what is not done.

---

## Run it

```bash
python workshops/idetc26/build/stage_local_checkpoints.py     # once, until HF has the checkpoints
jupyter lab workshops/idetc26/notebooks/01_find_the_best_model.ipynb
```

In Colab the notebook installs EngiOpt from a branch and needs no staging — but
the two trained models are skipped until the sweep publishes them (see below).

## What is in the box

```
workshops/idetc26/
  notebooks/01_find_the_best_model.ipynb   the session, generated from build/
  problems/beams2d.json                    the parameterization unit
  sealed/beams2d_physics.csv.enc           the answer, preregistered
  sealed/beams2d_physics.csv.enc.sha256    its published digest
  build/build_notebook.py                  notebook source -- edit here, not the .ipynb
  build/build_sealed_board.py              computes and seals the physics board
  build/stage_local_checkpoints.py         temporary; delete once the sweep publishes

engiopt/adversarial/                       the constructed models
engiopt/workshops/idetc26/                 bank, challenge, config, sealing
tests/test_adversarial.py                  asserts every trap actually holds
```

## The bank

Six models built to exploit a specific metric, and however many real trained
checkpoints are available. They subclass the same `Generator` contract and are
scored by the same evaluator, so nothing about how they load distinguishes them.

| model | wins | loses | the gap it exposes |
|---|---|---|---|
| `regurgitator` | `mmd`, `pca_mmd`, `viol`, `cond_err` | `novelty` | MMD is *minimized* by copying the training set |
| `one_trick_pony` | `gen_seconds` | `pixel_vendi`, `cond_err` | nothing checks the model read its input |
| `volume_cheater` | `viol`, `cond_err`, `novelty` | `iog`/`cog`/`fog` | feasibility is a floor, not evidence of quality |
| `blurry_regressor` | `gen_seconds`, `params` | everything physical | the honest naive baseline scores better than it should |
| `noise_doped` | `dpp`, `pixel_vendi` | `mmd`, `viol` | pixel diversity *rewards* corruption |
| `checkerboard` | `mmd`, `cond_err` | the eyeball test | looks broken, scores fine |

None of them needed training. All six together are under 200 lines. Each carries
exactly one point, so a team whose ranking is overturned can see precisely which
assumption did it.

`tests/test_adversarial.py` asserts each claim in that table against a live
board, which means the traps are verified before any expensive metric exists and
they cannot quietly stop working.

## Adding a problem

Write `problems/<problem_id>.json`. Nothing else changes:

```json
{
  "problem_id": "photonics2d",
  "spec": "photonics2d/v1",
  "opening_metrics": ["mmd", "dpp", "viol"],
  "withheld_metrics": ["novelty", "cond_err", "pixel_vendi", "pca_mmd", "gen_seconds"],
  "expensive_metrics": ["iog", "cog", "fog"],
  "bank": [{"kind": "constructed", "algo": "regurgitator"}]
}
```

`unavailable_metrics()` is *derived from the frozen spec*, not declared — so on
photonics2d, whose spec sets `volume_condition` to null, the feasibility and
condition columns report themselves as unavailable with a reason. A team
discovering mid-session that the column their neighbours are arguing about does
not exist for their problem is content, not a bug.

## Measured costs

Numbers from an actual run on a laptop CPU, not estimates:

| | cost |
|---|---|
| the cheap board, 8 models × 50 samples | **under 30 s total** |
| the physics board, per model × 50 samples | **~90–115 s** |
| the whole physics board, 8 models | **~15 min** |

This matters for planning. The project's working assumption was ~16 GPU-hours,
which came from SLURM *requested* walltime and had never been measured. The real
figure is two orders of magnitude smaller, and `beams2d`'s optimizer is CPU-bound
topology optimization — **a GPU does not help it.** A Colab GPU speeds up
sampling from the trained models and nothing else.

The practical consequence: the physics board is affordable enough that teams
could compute part of it live, which is a different workshop from the one where
it has to be sealed.

## Sealing

The physics board ships encrypted with its plaintext SHA256 committed beside it.
The digest is content, not security — it lets participants verify afterwards
that the answer was fixed before theirs was, which is the practice the session
argues for. `unseal_physics()` refuses to run until a verdict exists.

```bash
python workshops/idetc26/build/build_sealed_board.py \
    --problem-id beams2d --passphrase "..." --plaintext-out /tmp/board.csv
```

The plaintext copy is for the facilitator. Do not commit it.

## What the current bank actually shows

The full board on `beams2d`, seed 1, identities revealed:

| model | mmd↓ | viol↓ | novelty↑ | cond_err↓ | pixel_vendi↑ | gen_s↓ | iog↓ | fog↓ |
|---|---|---|---|---|---|---|---|---|
| `regurgitator` | **0.011** | **0.00** | **3.29** | **0.0006** | 30.3 | 0.030 | **1.7** | -2.43 |
| `one_trick_pony` | 0.967 | 0.94 | 4.01 | 0.070 | 1.0 | **0.0001** | 6.4e8 | 41.96 |
| `volume_cheater` | 0.112 | **0.00** | **35.0** | 0.0007 | **50.0** | 0.009 | 9.4e9 | 31.42 |
| `blurry_regressor` | 0.158 | 0.38 | 11.3 | 0.010 | 7.7 | 0.011 | 265 | **-3.52** |
| `noise_doped` | 0.053 | 0.96 | 13.6 | 0.047 | 47.7 | 0.002 | 214 | -2.72 |
| `checkerboard` | 0.062 | 0.80 | 9.64 | 0.029 | 30.1 | 0.002 | 90 | -2.95 |
| `cgan_cnn_2d` | 0.037 | 0.44 | 8.15 | 0.012 | 26.5 | 0.090 | 5.1e7 | -1.27 |
| `gan_cnn_2d` | 0.097 | 0.94 | 12.3 | 0.076 | 28.0 | 0.023 | 1.2e8 | 149.07 |

**Three results the plan did not anticipate.**

**1. The lookup table wins the physics too.** `PLAN.md`'s curation gate asks for
*cheap winner ≠ expensive winner*. It is not met: `regurgitator` tops `mmd`,
`pca_mmd`, `viol`, `cond_err`, `iog` **and** `cog`. That is not a bug — its
designs are genuinely optimal, because they are real optimal designs. Its
novelty score is the worst in the bank (3.29, rank 8 of 8), and **`novelty` is
the only column in eleven that notices.**

This is a sharper lesson than the one the plan was built around, but it is a
*different* one, and the session has to be re-cut for it. Either:

- **Keep it**, and make the arc "your whole board, physics included, can be
  topped by a fifty-line lookup table, and exactly one cheap column catches it";
- **or weaken the regurgitator** — retrieve from a held-out split, or add
  jitter — so the physics reveal overturns the cheap ranking as originally
  designed.

That is a pedagogy decision, not a code one. It should be made deliberately.

**2. The seed lottery barely fires.** Across seeds 1–3, `mmd` ranks do not move
at all and `viol` moves for two models. Most of the bank is retrieval-based and
therefore near-deterministic, so there is nothing for a seed to change. The
reveal-1 segment currently rests on the two stochastic trained models — and
there are only two. **It needs a larger trained field to work as designed.**

**3. `fog` disagrees with `iog`.** Final optimality gap crowns
`blurry_regressor`; initial optimality gap crowns `regurgitator`. Two metrics
from the same family, computed in the same pass, name different winners. That
disagreement is real and is the most defensible version of the session's thesis,
because neither number is cheap and neither is a proxy.

Two things worth flagging to whoever owns the evaluation layer:

- **The trained models score 1e7–1e8 on `iog`** while constructed ones score
  1e0–1e2. Plausible — both trained models generate infeasible designs (`viol`
  0.44 and 0.94) and the optimizer diverges from an infeasible start — but a gap
  metric that spans nine orders of magnitude on the same problem is hard to rank
  on, and worth a look.
- **`params` returns NaN for a model with no parameters**, because
  `_count_parameters` cannot distinguish "holds no `nn.Module`" from "unknown".
  A constructed model has zero trainable parameters, and reporting that as zero
  would let the cost family say "this beat your GAN with no parameters at all."

## Open questions

- **The trained field is two old checkpoints, staged from `artifacts/`.** They
  load and score sensibly, but they are W&B-era artifacts repackaged by
  `stage_local_checkpoints.py`, and in Colab they are skipped entirely because
  the HF model repos do not exist yet. The Euler sweep is producing the real
  pool; when it publishes, delete the staging script and drop the
  `local_model_dir` keys from the config.
- **The bank is six frauds and two real models.** That ratio is wrong for the
  session — the plan calls for a healthy majority of legitimate models — and it
  can only be fixed once the pool exists.
- **`cond_sensitivity` is not implemented.** Sweeping a condition at fixed *z*
  is a generator-level operation rather than a context-level one, so it does not
  fit the current metric signature. `cond_err` carries the conditions family for
  now.
- **The distortion ladder is only two rungs** (`noise_doped`, `checkerboard`).
  The graded severity sweep shared with the LV-metrics paper is not built.
- **This is one notebook, not the five-notebook arc** in `PLAN.md`. The reveal
  ladder is folded into it, gated on the verdict.
