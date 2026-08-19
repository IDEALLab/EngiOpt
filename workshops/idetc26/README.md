# IDETC'26 — Find the Best Model

A hands-on workshop about what you would have to measure before you were
entitled to say one generative model is better than another.

Teams get a bank of models for one engineering design problem — a lookup table,
a supervised deconvolutional network, GANs, VQGANs, a diffusion model, an
autoencoder, all named for what they are — and every cheap instrument the benchmark has. Then they find
that the columns disagree, that they disagree differently depending on the space
they measure in, and that the ranking moves when only the random seed changes.

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
cannot. Both sides of that paper's comparison are in the bank as **competitive
entries**, not as jokes: `knn_retrieval` at k=1 is the retrieval method, and
`deconv_regression` is the deconvolutional network it was measured against.

On `beams2d` the kNN baseline currently beats the conditional GAN on condition
adherence, at 1/1600th of VQGAN's sampling cost, and the two are
within 5% of each other on MMD. A leaderboard topped by retrieval is a result,
not a broken leaderboard.

## Run it

```bash
jupyter lab workshops/idetc26/notebooks/01_find_the_best_model.ipynb
```

Colab installs EngiOpt from the branch named by the single `BRANCH` constant in
`tools/build_notebook.py`. A GPU speeds up sampling from the trained models; it
does **not** speed up the physics, which is CPU-bound topology optimization.

## The notebook runs in one direction

Ten suspects are in the room, each claiming to be a good generative model for
the same problem, each handed the same 50 design briefs. One of them is the
best. Six sections, and each is about the same objects as the one before it, so
nothing needs re-introducing:

**1 · the data** → **2 · the suspects**, looked at, and a ranking written down
before any number → **3 · the questions**, one family at a time, several of
which contradict each other → **4 · the same questions in a fitted space**,
which is the pixel columns again with a projection in front of them →
**5 · the board**, the participant's own columns against the published simulator
run → **6 · free work**, with `case.help()` as the reference. The accusation at
the end is spoken aloud, not recorded by code.

That is a deliberate reversal of the earlier "toolbox with no order" build. A
catalogue of independent tools is a reference manual, and nobody learns a
subject from a reference manual in ninety minutes. The order is what gives a
participant a reason to run the next cell, and it gives the disagreement between
metrics somewhere to land.

**The prose is short on purpose.** Sentence fragments where a fragment says it,
full sentences only where a method is being explained, and no justification of
the API's design — that is what this README is for. The audience is researchers;
a cell that argues with them about why it exists is a cell they skim.

**There are two commands.** `case.evaluate` for numbers, `case.show` for
pictures:

```python
case = Case.open("beams2d")

case.help()                         # the cheat sheet: five calls, then every knob
case.models()                       # who is in the line-up
case.metrics()                      # what may be asked, by line of questioning

case.evaluate("diversity")          # one whole line of questioning
case.evaluate("mmd", models="knn")  # one question, one suspect
case.evaluate(["cost", "similarity"], controls=True)

case.show("diffusion")              # its designs
case.show("knn", "diffusion")       # two suspects on the same brief
case.show(answers)                  # a table of answers, drawn as ranks
case.show(answers, physics)         # two boards joined and drawn as one
case.show("knn", how="nearest_training")   # each design beside its closest training design
case.show("vqgan", "test", how="space_map")   # where its designs sit in a fitted space
```

**Neither command has a blanket form.** `case.evaluate()` and `case.show()` both
raise, and the omission is the pedagogy: a board of every cheap column against
every suspect is read rather than argued with, and a contact sheet of all ten
line-ups is skimmed. Which three columns a person chose is the session, so
naming the question is the work. Both errors say which call to make instead — a
bare `TypeError` in front of ninety people is a room full of raised hands.

**`case.help()` is the cheat sheet**, run as the sixth cell and pointed at again
in the free-work section. It opens with the five calls that matter — `models`,
`metrics`, `explain`, `evaluate`, `show` — and everything verbose sits below
that: the views, the latent map, the controls, every knob with its default. The
notebook documents none of it. Markdown argument tables went stale every time an
argument moved and made each section twice as long as the idea in it; a printed
sheet beside the code cannot.

The previous build had `compute`, `evaluate`, `board`, `score`, `run_physics`,
`seed_stability`, `reference_row`, `rank` and `winners` — nine ways to get a
number, two of which had names that meant the same thing in English. They are
one function now, with the differences expressed as arguments. A participant who
has to work out which of `compute` and `evaluate` they wanted is spending their
attention on the API rather than on the argument.

The interface is **simple by default and complete underneath**: `evaluate("mmd")`
is one word, and `evaluate("mmd", sigma=0.5, n_samples=10, random_conditions=True)`
argues with the kernel bandwidth and the comparison set for anyone who wants to.

**Suspects are named for what they are**, from the first cell — `knn_retrieval`,
`diffusion_2d_cond`, `cgan_cnn_2d` — and reachable by any unambiguous fragment
(`"diffusion"`, `"knn"`). The anonymized `Model A`/`Model B` framing is gone:
"why is the lookup table beating the diffusion model" is the question worth
having, and it cannot be asked of `Model C`.

**No two suspects differ only by training seed.** A seed is not a model, and a
line-up containing seed-pairs invites a ranking that separates them — which is a
ranking of the random number generator.

**And there is no `seeds=` argument**, deliberately. The spec freezes the 50
briefs (`EvalSpec.condition_seed`), so the only thing a seed could vary in
`evaluate` is the *sampling noise* on a fixed checkpoint — a far weaker question
than it appears, offered under a name that invited it to be mistaken for the
strong one. The question with teeth needs checkpoints at several **training**
seeds. The Hub has them (seeds 1–10 for each family's replicated configuration);
pulling them is the natural next version of the session.

### Six lines of questioning, and a seventh that is not about the models

Nobody arrives knowing what `pca_coverage` is. They can hold six questions in
their head, so every column the benchmark computes is filed under one, and
naming a family is as real a call as naming a metric:

| | asks | satisfied perfectly by |
|---|---|---|
| `cost` | what did it take to put this model in the room? | — |
| `similarity` | do its designs look like the real ones? | handing back the training set |
| `memorization` | is it inventing, or copying out of the case files? | random noise |
| `diversity` | has it more than one answer, or one story it repeats? | corruption |
| `obedience` | did it answer the question it was actually asked? | material that carries no load |
| `performance` | are the designs actually any good? | *(needs the simulator)* |

The seventh, `latent_space`, holds `lv_dual_gap`, `lv_active_dims` and
`pca_dims` — diagnostics of the fitted space the `lv_` columns are measured in,
never ranked, because they are not properties of a model. It was called
`instrument` and the accessor beside it is `case.latent_space()`; both are
`latent_space` now, since "instrument" told a participant nothing about which
part of the suite it belonged to. `case.latent_space()` names the autoencoder,
its measured active width, and the PCA width matched to it.

The right-hand column is the point: every family has a way of being satisfied by
a model that is obviously bad, which is why no single one settles the argument.

**These are not the registry's own families.** The registry files `novelty`
under `distribution`, beside `mmd` — but `mmd` is *minimized* by copying the
training set and `novelty` is the column that catches you doing it. Filing them
together is precisely what lets a board look coherent while containing its own
refutation, so `memorization` is split out. Likewise the registry files every
`lv_*` column under `latent`, describing what it depends on rather than what it
asks; here `lv_mmd` joins `mmd`, and the **space** it measures in becomes its own
column of the catalogue:

| the question | pixels | PCA subspace | learned latent |
|---|---|---|---|
| does this look like the real data? | `mmd` | `pca_mmd` | `lv_mmd` |
| is it copying the training set? | `novelty_ratio` | — | `lv_novelty` |
| did it cover the real modes? | — | `pca_coverage` | `lv_coverage` |
| how many distinct designs? | `pixel_vendi`, `dpp_geometric` | `pca_vendi` | `lv_vendi` |
| did it answer the brief? | `pixel_paired_distance` | — | `lv_paired_distance` |

Those rows disagree, and the disagreement is a fact about the spaces rather than
about the suspects.

Nobody writes plotting code. `case.show` dispatches on what it is handed — a
suspect name, two names, a board, a board and two of its columns — with the
specialist views behind `how=`: `designs`, `compare`, `conditions`,
`nearest_training`, `space_map`.

`space_map` is the only one that draws a *distribution* rather than designs: a
suspect's 50 designs as 50 points in the pinned latent space or the matched PCA
subspace, over the training set as a backdrop graded by how good each design is.
It is the shape that `mmd` reduces to one number, and collapse, near-miss and
missing coverage are all legible in it and none of them are legible in the
value. Drawing the same model in `space="lv"` and `space="pca"` is the cheapest
demonstration that a distribution metric is a claim about a space.

`case.evaluate` is `python -m engiopt.evaluate` in a notebook: same evaluator,
same frozen spec, same argument names, and it prints the CLI line that produces
the same numbers, so the skill transfers out of the notebook. Metrics are never
computed on a private code path.

The physics is **not a separate reveal** — it is the seventh line of questioning,
asked the same way as the other six. Ask it and the estimated time is printed
before the run starts, so a run that is longer than somebody wanted gets
interrupted like any other cell. The published board (`case.physics()`) is there
because nobody can run the simulator on 50 designs during a session.

## Nothing waits on sampling

Drawing the spec's 50 conditions from every suspect — one of them a diffusion model
on a CPU runtime — used to be the first thing a room did, in silence, twice if
anyone restarted the kernel. Those designs are a fixture: identical for every
team, every time. So they are precomputed and shipped inside the package
(`engiopt/workshops/idetc26/cache/`), and looking at a model costs nothing.

```bash
# rebuild after the bank changes -- a cluster/workstation job, not a laptop one
python workshops/idetc26/tools/build_design_cache.py --problem-id beams2d --seeds 1 2 3
```

Costs that are worth feeling are still paid in full: `case.evaluate(fresh=True)`
resamples from the checkpoints and times it, and the simulator costs what it
costs. A replayed `gen_seconds` says which machine measured it, since wall-clock
only compares models timed on the same one.

A cached bank member also does not download its checkpoint at `Case.open`,
so opening the workshop no longer pulls a 1.8 GB VQGAN before showing anybody
anything.

## What is in the box

```
workshops/idetc26/
  notebooks/01_find_the_best_model.ipynb   the session (a build product)
  tools/build_notebook.py                  notebook SOURCE -- edit here, not the .ipynb
  tools/build_design_cache.py              pre-sample the bank so the session never waits
  tools/search_pool.py                     score the pool, then search it for a bank
  tools/condition_probe.py                 does a model respond to its condition at all?
  tools/build_sealed_board.py              compute and seal the physics board
  tools/verify_notebook.py                 execute every cell, fail on the first error

engiopt/workshops/idetc26/                 bank, challenge, config, views, sealing
  problems/beams2d.json                    the parameterization unit
  sealed/                                  the preregistered physics board
  cache/                                   pre-sampled designs, shipped
engiopt/baselines/                         dataset-fitted models (see below)
engiopt/generators/knn_retrieval/          kNN as a first-class generator package
engiopt/generators/linear_regression/      ridge regression, likewise
tests/test_idetc26.py                      harness verification
```

The directory is `tools/`, not `build/`, because the root `.gitignore` ignores
`build/` and silently dropped every script in it from a commit.

**The data lives inside the package, not beside the notebooks.** Colab runs
`pip install git+...`, which ships `engiopt` and nothing else — no `workshops/`
tree. Problem configs, the sealed board and the design cache were unreachable
from a Colab runtime while they sat at the repository root, and are now package
data (`[tool.setuptools.package-data]` carries `*.json`, `sealed/*.enc` and
`cache/**/*.npz`). A source checkout still works: both locations are searched,
package first.

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

## The baselines are published like any other model

`knn_retrieval` and `deconv_regression` are full generator packages with
training scripts that write checkpoints through `save_checkpoint_package`. They
are not notebook-only conveniences. (`linear_regression` is one too and remains
in the repo; it was dropped from the bank in favour of the deconvolutional
network, which is the comparator the paper actually used.)

| | `knn_retrieval` | `deconv_regression` | `diffusion_2d_cond` | `vqgan` |
|---|---|---|---|---|
| checkpoint | 38.9 MB | **14.8 MB** | ~70 MB | ~1.8 GB |
| parameters | its training set | 3.7 M | 17.5 M | 102 M |
| 50 samples | 0.054 s | one forward pass | 452 s | 85 s |

Every one reports a real parameter count, so a kNN's data cost and a network's
weight cost land on the same axis. The checkpoint sizes side by side are a
better answer to *what is a model?* than any definition.

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
photonics2d, whose spec sets `volume_condition` to null, `viol` and the other
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
