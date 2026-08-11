# Adding a model to EngiOpt

EngiOpt keeps training and evaluation deliberately separate:

- **Training** stays a single self-contained script per model, in the CleanRL
  style. Write your algorithm however it reads best. There is no base class to
  inherit and no framework to learn.
- **Evaluation** is shared. You describe your model once, through the
  `Generator` contract, and every metric, leaderboard, and comparison works.

The contract is two methods and a handful of class attributes.

## 1. Create the package

```bash
cp -r engiopt/generators/_template engiopt/generators/my_model
```

The directory name **is** your `algo_id`: it keys the registry, the leaderboard,
and your HuggingFace repo. Use lowercase with underscores.

## 2. Write the training script

Put it at `engiopt/generators/my_model/my_model.py`. Follow any existing model
for the shape of it — `cgan_cnn_2d/cgan_cnn_2d.py` is a good short one.

The only requirement is how you save. Use the shared checkpoint helper so your
model lands where the loader expects:

```python
from engiopt.checkpoint_store import save_checkpoint_package

save_checkpoint_package(
    checkpoint_backend="hf",
    hf_entity=args.hf_entity,
    hf_repo_prefix=args.hf_repo_prefix,
    hf_private=False,
    problem_id=args.problem_id,
    algo=args.algo,  # must equal your algo_id
    seed=args.seed,
    checkpoint_files={"generator.pth": "generator.pth"},
    run_config=vars(args),  # everything needed to rebuild the model
    primary_files=["generator.pth"],
    condition_keys=cond_keys,  # the condition schema you trained against
    **checkpoint_identity(args),  # files this run under its own hyperparameters
)
```

`checkpoint_identity(args)` is what lets many hyperparameter settings coexist.
Each run is stored at `{problem_id}/cfg_{fingerprint}/seed_{seed}`, and a run
using your script's *default* hyperparameters additionally claims the canonical
`{problem_id}/seed_{seed}` -- which is what the bare model name resolves to. A
sweep therefore can never redefine what your model means.

`run_config` matters: it is what `from_pretrained` reads to reconstruct your
network, so every architectural hyperparameter must be in `Args`.

### How many conditions does your model have?

Not `len(problem.conditions_keys)`. That is the full contract, and it is wider
than a dense condition tensor can carry: thermoelastic2d declares seven
conditions, four of which are 65x65 boundary matrices, and photonics2d declares
solver settings that are not dataset columns at all. Size your network from the
*scalar* conditions instead:

```python
from engiopt.transforms import condition_keys

cond_keys = condition_keys(problem)  # e.g. ("volume_fraction_target", "rmin", "weight")
n_conds = len(cond_keys)
```

Use `cond_keys` for your dataloader columns too, and pass it to
`save_checkpoint_package` as above. Your adapter then rebuilds the network for
exactly the columns the run saw:

```python
from engiopt.core import condition_keys_for

n_conds = len(condition_keys_for(problem, resolved))
```

That is what keeps a checkpoint loadable after a problem gains a condition, and
what makes the evaluator refuse -- with a clear message -- to feed your model
conditions it was not trained on.

Also reuse the shared helpers rather than rewriting them:

| Need | Use |
|---|---|
| Seeding | `engiopt.reproducibility.seed_training` |
| Deterministic dataloading | `engiopt.reproducibility.make_dataloader_generator` |
| Device selection | `engiopt.core.pick_device` |
| Design shape (incl. dict spaces) | `engiopt.core.design_shape_of` |

## 3. Fill in the adapter

Edit `engiopt/generators/my_model/adapter.py`. Declare what your model is:

```python
class MyModel(Generator):
    algo_id = "my_model"  # == directory name
    conditional = True  # does _sample use the conditions?
    design_kinds = ("2d",)  # 1d / 2d / 3d / dict
    checkpoint_files = ("generator.pth",)
    primary_state_key = "generator"
    output_clip = (1e-3, 1.0)  # or None
```

Then implement the two methods:

- **`build(resolved, problem, device, **base)`** — rebuild the network from
  `resolved.run_config` and `resolved.files`, then hand it to `cls(...)`.
  Fetching the package, choosing the device, and selecting the configuration
  are already done for you.
- **`_sample(conditions, n)`** — produce `n` designs. Seeding, timing, reshaping
  and clamping are handled for you; return whatever shape is natural.

```python
@classmethod
def build(cls, resolved, problem, device, **base):
    config = resolved.run_config
    net = MyNet(latent_dim=config["latent_dim"], design_shape=problem.design_space.shape).to(device)
    net.load_state_dict(th.load(resolved.files["generator.pth"], map_location=device)[cls.primary_state_key])
    net.eval()
    return cls(net=net, latent_dim=config["latent_dim"], problem=problem, device=device, **base)
```

Exactly one `Generator` subclass per adapter module — the same rule EngiBench
applies to `Problem` classes, so the name-to-model mapping stays unambiguous.

### Conditions

`_sample` receives a `ConditionBatch`, not a bare tensor:

```python
cond = conditions.require_tensor(self.algo_id)  # (n, n_conds) on your device
conditions.dataset  # original columns, if you need them
conditions.keys  # condition names, in column order
```

Most models want `require_tensor`. Reach for `dataset` only for conditions that
**cannot travel in the dense tensor at all** — thermoelastic2d's 65x65 boundary
matrices, for instance, which `condition_keys` excludes for exactly that reason.

Do **not** use it to re-derive preprocessing. Dropping constant columns or
re-fitting a normalizer against `dataset` at evaluation time changes the scale
your network sees, because the 50 evaluation rows have neither the training
split's statistics nor the same set of columns that never vary. Anything you
fitted during training travels with the weights instead, and is replayed on load:

```python
save_checkpoint_package(
    ...,
    condition_stats={"mean": [...], "std": [...]},   # if you rescaled conditions
    condition_normalizer=normalizer_state(cond_norm),  # if you used a min/max Normalizer
    design_normalizer=normalizer_state(design_norm),
)
```

These are part of your model's identity: they are folded into the checkpoint
content hash, so changing a bound produces a different checkpoint rather than
inheriting the previous one's leaderboard row.

Unconditional models ignore the argument entirely, which is allowed. They are
still handed conditions so that `cond_sens` can measure it — see
[LEADERBOARD.md](LEADERBOARD.md).

## 4. Check it

```bash
python -m engiopt.evaluate --list-generators              # your model should appear
python -m engiopt.evaluate --problem-id beams2d --generators my_model --hf-entity my-hf-username
```

The default pass runs the cheap metrics: `mmd`, `dpp`, `viol`, and the two
integrity checks (`novelty`, `cond_sens`). Feasibility is among them because it
describes the design as generated, so it is a constraint check rather than a
solver run. `--include-expensive` adds the optimality gaps (`iog`, `cog`,
`fog`), which do run the optimizer and are slow — leave them off while
iterating.

Watch two columns while you develop:

- **`cond_sens`** should be greater than zero if you declared `conditional =
  True`. Exactly zero means your conditions are not reaching the network, which
  is a wiring bug far more often than a modelling choice.
- **`copy_rate`** should be near zero. High means your model is reproducing
  training designs rather than generating, and the board will publish it without
  ranking it.

## 5. Publish it

See [LEADERBOARD.md](LEADERBOARD.md) for the full submission path. The short
version: train and evaluate under your own `--hf-entity`, run seeds 1, 2 and 3,
and push. Your rows appear immediately and are ranked once a runner has
re-fetched the weights and reproduced the scores.

## 6. Adding a metric instead

Metrics are registered functions, not model methods — a metric compares a
generated set against a reference set under a problem, so it belongs to the
comparison rather than to any one model:

```python
from engiopt.evaluation import register_metric


@register_metric("my_metric", family="diversity", cost="cheap", higher_is_better=True)
def my_metric(ctx) -> float:
    """One line, shown by --list-metrics."""
    return float(...)  # ctx.gen_flat, ctx.ref_flat, ctx.conditions, ...
```

Declare `cost="expensive"` if it touches `ctx.optimization` (the simulator or
optimizer). The registry enforces the split, so a cheap run can never
accidentally launch a simulation.

Importing the module is what registers it — which means you can define a metric
in a notebook cell and it will appear in the next leaderboard you build.

## 7. Where things live

| Artifact | Home |
|---|---|
| Weights, `run_config.json`, `metadata.json` | HuggingFace model repo |
| Evaluation metrics (`metrics.json`) | HuggingFace, beside the weights |
| The leaderboard | HuggingFace dataset repo |
| Loss curves, sample images | Weights & Biases, when `--track` is on |

HuggingFace holds everything that must be reloaded or compared; W&B holds what
you only look at. Nothing in the evaluation path requires W&B -- training with
`--track false` produces exactly the same checkpoints and scores.

To attach a model's scores to its own checkpoint:

```bash
python -m engiopt.evaluate --problem-id beams2d --generators my_model --attach-metrics
```

## 8. Evaluation specs

A leaderboard only means something if every row was measured the same way. Each
problem has a committed spec (`engiopt/specs/<problem_id>/v1.json`) freezing the
test conditions, sample count, metric list, and kernel bandwidth. Every result
records the `spec_version` it came from.

Do not edit a published spec. If the protocol has to change, freeze a new
version:

```bash
python -m engiopt.evaluation.spec --problem-id my_problem --version v2
```
