# Adding a model to EngiOpt

EngiOpt keeps training and evaluation deliberately separate:

- **Training** stays a single self-contained script per model, in the CleanRL
  style. Write your algorithm however it reads best. There is no base class to
  inherit and no framework to learn.
- **Evaluation** is shared. You describe your model once, through the
  `Generator` contract, and every metric, leaderboard, and comparison works.

The contract is two methods and a handful of class attributes.

## The whole path at a glance

One directory, three files, four steps. The sections below explain each one.

| Step | What you do | What you get |
|---|---|---|
| Add your code | Copy `engiopt/generators/_template` to `engiopt/generators/my_model`. Write the training script and fill in the adapter | Your model is listed by `--list-generators`. No registry to edit |
| Publish checkpoints | Run the training script with `--save-model --hf-entity my-hf-username`, once per seed | One package per seed in your own HuggingFace repo |
| Generate designs | `MyModel.from_pretrained(problem, problem_id="beams2d", hf_entity="my-hf-username").sample(conditions)` | Designs, with the conditions prepared the way training recorded |
| Get a leaderboard row | `python -m engiopt.evaluate ... --include-expensive --push-to IDEALLab/engiopt-leaderboard` | Rows that appear at once, marked unverified |

```bash
cp -r engiopt/generators/_template engiopt/generators/my_model

for seed in 1 2 3; do
  python engiopt/generators/my_model/my_model.py \
    --problem-id beams2d --seed $seed --save-model --hf-entity my-hf-username
done

python -m engiopt.evaluate --problem-id beams2d --generators my_model --seeds 1 2 3 \
  --hf-entity my-hf-username --include-expensive --push-to IDEALLab/engiopt-leaderboard
```

Two accounts are involved before you start. Publishing needs a HuggingFace
account and a write token (`huggingface-cli login`). Training logs to Weights &
Biases by default, which needs a login of its own. Pass `--no-track` to train
without one.

Three steps still need a maintainer today. Your code has to be merged before a
row can be ranked, because the runner rebuilds your model from this repository.
`--push-to` needs write access to the board. A maintainer runs the verification.
[#78](https://github.com/IDEALLab/EngiOpt/issues/78) tracks opening those up.

## 1. Create the package

```bash
cp -r engiopt/generators/_template engiopt/generators/my_model
```

The directory name **is** your `algo_id`: it keys the registry, the leaderboard,
and your HuggingFace repo. Use lowercase with underscores.

The template is a working pair of a training script and a loader, plus an empty
`__init__.py`. If your model is not called `my_model`, the name has to change in
three places, and they must agree:

| Where | What to change |
|---|---|
| The training script's filename | `my_model.py` becomes `<your_name>.py`. The script names its checkpoint repository after its own filename |
| `algo_id` in `adapter.py` | `"my_model"` becomes `"<your_name>"` |
| The import at the top of `adapter.py` | It must point at `engiopt.generators.<your_name>.<your_name>` |

A mismatch is easy to miss, because the model is still listed. This test
catches it:

```bash
python -m pytest tests/test_generator_registry.py -q
```

## 2. Write the training script

Put it at `engiopt/generators/my_model/my_model.py`. The template directory
ships one under that name that is complete and small: replace its network and
its training step. `cgan_cnn_2d/cgan_cnn_2d.py` is a good short example of a
real model.

The only requirement is how you save. Use the shared checkpoint helper so your
model lands where the loader expects:

```python
from engiopt.checkpoint_store import save_checkpoint_package
from engiopt.core import checkpoint_identity

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
`{problem_id}/seed_{seed}`. That path is what the bare model name resolves to. A
sweep therefore can never redefine what your model means.

`run_config` matters: it is what `from_pretrained` reads to reconstruct your
network, so every architectural hyperparameter must be in `Args`.

`--save-model` is off by default, so a trial run uploads nothing. A run without
it leaves no package, and there is then nothing to evaluate.

### Record what training cost

The leaderboard reports what a model costs to train, and it can only report
what the training script recorded. Time your gradient steps with the shared
clock and pass its fields to the save call. That is four lines:

```python
from engiopt.train_clock import TrainClock

clock = TrainClock(device)
clock.start()
for step in range(args.n_steps):
    ...  # one gradient step
    if step % args.sample_interval == 0:
        clock.pause()  # logging, images and validation are not training
        ...
        clock.start()
clock.pause()

save_checkpoint_package(..., metadata=clock.metadata(n_train_steps=args.n_steps))
```

The clock records three things: the number of steps, the seconds spent in
them, and the name of the device. One pass of your training loop is one step,
however many networks it updates. The step count is the same on any machine.
The seconds are not, which is why the device is recorded beside them.

You write nothing for the other cost columns. The time to generate a design,
the optimizer time a generated design saves, and `designs_to_break_even` are
all measured by the evaluator, the same way for every model. A checkpoint that
recorded no training cost still evaluates; its `train_seconds`, `n_train_steps`
and `designs_to_break_even` cells are blank.

### How many conditions does your model have?

Not `len(problem.conditions_keys)`. That is the full contract, and it is wider
than a dense condition tensor can carry: thermoelastic2d declares seven
conditions, four of which are 65x65 boundary matrices, and photonics2d declares
solver settings that are not dataset columns at all. Size your network from the
*scalar* conditions instead:

```python
from engiopt.transforms import condition_keys

cond_keys = condition_keys(problem)  # e.g. ("volfrac", "rmin", "weight")
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
what makes the evaluator refuse, with a clear message, to feed your model
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

- **`build(resolved, problem, device, **base)`**: rebuild the network from
  `resolved.run_config` and `resolved.files`, then hand it to `cls(...)`.
  Fetching the package, choosing the device, and selecting the configuration
  are already done for you.
- **`_sample(conditions, n)`**: produce `n` designs. Seeding, timing, reshaping
  and clamping are handled for you; return whatever shape is natural.

```python
def __init__(self, net, latent_dim, **kwargs):
    super().__init__(**kwargs)  # the base class keeps the problem, device and checkpoint details
    self.net = net
    self.latent_dim = latent_dim


@classmethod
def build(cls, resolved, problem, device, **base):
    config = resolved.run_config
    net = MyNet(
        latent_dim=config["latent_dim"],
        n_conds=len(condition_keys_for(problem, resolved)),
        design_shape=design_shape_of(problem),
    ).to(device)
    state = th.load(resolved.files["generator.pth"], map_location=device, weights_only=True)
    net.load_state_dict(state[cls.primary_state_key])
    net.eval()
    return cls(net=net, latent_dim=config["latent_dim"], problem=problem, device=device, **base)
```

This is the code in the template's `adapter.py`, which runs as copied against
the template's training script. `condition_keys_for` and `design_shape_of` come
from `engiopt.core`.

Three names have to match between the two files, because one writes what the
other reads: the checkpoint filename (`generator.pth`), the key the weights are
saved under (`generator`), and the network's constructor arguments.

Exactly one `Generator` subclass per adapter module. EngiBench applies the same
rule to `Problem` classes, so the name-to-model mapping stays unambiguous.

### Conditions

`_sample` receives a `ConditionBatch`, not a bare tensor:

```python
cond = conditions.require_tensor(self.algo_id)  # (n, n_conds) on your device
conditions.dataset  # original columns, if you need them
conditions.keys  # condition names, in column order
```

The base class prepares that tensor before `_sample` is called. It selects the
columns your run recorded as `condition_keys`, in that order, and applies the
`condition_stats` your run recorded, if any. So `require_tensor` returns what
your network saw in training, and `_sample` does no scaling of its own. If you
recorded no `condition_stats`, the values arrive unscaled.

Most models want `require_tensor`. Reach for `dataset` only for conditions that
**cannot travel in the dense tensor at all**. thermoelastic2d's 65x65 boundary
matrices are one example, and `condition_keys` excludes them for exactly that reason.

Do **not** use it to re-derive preprocessing. Dropping constant columns or
re-fitting a normalizer against `dataset` at evaluation time changes the scale
your network sees, because the 50 evaluation rows have neither the training
split's statistics nor the same set of columns that never vary. Anything you
fitted during training travels with the weights instead, and is replayed on load:

```python
from engiopt.transforms import normalizer_state

save_checkpoint_package(
    ...,
    condition_stats={"mean": [...], "std": [...]},  # if you rescaled conditions
    condition_normalizer=normalizer_state(cond_norm),  # if you used a min/max Normalizer
    design_normalizer=normalizer_state(design_norm),
)
```

These are part of your model's identity: they are folded into the checkpoint
content hash, so changing a bound produces a different checkpoint rather than
inheriting the previous one's leaderboard row.

Unconditional models ignore the argument entirely, which is allowed. They are
still handed conditions so that `cond_sens` can measure it. See
[LEADERBOARD.md](LEADERBOARD.md).

## 4. Check it

```bash
python -m pytest tests/test_generator_registry.py -q      # the names agree
python -m engiopt.evaluate --list-generators              # your model should appear
python -m engiopt.evaluate --problem-id beams2d --generators my_model --hf-entity my-hf-username
```

The last command loads the checkpoint trained at your script's default
settings. A run with any setting changed, such as a short `--n-steps 50` trial,
is stored under its own configuration fingerprint and is not what the bare model
name resolves to. To evaluate such a run, name its fingerprint. The
package path in your HuggingFace repository shows it as `cfg_<fingerprint>`:

```bash
python -m engiopt.evaluate --problem-id beams2d --generators my_model \
  --config-fingerprints my_model:<fingerprint> --hf-entity my-hf-username
```

There is no way yet to check the loader against a checkpoint without publishing
it. The first test that `build` matches what the training script saved is this
evaluation, after an upload.

The default pass runs every metric that needs no simulator: the set-level
ones (`mmd`, `coverage`, `vendi`, `dpp`), the per-condition ones
(`per_condition_distance`, `volume_error`), feasibility (`viol`), the two
integrity checks (`train_distance`, `cond_sens`), and the cost
columns. `--include-expensive` adds the columns that re-optimize from each
generated design. Those are the optimality gaps `iog`, `cog` and `fog`, the
call-budget metrics `calls_to_near_optimum`, `gap_after_calls` and
`reaches_reference_rate`, and the cost column `designs_to_break_even`. They
run the optimizer and are slow. Leave them off while iterating. `python -m engiopt.evaluate
--list-metrics` prints the question each column answers.

Watch two columns while you develop:

- **`cond_sens`** should be greater than zero if you declared `conditional =
  True`. Exactly zero means your conditions are not reaching the network, which
  is a wiring bug far more often than a modeling choice.
- **`train_distance_ratio`** should be near one. Near zero means your model is
  reproducing training designs rather than generating; the board shows it and
  leaves the reading to people.

`cond_sens` draws the batch a second time, under shuffled conditions, so it
doubles sampling cost. That is nothing for a GAN and noticeable for a diffusion
model. Pass `--metrics mmd viol` while iterating if it slows you down, then
run the full list before publishing.

## 5. Publish it

See [LEADERBOARD.md](LEADERBOARD.md) for the full submission path. The short
version: train and evaluate under your own `--hf-entity`, run seeds 1, 2 and 3,
and push. Your rows appear immediately and are ranked once a runner has
re-fetched the weights and reproduced the scores.

## 6. Adding a metric instead

Metrics are registered functions, not model methods. A metric compares a
generated set against a reference set under a problem, so it belongs to the
comparison rather than to any one model:

```python
from engiopt.evaluation import register_metric


@register_metric("my_metric", family="diversity", cost="cheap", higher_is_better=True)
def my_metric(ctx) -> float:
    """What question does the number answer? This line is what readers see."""
    return ctx.reduce(...)  # one value per design -> the caller's aggregation (mean or median)
```

Five declarations: the name, the family (which question it belongs to), the
cost, the direction, which is `None` for a diagnostic that is read but never ranked on,
and, when the metric needs the actual designs rather than codes in some space
(a constraint check, a copy corpus), `pixel_only=True`. Declare
`cost="expensive"` if it touches `ctx.optimization` (the simulator or
optimizer); the registry enforces the split, so a cheap run can never
accidentally launch a simulation. Where the metric is computed (`space=`) and
how per-design values collapse to one number (`aggregation=`) are chosen when a
board is evaluated, not by the metric.

Importing the module is what registers it. You can therefore define a metric
in a notebook cell and it will appear in the next leaderboard you build.

## 7. Where things live

| Artifact | Home |
|---|---|
| Weights, `run_config.json`, `metadata.json` | HuggingFace model repo |
| Evaluation metrics (`metrics.json`) | HuggingFace, beside the weights |
| The leaderboard | HuggingFace dataset repo |
| Loss curves, sample images | Weights & Biases, when `--track` is on |

HuggingFace holds everything that must be reloaded or compared; W&B holds what
you only look at. Nothing in the evaluation path requires W&B. Training with
`--no-track` produces exactly the same checkpoints and scores.

To attach a model's scores to its own checkpoint:

```bash
python -m engiopt.evaluate --problem-id beams2d --generators my_model --attach-metrics
```

## 8. Evaluation specs

A leaderboard only means something if every row was measured the same way. Each
problem has committed specs (`engiopt/specs/<problem_id>/<version>.json`, the newest being the default) freezing the
test conditions, sample count, metric list, and kernel bandwidth. Every result
records the `spec_version` it came from.

Do not edit a published spec. If the protocol has to change, freeze a new
version:

```bash
python -m engiopt.evaluation.spec --problem-id my_problem --version v2
```
