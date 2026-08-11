# The EngiOpt leaderboard

A public board has two problems an internal one does not, and they are
independent. Someone can report numbers no model produced, and someone can
report perfectly honest numbers from a model that games the metric instead of
solving the problem. Neither defence helps with the other.

- Against **fabrication**: re-execution. Nothing is ranked until a runner has
  fetched the weights itself and reproduced the score.
- Against **gaming**: measurement. The scoring protocol is public and cannot be
  un-published, so the board measures the exploits it cannot prevent and
  declines to rank what trips them.

## Submitting

Nothing here needs write access to `IDEALLab`. Publish your checkpoints to your
own HuggingFace account:

```bash
python engiopt/generators/my_model/my_model.py \
  --problem-id beams2d --seed 1 --hf-entity my-hf-username

python -m engiopt.evaluate \
  --problem-id beams2d --generators my_model --seeds 1 2 3 \
  --hf-entity my-hf-username \
  --include-expensive \
  --push-to IDEALLab/engiopt-leaderboard
```

Your rows land immediately, marked `verified=false`. They are visible on the
board and absent from the ranking until a runner re-scores them.

Three things are worth knowing before you run it.

**Your checkpoints must be on the Hub.** Every row records the repo, path,
revision, and content hash of the weights it was scored on, and a row without
that address is refused at publish time — not because it is untrusted, but
because nobody, including you, could ever re-run it. Evaluating with
`--model-source local` produces exactly such a row.

**Run the seeds the spec asks for.** `EvalSpec.required_seeds` is `(1, 2, 3)`,
and an entry missing any of them is published but not ranked. This is the
cherry-picking rule: a count would still let you run twenty seeds and publish
your best three, which turns a median into a maximum while every individual row
stays honest. Naming the seeds removes the choice.

**You cannot mark your own row verified.** `push_to_hub` clears the column.

## Verification

```bash
# audit anyone's rows; needs no credentials and writes nothing
python -m engiopt.verify --board IDEALLab/engiopt-leaderboard

# the official runner
python -m engiopt.verify --board IDEALLab/engiopt-leaderboard \
  --verifier ideallab-ci --include-expensive --publish
```

Verification fetches the package **at the revision the row recorded**, checks
that it still hashes to what was scored, rebuilds the model through its
registered adapter, and scores it again. What gets published is the runner's own
numbers, not a verdict on yours.

That distinction is deliberate. Sampling from one seed on different hardware
genuinely produces different designs — CUDA and CPU draw different values from
the same generator state — so a pass/fail comparison would need a tolerance loose
enough to wave through real fudging. Re-scoring sidesteps it: the runner's number
is the number, and yours becomes a claim reported as corroborated or not.

Four outcomes:

| Status | Meaning |
|---|---|
| `verified` | Fetched, re-scored. The row now carries the runner's numbers. |
| `weights_moved` | The package no longer hashes to what the row recorded. Not re-scored — whatever is there now is not what earned the score. |
| `unresolvable` | The address led nowhere, or the package would not load. |
| `unknown_generator` | No registered adapter can rebuild the model. |

The runner is not privileged. It is the first auditor, and because every row
carries a full address, the same command run by anyone else produces the same
answer — which is what keeps the runner honest too.

## What gets ranked

A row is published if it can be checked, and ranked if checking it went well.
Those are different bars on purpose: deleting a submission is moderation, and
declining to rank one is a statement about what its number measures. Only the
second scales.

To be ranked, a row must be `verified`, carry no disqualifying flag, and belong
to an entry covering every required seed.

| Flag | Trips when |
|---|---|
| `unverified` | No runner has reproduced it yet. |
| `memorized` | `copy_rate` exceeds the spec's `max_copy_rate`. |
| `ignores_conditions` | `cond_sens` is exactly zero for a model registered as conditional. |

## The copying problem

This is the one worth understanding before reading any number on the board.

The spec is public. It names the condition seed, the sample count, and a digest
of the drawn conditions, so anyone can recompute exactly which 50 test rows are
scored — and the dataset supplies the optimal design for each of them. A lookup
table keyed on the condition vector returns those designs and therefore posts a
perfect `mmd` and a near-zero `iog` and `fog`.

That is not an implementation flaw. Those metrics are *defined* as closeness to
the reference designs, and a retrieval system is closest. No amount of care in
the evaluator changes it, and it cannot be fixed by hiding which rows are
scored: the whole dataset is public, so a memorizer memorizes all of it.

So the board measures retrieval instead:

- **`novelty`** — mean per-element RMS distance from each generated design to the
  nearest design the model could have copied: the training split it was fitted
  on, plus the reference designs the protocol names.
- **`copy_rate`** — the share of the batch closer than `copy_tol`. This is the
  one that flags an entry.

Both are diagnostic, with no ranking direction, and that is not an oversight.
Ranking on novelty would put pure noise in first place — zero means retrieval,
but large means only "unlike the data", which a broken model also achieves.
Closing one gaming vector by opening another is not progress. The same reasoning
applies to `cond_sens`: an unconditional model is a legitimate thing to build,
and responding to conditions *wrongly* also moves the output, so a large value is
not by itself a good one.

Read them next to `mmd` and `viol`, never on their own.

### What would actually close it

Scoring on conditions that have no published solution. A v2 spec would draw
conditions from the condition space rather than from dataset rows, and carry the
reference *objective* for each — computed once by whoever freezes the spec, with
no design released. A lookup table then has nothing to look up, and `mmd` moves
to comparing against the training distribution rather than against 50 named
designs.

That is the right benchmark for a generative design model, and it is not built
here: freezing such a spec means an optimizer run per condition. The mechanism
this PR adds — flags, eligibility, and a spec version that carries its own
thresholds — is what a v2 would plug into.

## Reading the board

```python
import pandas as pd
from engiopt.evaluation.leaderboard import disagreement, load_from_hub, rank
from engiopt.evaluation.spec import EvalSpec

board = load_from_hub("IDEALLab/engiopt-leaderboard")
spec = EvalSpec.load("beams2d/v1")

rank(board, "fog", eval_spec=spec)                    # the public ordering
rank(board, "fog", eval_spec=spec, eligible_only=False)  # everything, including claims
disagreement(board, ["mmd", "dpp", "fog"], eval_spec=spec)
```

`rank` refuses a diagnostic metric outright rather than inventing a direction
for it. `disagreement` is the view worth looking at first: where the metrics
disagree about the winner is the only place the board is telling you something a
single number could not.
