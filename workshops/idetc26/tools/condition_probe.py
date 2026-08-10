"""Does a model respond to the condition it is given at all?

`cond_err` says a model missed its volume-fraction target. It cannot say *why*.
A model that tracks the request with a constant offset and a model that ignores
the request entirely produce the same `cond_err`, and they are completely
different failures -- the first is a calibration bug, the second means the
conditioning pathway never learned anything.

This sweeps the requested volume fraction across its range while holding every
other condition and the sampling noise fixed, then regresses realized volume
fraction on requested. The slope is the answer:

    slope ~ 1     the model follows the request
    slope ~ 0     the model ignores it
    slope < 0     the conditioning is wired backwards

    python workshops/idetc26/tools/condition_probe.py --algos cgan_cnn_2d vqgan
"""

from __future__ import annotations

import argparse

import numpy as np
import torch as th

from engiopt.core import ConditionBatch
from engiopt.evaluation import Evaluator

VOLUME_CONDITION = "volfrac"
SWEEP_POINTS = 9

FOLLOWS_REQUEST = 0.5
"""Slope above which a model is tracking its condition rather than drifting with it."""

RESPONDS_WEAKLY = 0.15
"""Slope below which the conditioning pathway is doing essentially nothing."""


def probe(evaluator: Evaluator, generator, *, n_repeats: int = 4, seed: int = 0) -> dict[str, float]:
    """Sweep the volume-fraction request and measure how the output responds.

    Every other condition is held at the median of the evaluation set, so the
    only thing changing is the quantity being probed.

    Args:
        evaluator: Supplies the problem and the condition schema.
        generator: A loaded generator.
        n_repeats: Draws per sweep point, averaged to damp sampling noise.
        seed: Torch seed, so the same noise is reused across sweep points.

    Returns:
        `slope`, `intercept`, and `r2` of realized-on-requested volume fraction.
    """
    keys = list(evaluator.resolved.condition_keys)
    column = keys.index(VOLUME_CONDITION)

    base = evaluator.resolved.conditions_tensor.float()
    median = base.median(dim=0).values
    requested = np.linspace(float(base[:, column].min()), float(base[:, column].max()), SWEEP_POINTS)

    realized = []
    for target in requested:
        conditions = median.repeat(n_repeats, 1).clone()
        conditions[:, column] = float(target)
        th.manual_seed(seed)
        designs = generator.sample(ConditionBatch(tensor=conditions, keys=tuple(keys)), n=n_repeats)
        realized.append(float(np.asarray(designs).reshape(n_repeats, -1).mean()))

    slope, intercept = np.polyfit(requested, realized, 1)
    residual = np.asarray(realized) - (slope * requested + intercept)
    variance = np.var(realized)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(1 - residual.var() / variance) if variance > 0 else 0.0,
        "realized_range": float(max(realized) - min(realized)),
        "requested_range": float(requested.max() - requested.min()),
    }


def main() -> None:
    """Probe each requested model and print a table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-id", default="beams2d")
    parser.add_argument("--spec", default="beams2d/v1")
    parser.add_argument("--algos", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config-fingerprint", default=None)
    args = parser.parse_args()

    from engiopt.utils.all_generators import BUILTIN_GENERATORS

    evaluator = Evaluator.for_problem(args.problem_id, spec=args.spec)

    print(f"{'model':24s} {'slope':>8s} {'r2':>8s} {'out range':>10s} {'in range':>9s}   reading")
    for algo in args.algos:
        try:
            generator = BUILTIN_GENERATORS[algo].from_pretrained(
                evaluator.problem,
                problem_id=args.problem_id,
                seed=args.seed,
                model_source="hf",
                config_fingerprint=args.config_fingerprint,
            )
            result = probe(evaluator, generator)
        except Exception as exc:  # noqa: BLE001 - report and continue to the next model
            print(f"{algo:24s} FAILED {type(exc).__name__}: {str(exc)[:80]}")
            continue

        if result["slope"] > FOLLOWS_REQUEST:
            reading = "follows the request"
        elif result["slope"] > RESPONDS_WEAKLY:
            reading = "responds weakly"
        else:
            reading = "IGNORES the request"
        print(
            f"{algo:24s} {result['slope']:8.3f} {result['r2']:8.3f} "
            f"{result['realized_range']:10.4f} {result['requested_range']:9.4f}   {reading}"
        )


if __name__ == "__main__":
    main()
