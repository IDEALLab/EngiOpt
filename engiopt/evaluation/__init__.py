"""Evaluating generators and building leaderboards.

    from engiopt.evaluation import Evaluator
    from engiopt.utils.all_generators import BUILTIN_GENERATORS

    ev = Evaluator.for_problem("beams2d", spec="beams2d/v1")
    gen = BUILTIN_GENERATORS["cgan_cnn_2d"].from_pretrained(ev.problem, problem_id="beams2d", seed=1)
    ev.score(gen)

Metrics are registered functions rather than model methods, because a metric
compares a generated set against a reference set under a problem -- it is a
property of the comparison, not of the model.
"""

from engiopt.evaluation.context import EvaluationContext
from engiopt.evaluation.context import OptimizationResults
from engiopt.evaluation.evaluator import Evaluator
from engiopt.evaluation.leaderboard import already_evaluated
from engiopt.evaluation.leaderboard import append_rows
from engiopt.evaluation.leaderboard import disagreement
from engiopt.evaluation.leaderboard import load_from_hub
from engiopt.evaluation.leaderboard import merge_rows
from engiopt.evaluation.leaderboard import push_to_hub
from engiopt.evaluation.leaderboard import rank
from engiopt.evaluation.registry import MetricRegistry
from engiopt.evaluation.registry import METRICS
from engiopt.evaluation.registry import MetricSpec
from engiopt.evaluation.registry import register_metric
from engiopt.evaluation.spec import EvalSpec
from engiopt.evaluation.spec import ResolvedSpec

__all__ = [
    "METRICS",
    "EvalSpec",
    "EvaluationContext",
    "Evaluator",
    "MetricRegistry",
    "MetricSpec",
    "OptimizationResults",
    "ResolvedSpec",
    "already_evaluated",
    "append_rows",
    "disagreement",
    "load_from_hub",
    "merge_rows",
    "push_to_hub",
    "rank",
    "register_metric",
]
