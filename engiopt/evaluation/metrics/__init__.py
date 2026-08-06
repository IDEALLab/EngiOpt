"""Metric implementations.

Importing this package registers every built-in metric in
`engiopt.evaluation.registry.METRICS`. Third-party metrics register themselves
the same way: decorate a function with `@register_metric` and import the module.
"""

from engiopt.evaluation.metrics import builtin
from engiopt.evaluation.metrics import latent

__all__ = ["builtin", "latent"]
