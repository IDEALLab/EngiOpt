"""The IDETC'26 "which of these is the best model?" challenge.

A line-up of generative models for one engineering design problem, and a set of
questions you are allowed to put to them. One of them is the best. Working out
which -- and, harder, working out what you had to measure before you were
entitled to say so -- is the session.

The answer is meant to be hard: the columns disagree, they disagree differently
depending on the space they measure in, and the ranking moves when only the
random seed changes.

The module is deliberately thin. Everything real -- loading, sampling, scoring
-- is the ordinary `engiopt.evaluation` path, because a challenge run on a
special code path would prove nothing about the benchmark.

There are four commands:

    from engiopt.workshops.idetc26 import Case

    case = Case.open("beams2d")
    case.models()                   who is in the line-up
    case.metrics()                  what you may ask, by line of questioning
    case.evaluate("diversity")      put a question to them
    case.show("diffusion")          look at what one of them produced
"""

from engiopt.workshops.idetc26.bank import BankMember
from engiopt.workshops.idetc26.bank import ModelBank
from engiopt.workshops.idetc26.case import Case
from engiopt.workshops.idetc26.config import WorkshopConfig
from engiopt.workshops.idetc26.designs import DesignStore
from engiopt.workshops.idetc26.families import FAMILIES
from engiopt.workshops.idetc26.families import Family
from engiopt.workshops.idetc26.families import family_of

__all__ = [
    "FAMILIES",
    "BankMember",
    "Case",
    "DesignStore",
    "Family",
    "ModelBank",
    "WorkshopConfig",
    "family_of",
]
