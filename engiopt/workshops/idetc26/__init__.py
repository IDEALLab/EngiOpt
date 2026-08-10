"""The IDETC'26 "find the best model" challenge.

Teams receive a bank of anonymized generators, compute the metrics they can
actually afford, commit publicly to a winner, and then watch the ranking come
apart -- first under a change of seed, then under the columns they were not
given, then under the disclosure that several bank members are constructed
frauds.

The module is deliberately thin. Everything real -- loading, sampling, scoring
-- is the ordinary `engiopt.evaluation` path, because a challenge run on a
special code path would prove nothing about the benchmark.

    from engiopt.workshops.idetc26 import Challenge

    ch = Challenge.open("beams2d", team="orange")
    ch.gallery()                       # look before you measure
    board = ch.board()                 # the cheap metrics, anonymized
    ch.submit(winner="Model C", why="best MMD by a wide margin")
    ch.reveal()                        # seeds, withheld columns, identities
"""

from engiopt.workshops.idetc26.bank import BankMember
from engiopt.workshops.idetc26.bank import ModelBank
from engiopt.workshops.idetc26.challenge import Challenge
from engiopt.workshops.idetc26.challenge import Verdict
from engiopt.workshops.idetc26.config import WorkshopConfig

__all__ = ["BankMember", "Challenge", "ModelBank", "Verdict", "WorkshopConfig"]
