"""Generative models shipped with EngiOpt.

Each subpackage holds one model family: a self-contained training script (kept
single-file, in the CleanRL style) plus an `adapter.py` implementing the
`Generator` contract from `engiopt.core`.

Use `engiopt.utils.all_generators.BUILTIN_GENERATORS` to look models up by name.
"""
