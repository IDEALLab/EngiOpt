"""Workshop harnesses, one subpackage per event.

This file is not decorative. `[tool.setuptools.packages.find]` in
`pyproject.toml` uses the regular (non-namespace) finder, which walks a
directory only if it carries an `__init__.py`. Without this file setuptools
stopped here, so `engiopt.workshops.idetc26` was never declared as a package --
and neither the module nor its package data (problem configs, the sealed board,
the pre-sampled design cache) shipped in a wheel.

That is invisible from a source checkout, where `engiopt.workshops` resolves as
an implicit namespace package and everything imports. It is fatal in Colab,
which runs `pip install git+...` and gets only what setuptools declared.
"""
