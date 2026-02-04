# Description

Please include a summary of the change and which issue is fixed. Please also include relevant motivation and context. List any dependencies that are required for this change.

Fixes # (issue)

## Type of change

Please delete options that are not relevant.

- [ ] Documentation only change (no code changed)
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New algorithm (non-breaking change which adds a new model/algorithm)
- [ ] Improvement to existing algorithm (non-breaking change which improves functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] This change requires a documentation update

### Screenshots

Please attach before and after screenshots of the change if applicable (e.g., generated designs, training curves, WandB dashboards).

<!--
Example:

| Before | After |
| ------ | ----- |
| _gif/png before_ | _gif/png after_ |


To upload images to a PR -- simply drag and drop an image while in edit mode and it should upload the image directly. You can then paste that source into the above before/after sections.
-->

# Checklist:

### Code Quality
- [ ] I have run the [`pre-commit` checks](https://pre-commit.com/) with `pre-commit run --all-files`
- [ ] I have run `ruff check .` and `ruff format`
- [ ] I have run `mypy .`
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] My changes generate no new warnings

### CleanRL Philosophy (for new/modified algorithms)
- [ ] The implementation follows the [CleanRL](https://github.com/vwxyzjn/cleanrl) single-file philosophy: all training logic is contained in one file
- [ ] The code is reproducible: random seeds are set, PyTorch determinism is enabled
- [ ] Hyperparameters are configurable via command-line arguments using `tyro`
- [ ] WandB logging is integrated with `--track` flag support
- [ ] The model can be saved and restored via WandB artifacts (`--save-model` flag)

### Algorithm Completeness (for new algorithms)
- [ ] Both training script (`algorithm.py`) and evaluation script (`evaluate_algorithm.py`) are provided
- [ ] The algorithm works with EngiBench's `Problem` interface
- [ ] The algorithm is added to the README table with correct metadata

### Documentation
- [ ] I have made corresponding changes to the documentation
- [ ] New algorithms include docstrings explaining the approach and any paper references

<!--
As you go through the checklist above, you can mark something as done by putting an x character in it

For example,
- [x] I have done this task
- [ ] I have not done this task
-->


# Reviewer Checklist:

- [ ] The content of this PR brings value to the community. It is not overly specific to a particular use case.
- [ ] The tests and checks pass (linting, formatting, type checking).
- [ ] The code follows the CleanRL single-file philosophy where appropriate.
- [ ] The code is understandable and commented. No large code blocks are left unexplained. Can I read and understand the code easily?
- [ ] There is no merge conflict.
- [ ] For new algorithms: both training and evaluation scripts are provided.
- [ ] For new algorithms: WandB integration is complete (logging, artifacts, reproducibility).
- [ ] For bugfixes: it is a robust fix and not a hacky workaround.
- [ ] The changes do not break existing trained models or evaluation workflows without good reason.
