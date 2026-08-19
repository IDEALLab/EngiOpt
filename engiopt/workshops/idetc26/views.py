"""Prebuilt figures, so that looking at something costs one line and no plotting.

The session asks people to decide what is worth looking at. It does not ask them
to write matplotlib -- most of the room does not want to, and a team that spends
ten minutes on a subplot grid has spent them on the wrong thing. So every view
worth having is a method with a sensible default, and the choice a team makes is
*which* view and *for which model*, which is the choice that carries the lesson.

Two conventions hold throughout, both from the same principle -- identity is
never carried by colour alone:

- Designs are drawn on one grey ramp, dark where there is material. That is the
  topology-optimization convention, and it means a design reads the same in
  print, in greyscale, and to a colourblind reader.
- Where several models appear in one axes, every one is **directly labelled**.
  Ten models would need ten hues no palette can separate honestly, so colour is
  spent on magnitude -- how far a model moved, how good a rank is -- and never
  on which model it is.
"""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    import pandas as pd

    from engiopt.workshops.idetc26.case import Case

INK = "#0b0b0b"
"""Primary text."""

INK_SOFT = "#52514e"
"""Secondary text: annotations, axis labels, the units under a number."""

GRID = "#e2e1dc"
"""Grid and frame lines, which should recede."""

BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab"]
"""One sequential hue, light to dark, for magnitude. Never for identity."""

BLUE_LINES = BLUE[3:]
"""The same ramp with its palest steps dropped, for lines rather than fills.

A fill can recede toward the surface at the bottom of its range -- the cell is
still a shape, and its number is written on it. A 2px line cannot: the palest
steps sit near 2:1 against white, and a line the reader cannot follow is a model
they cannot check. The bottom of this ramp is the lightest step that still
carries.
"""

ACCENT = "#2a78d6"
"""The single hue used for marks when there is only one series."""

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
"""Categorical hues, in fixed order, for the few views that show several models.

Three and no more: past three, a scatter's every-pair separation stops holding
and two series become indistinguishable to a colourblind reader. A view that
would need a fourth caps instead of inventing a hue."""

MARKERS = ["o", "^", "s"]
"""Paired with SERIES, so identity survives being printed in greyscale."""

BACKDROP = "#b9b8b1"
"""The training set behind a latent map: present, faded, never the subject.

Thousands of points showing where the data lives. It is drawn at low alpha and
small size on purpose -- it is context for the comparison, and anything bolder
would compete with the two series that are actually being compared."""


class Views:
    """Every prebuilt figure. Nobody calls these directly -- `case.show()` does.

    They are kept as separate methods rather than one long function because each
    is a different picture with a different default, and `case.show` is a thin
    dispatch over them whose whole job is that a participant only has to
    remember one name.

    Attributes:
        case: The case file these views draw from.
    """

    def __init__(self, case: Case) -> None:
        self.case = case
        self._projections: dict[str, tuple[Any, str]] = {}

    # ------------------------------------------------------------------
    # Looking at designs
    # ------------------------------------------------------------------

    def designs(self, *models: str, n: int = 4, seed: int = 1, fresh: bool = False) -> Figure:
        """Designs from the suspects you name, or from every one of them.

        One suspect gets `n` designs laid out in a grid, because when you have
        asked about one model you want to see a lot of it. Several get one row
        each, because then the comparison is the point and four across is what
        fits on a slide.

        Args:
            *models: Suspects to show, by name or by any unambiguous fragment.
                Omitted means all of them.
            n: Designs per suspect.
            seed: Sampling seed.
            fresh: Resample from the checkpoint rather than using the cache.

        Returns:
            The figure, so it can be saved or resized.
        """
        import matplotlib.pyplot as plt

        labels = self._labels(models)
        if len(labels) == 1:
            drawn = self.case.designs(labels[0], seed=seed, fresh=fresh)
            count = min(n, len(drawn))
            if count < n:
                print(f"  The spec asks {len(drawn)} conditions, so {n} is more designs than exist. Showing {count}.")
            return self.grid(drawn[:count], title=f"{labels[0]}: {count} designs")

        figure, axes = plt.subplots(len(labels), n, figsize=(1.9 * n, 1.5 * len(labels)), squeeze=False)
        for row, label in enumerate(labels):
            drawn = self.case.designs(label, seed=seed, fresh=fresh)
            for column in range(n):
                self._draw_design(axes[row][column], drawn[column])
            self._row_label(axes[row][0], label)
        self._finish(figure, f"{self.case.config.display_name}: {n} designs from each suspect")
        return figure

    def grid(self, designs: Any, title: str = "", columns: int = 5) -> Figure:
        """Draw designs you already have in hand.

        What the other views build on, and what to call on any array of designs
        of your own -- including one you computed yourself from
        `case.designs(...)`.

        Args:
            designs: An array of designs.
            title: Heading for the figure.
            columns: Designs per row; rows are added as needed.

        Returns:
            The figure.
        """
        import matplotlib.pyplot as plt

        designs = np.asarray(designs)
        columns = max(1, min(columns, len(designs)))
        rows = math.ceil(len(designs) / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(1.9 * columns, 1.5 * rows), squeeze=False)
        for position, axis in enumerate(axes.flat):
            if position < len(designs):
                self._draw_design(axis, designs[position])
            else:
                axis.axis("off")
        self._finish(figure, title or f"{len(designs)} designs")
        return figure

    def compare(self, *models: str, n: int = 4, seed: int = 1) -> Figure:
        """A few models side by side, against the real optimal design for each condition.

        Every column is one condition, so a row can be read across: this is what
        each model produced when asked the same question.

        Args:
            *models: Labels to compare. Omitted means all of them.
            n: Conditions shown.
            seed: Sampling seed.

        Returns:
            The figure.
        """
        import matplotlib.pyplot as plt

        labels = self._labels(models)
        rows: list[tuple[str, Any]] = [("reference", np.asarray(self.case.evaluator.resolved.ref_designs))]
        rows += [(label, self.case.designs(label, seed=seed)) for label in labels]

        figure, axes = plt.subplots(len(rows), n, figsize=(1.9 * n, 1.5 * len(rows)), squeeze=False)
        for row, (name, designs) in enumerate(rows):
            for column in range(n):
                self._draw_design(axes[row][column], designs[column])
            self._row_label(axes[row][0], name)
        for column in range(n):
            axes[0][column].set_title(f"condition {column + 1}", fontsize=8, color=INK_SOFT)
        self._finish(figure, "Same condition down each column")
        return figure

    def nearest_training(self, model: str, n: int = 4, seed: int = 1) -> Figure:
        """Each design beside the closest design in the training set.

        The memorization check, done by eye. A model that has learned the
        problem produces something that is *like* the training data; a model
        that has memorized it produces something that is the training data, and
        the two are indistinguishable in any distribution metric.

        Args:
            model: Which suspect to check.
            n: Designs to check.
            seed: Sampling seed.

        Returns:
            The figure.

        Raises:
            ValueError: If the problem has no training split to compare against.
        """
        import matplotlib.pyplot as plt

        train = self.case.evaluator.train_designs
        if train is None:
            raise ValueError(f"{self.case.config.problem_id!r} has no training split, so there is nothing to be near.")

        model = self.case.resolve(model).label
        designs = self.case.designs(model, seed=seed)[:n]
        anchors = np.asarray(train).reshape(len(train), -1)
        flat = np.asarray(designs).reshape(len(designs), -1)
        distances = np.linalg.norm(flat[:, None, :] - anchors[None, :, :], axis=2)
        nearest = distances.argmin(axis=1)

        figure, axes = plt.subplots(2, n, figsize=(1.9 * n, 3.4), squeeze=False)
        for column in range(len(designs)):
            self._draw_design(axes[0][column], designs[column])
            self._draw_design(axes[1][column], np.asarray(train)[nearest[column]])
            axes[1][column].set_title(f"distance {distances[column, nearest[column]]:.3f}", fontsize=8, color=INK_SOFT)
        self._row_label(axes[0][0], model)
        self._row_label(axes[1][0], "closest\ntraining design")
        self._finish(figure, f"{model}: is it generating, or remembering?")
        return figure

    def conditions(self, model: str, n: int = 6, seed: int = 1) -> Figure:
        """Designs labelled with what was asked for and what came back.

        A generative model is supposed to answer a question, not just produce
        something plausible. This is the view where a model that ignores its
        conditions gives itself away, and it needs no metric.

        Args:
            model: Which model to inspect.
            n: Conditions shown, drawn evenly across the spec's range.
            seed: Sampling seed.

        Returns:
            The figure.
        """
        import matplotlib.pyplot as plt

        model = self.case.resolve(model).label
        designs = self.case.designs(model, seed=seed)
        requested = self._requested_volume()
        order = np.argsort(requested) if requested is not None else np.arange(len(designs))
        picks = order[np.linspace(0, len(order) - 1, min(n, len(order))).astype(int)]

        figure, axes = plt.subplots(1, len(picks), figsize=(1.9 * len(picks), 2.3), squeeze=False)
        for column, index in enumerate(picks):
            axis = axes[0][column]
            self._draw_design(axis, designs[index])
            realized = float(np.mean(np.asarray(designs[index])))
            if requested is None:
                axis.set_title(f"got {realized:.2f}", fontsize=8, color=INK_SOFT)
                continue
            asked = float(requested[index])
            axis.set_title(f"asked {asked:.2f}\ngot {realized:.2f}", fontsize=8, color=INK_SOFT)
        name = self.case.evaluator.spec.volume_condition or "the condition"
        self._finish(figure, f"{model}: did it answer the question? ({name}, sorted)")
        return figure

    # ------------------------------------------------------------------
    # Looking at boards
    # ------------------------------------------------------------------

    def board(self, frame: pd.DataFrame) -> Figure:
        """A board as ranks: suspects down the side, columns across, 1 is best.

        Raw metric values span twenty orders of magnitude across columns, so a
        board of values cannot be read as a picture. Ranks can, and reading one
        makes the disagreement between columns visible in a way the numbers do
        not.

        Args:
            frame: Any board, e.g. what `case.evaluate()` returned.

        Returns:
            The figure.

        Raises:
            ValueError: If nothing in the board can be ranked.
        """
        import matplotlib.pyplot as plt

        ranked = self.case.rank(frame)
        if ranked.empty:
            raise ValueError("Nothing in that board can be ranked: no column has a direction, or all are constant.")

        values = ranked.to_numpy(dtype=float)
        best, worst = 1.0, float(np.nanmax(values))
        figure, axis = plt.subplots(figsize=(0.95 * len(ranked.columns) + 2.6, 0.42 * len(ranked) + 1.8))
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                rank = values[row, column]
                if np.isnan(rank):
                    continue
                # Darkest = best, so the eye lands on rank 1 without a key.
                shade = BLUE[round((1 - (rank - best) / max(worst - best, 1)) * (len(BLUE) - 1))]
                axis.add_patch(plt.Rectangle((column + 0.03, row + 0.03), 0.94, 0.94, facecolor=shade, edgecolor="none"))
                axis.text(
                    column + 0.5,
                    row + 0.5,
                    f"{int(rank)}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#ffffff" if rank <= (best + worst) / 2 else INK,
                )
        axis.set_xlim(0, values.shape[1])
        axis.set_ylim(values.shape[0], 0)
        axis.set_xticks(np.arange(values.shape[1]) + 0.5)
        axis.set_xticklabels(ranked.columns, rotation=45, ha="left", fontsize=9, color=INK_SOFT)
        axis.xaxis.set_ticks_position("top")
        axis.set_yticks(np.arange(values.shape[0]) + 0.5)
        axis.set_yticklabels(ranked.index, fontsize=9, color=INK)
        for side in axis.spines.values():
            side.set_visible(False)
        axis.tick_params(length=0)
        self._finish(figure, "Rank in each column, 1 = best", pad=1.4)
        return figure

    def scatter(self, frame: pd.DataFrame, x: str, y: str) -> Figure:
        """Two columns of a board against each other, one labelled dot per model.

        The fastest way to see that two metrics are measuring the same thing --
        or that they are not.

        Args:
            frame: A board.
            x: Column on the horizontal axis.
            y: Column on the vertical axis.

        Returns:
            The figure.

        Raises:
            KeyError: If either column is not in the board.
        """
        import matplotlib.pyplot as plt

        for column in (x, y):
            if column not in frame.columns:
                raise KeyError(f"{column!r} is not in this board. It has: {list(frame.columns)}")

        figure, axis = plt.subplots(figsize=(6.0, 4.6))
        axis.scatter(frame[x], frame[y], s=64, color=ACCENT, edgecolor="#ffffff", linewidth=1.5, zorder=3)
        for label, row in frame.iterrows():
            axis.annotate(
                str(label),
                (row[x], row[y]),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=9,
                color=INK,
            )
        axis.set_xlabel(self._axis_label(x), fontsize=10, color=INK_SOFT)
        axis.set_ylabel(self._axis_label(y), fontsize=10, color=INK_SOFT)
        axis.grid(visible=True, color=GRID, linewidth=1, zorder=0)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
        axis.tick_params(colors=INK_SOFT, length=0)
        self._finish(figure, f"{y} against {x}")
        return figure

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _labels(self, models: tuple[str, ...]) -> list[str]:
        """Resolve the models a view was asked for, defaulting to the whole bank.

        Fragments are accepted, so `compare("knn", "diffusion")` works and the
        figure is still labelled with the models' full names.
        """
        if not models:
            return list(self.case.bank.labels)
        return [self.case.resolve(model).label for model in models]

    def _requested_volume(self) -> np.ndarray | None:
        """The volume fraction each condition asked for, if this problem has one."""
        name = self.case.evaluator.spec.volume_condition
        conditions = self.case.evaluator.resolved.conditions
        if name is None or conditions is None or name not in conditions.column_names:
            return None
        return np.asarray(conditions[name], dtype=float)

    @staticmethod
    def _axis_label(column: str) -> str:
        """A column name with the direction that makes it good."""
        from engiopt.workshops.idetc26.case import column_direction

        direction = column_direction(column)
        if direction is None:
            return column
        return f"{column}  ({'higher' if direction else 'lower'} is better)"

    @staticmethod
    def _draw_design(axis: Any, design: Any) -> None:
        """Draw one design, dark where there is material."""
        axis.imshow(np.asarray(design).squeeze(), cmap="gray_r", vmin=0, vmax=1, aspect="equal")
        axis.set_xticks([])
        axis.set_yticks([])
        for side in axis.spines.values():
            side.set_color(GRID)

    @staticmethod
    def _row_label(axis: Any, text: str) -> None:
        """Name a row of a design grid, on the left where a reader starts."""
        axis.set_ylabel(text, rotation=0, ha="right", va="center", fontsize=10, color=INK, labelpad=8)

    @staticmethod
    def _finish(figure: Figure, title: str, *, pad: float = 1.1) -> None:
        """Title a figure and tighten it, the same way every time."""
        figure.suptitle(title, fontsize=12, color=INK)
        figure.tight_layout(pad=pad)

    # ------------------------------------------------------------------
    # Interactive viewers
    # ------------------------------------------------------------------

    def dataset(self, split: str = "train", *, n_nearest: int = 1) -> Any:
        """Browse one dataset split by condition, before any suspect is involved.

        The first thing worth knowing about a design problem is what its
        solutions look like and how they change with the brief. Sliders over the
        conditions and the nearest real design to them answers that in a way no
        column can, and it gives every later comparison something to be read
        against.

        Args:
            split: Dataset split to browse.
            n_nearest: How many nearest designs to show for the slider position.

        Returns:
            The widget, or a static figure when ipywidgets is unavailable.
        """
        designs, conditions, keys = self._split_data(split)

        def render(**values: float) -> None:
            order = self._nearest(conditions, values, keys)[:n_nearest]
            figure = self.grid(designs[order], title=f"{split} split: nearest to the requested conditions")
            for position, index in enumerate(order):
                actual = ", ".join(f"{key}={conditions[index, j]:.3g}" for j, key in enumerate(keys))
                figure.axes[position].set_title(actual, fontsize=7, color=INK_SOFT)
            self._show(figure)

        return self._with_sliders(render, conditions, keys)

    def side_by_side(self, left: str, right: str, seed: int = 1) -> Any:
        """Two sources answering the same condition, with the condition on sliders.

        Either side may be a model in the bank or the data itself -- `"test"`
        for the reference optimum the spec scores against, `"train"` for the
        nearest training design. Putting a model beside the data at the *same*
        brief is the comparison the gallery cannot make: a gallery shows four
        designs chosen by nobody, and this shows the one you asked for.

        Args:
            left: Model name, or `"test"` / `"train"`.
            right: Likewise.
            seed: Sampling seed for whichever sides are models.

        Returns:
            The widget, or a static figure when ipywidgets is unavailable.
        """
        keys = list(self.case.evaluator.resolved.condition_keys)
        spec_conditions = np.asarray(self.case.evaluator.resolved.conditions_tensor.detach().cpu().numpy(), dtype=float)
        # Resolved once, outside the redraw: a bad name should fail immediately
        # rather than on the first slider move, and the panels should be titled
        # with the model's real name rather than the fragment that was typed.
        sources = [self._resolve_source(name) for name in (left, right)]

        def render(**values: float) -> None:
            index = int(self._nearest(spec_conditions, values, keys)[0])
            asked = ", ".join(f"{key}={spec_conditions[index, j]:.3g}" for j, key in enumerate(keys))
            panels = [
                (label, self._design_from(kind, label, index, spec_conditions[index], seed)) for kind, label in sources
            ]

            import matplotlib.pyplot as plt

            figure, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 2.6), squeeze=False)
            for column, (name, design) in enumerate(panels):
                self._draw_design(axes[0][column], design)
                realized = float(np.mean(np.asarray(design)))
                axes[0][column].set_title(f"{name}\nrealized mean {realized:.3f}", fontsize=9, color=INK)
            self._finish(figure, f"condition {index}: {asked}")
            self._show(figure)

        return self._with_sliders(render, spec_conditions, keys)

    # ------------------------------------------------------------------
    # Viewer internals
    # ------------------------------------------------------------------

    def _resolve_source(self, name: str) -> tuple[str, str]:
        """Turn a side's name into `(kind, label)`.

        The data can stand in for a model on either side, so `"test"` and
        `"train"` are accepted names -- and the label that comes back says which
        one it is, since "train" on a panel is ambiguous about whether you are
        looking at a design or a model named after one.
        """
        if name.lower() in {"test", "reference", "ref"}:
            return "reference", "test set"
        if name.lower() == "train":
            return "train", "train set"
        return "model", self.case.resolve(name).label

    def _design_from(self, kind: str, label: str, index: int, condition: Any, seed: int) -> Any:
        """One design for one condition, from a model or from the data itself."""
        if kind == "reference":
            return np.asarray(self.case.evaluator.resolved.ref_designs)[index]
        if kind == "train":
            designs, conditions, keys = self._split_data("train")
            nearest = self._nearest(conditions, dict(zip(keys, np.asarray(condition, dtype=float))), keys)[0]
            return designs[nearest]
        return self.case.designs(label, seed=seed)[index]

    def _split_data(self, split: str) -> tuple[Any, Any, list[str]]:
        """Designs and scalar conditions for one dataset split, loaded once."""
        cached = getattr(self, "_split_cache", None)
        if cached is None:
            cached = {}
            self._split_cache = cached
        if split not in cached:
            from engiopt.transforms import condition_keys

            problem = self.case.evaluator.problem
            keys = list(condition_keys(problem))
            data = problem.dataset[split]
            designs = np.asarray(data["optimal_design"], dtype=float)
            conditions = np.stack([np.asarray(data[key], dtype=float) for key in keys], axis=1)
            cached[split] = (designs, conditions, keys)
        return cached[split]

    @staticmethod
    def _nearest(conditions: Any, values: dict[str, float], keys: list[str]) -> Any:
        """Indices of the rows closest to `values`, each condition weighted equally.

        Normalized by each condition's own range, so a volume fraction on [0.15,
        0.45] is not drowned out by a filter radius on [1, 3].
        """
        distances = np.zeros(len(conditions))
        for j, key in enumerate(keys):
            column = conditions[:, j]
            spread = float(column.max() - column.min()) or 1.0
            distances += ((column - float(values[key])) / spread) ** 2
        return np.argsort(distances)

    def _with_sliders(self, render: Any, conditions: Any, keys: list[str]) -> Any:
        """Wire one slider per condition, or fall back to a single static view.

        The fallback matters: `ipywidgets` is present in Colab and in most local
        installs but not all of them, and a viewer that raises on import would
        take the notebook down at the exact moment somebody is trying to look at
        a design.
        """
        try:
            from ipywidgets import FloatSlider
            from ipywidgets import interact
        except ImportError:
            print("  [note] ipywidgets is not installed, so this is a single static view.")
            render(**{key: float(np.median(conditions[:, j])) for j, key in enumerate(keys)})
            return None

        sliders = {}
        for j, key in enumerate(keys):
            low, high = float(conditions[:, j].min()), float(conditions[:, j].max())
            sliders[key] = FloatSlider(
                value=float(np.median(conditions[:, j])),
                min=low,
                max=high,
                step=(high - low) / 100 if high > low else 0.01,
                description=f"{key}:",
                continuous_update=False,
                readout_format=".3g",
            )
        return interact(render, **sliders)

    @staticmethod
    def _show(figure: Figure) -> None:
        """Draw a freshly rendered figure and drop the previous one.

        Closed after display because a slider redraws on every move: without
        this, figures accumulate for as long as somebody is dragging and
        matplotlib starts warning about it after twenty.
        """
        from IPython.display import clear_output
        from IPython.display import display
        import matplotlib.pyplot as plt

        clear_output(wait=True)
        display(figure)
        plt.close(figure)

    def space_map(self, *models: str, space: str = "lv", seed: int = 1) -> Figure:
        """Generated and real designs plotted in the top two dimensions of a space.

        The distribution columns reduce a whole comparison to one number. This
        is the same comparison with its shape left in: whether a model covers
        the real designs, sits beside them, or has collapsed into a corner is
        visible here and is not recoverable from an MMD.

        "Top two" means the two dimensions with the most variance *in the
        reference designs*, so the view shows the axes along which real designs
        actually differ. For PCA that is the first two components by
        construction; for a pruned latent space it has to be measured, because
        active dimensions are not ordered.

        The **training set is always the faded backdrop** -- thousands of points
        whose job is to show where the data lives, not to be read individually.
        On top of it go the two sources you name, each with its own hue *and*
        its own marker, so identity survives a greyscale print and a colourblind
        reader. Either of them may be `"test"`, which is how you ask whether a
        model lands where the real held-out designs land.

        Args:
            *models: Up to two sources: a model name, or `"test"` / `"train"`.
                Defaults to the first two models in the bank. Two and no more --
                past three series on one scatter the every-pair separation stops
                holding.
            space: `"lv"` for the pinned autoencoder's latent space, `"pca"` for
                the matched PCA subspace.
            seed: Sampling seed.

        Returns:
            The figure.

        Raises:
            ValueError: If `space` is not one of the two, or the latent space
                was asked for on a problem whose spec pins no autoencoder.
        """
        import matplotlib.pyplot as plt

        if space not in {"lv", "pca"}:
            raise ValueError(f"space must be 'lv' or 'pca', not {space!r}.")
        chosen = list(models)[:2] or list(self.case.bank.labels)[:2]

        reference, axis_names = self._codes(self.case.evaluator.resolved.ref_designs, space)
        # Ordered by spread in the real designs: the axes worth plotting are the
        # ones the data actually varies along, whichever space it is measured in.
        top = np.argsort(reference.var(axis=0))[::-1][:2]

        figure, axis = plt.subplots(figsize=(6.4, 5.2))

        train = self.case.evaluator.train_designs
        performance, objective, direction = self._train_performance()
        graded = False
        if train is not None:
            train_codes, _ = self._codes(train, space)
            # Only grade when the values line up one for one with the designs.
            # A backdrop coloured by a mismatched array is not a weaker figure,
            # it is a wrong one.
            graded = performance is not None and len(performance) == len(train_codes)
            if graded:
                low, high = np.nanpercentile(performance, [2, 98])
                drawn = axis.scatter(
                    train_codes[:, top[0]],
                    train_codes[:, top[1]],
                    s=14,
                    c=performance,
                    cmap=_ramp(),
                    vmin=low,
                    vmax=high,
                    alpha=0.65,
                    linewidth=0,
                    label=f"training set ({len(train_codes)})",
                    zorder=1,
                )
                bar = figure.colorbar(drawn, ax=axis, pad=0.02)
                bar.set_label(f"{objective} of the training designs ({direction}, 2-98%)", fontsize=9, color=INK_SOFT)
                bar.ax.tick_params(colors=INK_SOFT, length=0)
                bar.outline.set_visible(False)
            else:
                axis.scatter(
                    train_codes[:, top[0]],
                    train_codes[:, top[1]],
                    s=14,
                    color=BACKDROP,
                    alpha=0.45,
                    linewidth=0,
                    label=f"training set ({len(train_codes)})",
                    zorder=1,
                )

        # Blue carries magnitude the moment the backdrop is graded, so identity
        # moves off it: the sources take the two hues the ramp cannot be
        # confused with rather than sitting in the middle of its range.
        hues = SERIES[1:] if graded else SERIES
        marks = MARKERS[1:] if graded else MARKERS
        for position, name in enumerate(chosen):
            codes, label = self._source_codes(name, space, seed)
            axis.scatter(
                codes[:, top[0]],
                codes[:, top[1]],
                s=68,
                color=hues[position % len(hues)],
                edgecolor="#ffffff",
                linewidth=1.2,
                label=label,
                marker=marks[position % len(marks)],
                zorder=2 + position,
            )

        axis.set_xlabel(f"{axis_names} {top[0]}", fontsize=10, color=INK_SOFT)
        axis.set_ylabel(f"{axis_names} {top[1]}", fontsize=10, color=INK_SOFT)
        axis.grid(visible=True, color=GRID, linewidth=1, zorder=0)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
        axis.tick_params(colors=INK_SOFT, length=0)
        axis.legend(frameon=False, fontsize=9, labelcolor=INK)
        self._finish(figure, f"Where the designs sit, in the two widest {axis_names.lower()} directions")
        return figure

    def _train_performance(self) -> tuple[Any, str, str]:
        """The objective value of each training design, and how to read it.

        The backdrop is every design the models were fitted on, and until now it
        said only *where* the data lives. These are optimal designs, so each one
        carries the objective the optimizer reached for its conditions -- which
        turns the same cloud into a statement about where in the space the good
        designs are. A model can then be read against the part of the manifold
        that matters rather than against its outline.

        Free: the value is a column of the dataset already in memory, no
        simulator involved.

        Returns:
            `(values, name, direction)`, or `(None, "", "")` when the problem
            declares no objective or the split does not carry it.
        """
        problem = self.case.evaluator.problem
        objectives = tuple(getattr(problem, "objectives", ()) or ())
        dataset = getattr(problem, "dataset", None)
        if not objectives or dataset is None or "train" not in dataset:
            return None, "", ""

        key = str(objectives[0][0])
        split = dataset["train"]
        if key not in getattr(split, "column_names", []):
            return None, "", ""

        values = np.asarray(split[key], dtype=float)
        if not np.isfinite(values).any():
            return None, "", ""
        minimized = "MIN" in str(getattr(objectives[0][1], "name", objectives[0][1])).upper()
        return values, key, "lower is better" if minimized else "higher is better"

    def _source_codes(self, name: str, space: str, seed: int) -> tuple[Any, str]:
        """Encode one named source -- a model, or the data itself -- into a space.

        The same vocabulary `side_by_side` uses, so `"test"` means the same
        thing in both views.
        """
        kind, label = self._resolve_source(name)
        if kind == "reference":
            designs = np.asarray(self.case.evaluator.resolved.ref_designs)
        elif kind == "train":
            designs = np.asarray(self._split_data("train")[0])
        else:
            designs = np.asarray(self.case.designs(label, seed=seed))
        return self._codes(designs, space)[0], label

    def _codes(self, designs: Any, space: str) -> tuple[Any, str]:
        """Encode designs into the requested space, with a name for its axes."""
        project, axis_name = self._projection(space)
        return project(np.asarray(designs)), axis_name

    def _projection(self, space: str) -> tuple[Any, str]:
        """A projection into `space`, fitted once and valid for any number of designs.

        Read off a context built from the *reference* designs rather than from
        whatever is being drawn, because `context_from_designs` pairs the
        designs it is handed with the spec's conditions one for one and so
        cannot hold a training split of thousands. Neither fitted object depends
        on that pairing: the PCA comes from the validation split and the encoder
        is the instrument the spec pins, so the backdrop lands in exactly the
        space the metrics measure in.
        """
        if space in self._projections:
            return self._projections[space]

        context = self.case.evaluator.context_from_designs(np.asarray(self.case.evaluator.resolved.ref_designs))
        if space == "pca":
            pca = context.pca_model

            def project_pca(designs: Any) -> Any:
                flat = np.asarray(designs)
                return pca.transform(flat.reshape(len(flat), -1))

            self._projections[space] = (project_pca, "PCA component")
        else:
            from engiopt.lvae.encode import encode_active

            encoder = context.require_latent_lvae().encoder
            device = next(encoder.parameters()).device

            def project_latent(designs: Any) -> Any:
                return encode_active(encoder, np.asarray(designs), device)

            self._projections[space] = (project_latent, "latent dimension")
        return self._projections[space]


def _ramp() -> Any:
    """`BLUE` as a continuous colormap, for the one view that grades a cloud."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("engiopt_blue", BLUE)


def _panel_title(kind: str, label: str) -> str:
    """What a `side_by_side` panel calls its source.

    Qualified here rather than in the shared label, because these only make
    sense when one condition is on screen: in a legend over a whole set they
    would be wrong.
    """
    if kind == "reference":
        return f"{label} (optimum for this condition)"
    if kind == "train":
        return f"{label} (nearest design)"
    return label
