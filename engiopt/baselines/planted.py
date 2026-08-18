"""Planted models: built to top a column they have no business topping.

The reference instruments in `references.py` are labelled calibration standards
-- a scale bar under the board, never ranked. These are the opposite. They sit
*in* the line-up, under names that read like methods, and they are ranked
against the trained checkpoints. Each one is engineered so that some column a
paper would actually report puts it first, while a column that costs more, or
that nobody reports, puts it last.

The point is not to catch participants out. It is that **the contradiction is
guaranteed by construction rather than hoped for**: a bank assembled only from
real checkpoints might happen to have a consistent winner, in which case an
afternoon of metric argument teaches nothing. Something has to top `mmd` and
lose the physics, or there is no exercise.

Three rules keep this honest, and all three matter:

1. **Training split only.** Every construction here retrieves, blends or
   perturbs designs from `train` -- the same data every checkpoint in the bank
   was fitted on. A construction that reached into `val` or `test` would be an
   oracle wearing a model's name, and its win would be a leak rather than a
   measurement. (The reference instruments *do* use `val`; that is defensible
   for a row that is never ranked, and would not be defensible here.)
2. **Names imply, summaries never lie.** `ensemble_2d` sounds like something
   somebody trained, which is the whole cover, but every `summary` in this file
   is a literally true description of the mechanism. A participant who reads
   carefully is *supposed* to be able to work it out; that is the reward for
   reading carefully.
3. **No fabricated numbers.** These declare their real parameter count -- the
   training set they carry, exactly as `knn_retrieval` does -- and their real
   fit time. Faking a cost column to complete a disguise would corrupt the one
   board the session teaches people to trust, and the honest value is a fair
   clue for anyone who looks at the cost family first.

Every one of them is disclosed at the reveal, with its `built_to` line, when the
physics board is unsealed. Concealing the construction afterwards would sour the
lesson; disclosing it is the lesson -- these took an afternoon, they carry no
weights, and they beat trained models on the columns those models are published
with.

**Whether a given construction actually wins its target column is a
measurement, not a claim.** `wins` and `loses` are the intent; the board
decides, and a construction that tops nothing is dead weight in the line-up and
should be dropped from the config rather than argued for.
"""

from __future__ import annotations

from typing import Any, ClassVar, TYPE_CHECKING

import numpy as np

from engiopt.baselines.base import DatasetGenerator
from engiopt.baselines.base import match_volume_fraction

if TYPE_CHECKING:
    import numpy.typing as npt

    from engiopt.core import ConditionBatch


class PlantedModel(DatasetGenerator):
    """A construction that competes in the line-up rather than calibrating it.

    Class attributes:
        built_to: What this model was built to break, in one line. Printed at
            the reveal, so it has to name the column and the mechanism rather
            than gesture at a moral.
    """

    planted: ClassVar[bool] = True
    bank_eligible: ClassVar[bool] = False
    """Never admissible as a `baseline`. The bank's `planted` kind is the only
    door in, so nothing lands in a line-up by inheriting a permissive default."""

    built_to: ClassVar[str] = ""

    volume_conditions: ClassVar[tuple[str, ...]] = ("volfrac", "volume")
    """Names a volume budget goes by, in the order they are looked for.

    Spelled as candidates rather than one name because the problems disagree:
    beams2d calls it `volfrac` and heatconduction2d calls it `volume`, and a
    construction that hardcodes either one quietly stops matching the budget on
    the other -- which does not fail, it just silently drops the feasibility
    half of the trap. photonics2d has no budget at all, and there the
    constructions must run unchanged rather than raise.

    The frozen spec is the real authority (`EvalSpec.volume_condition`), but a
    generator is handed a problem rather than a spec, so this is the closest
    honest approximation available here.
    """

    def _budget_column(self) -> int | None:
        """Position of this problem's volume budget in the condition tensor, if it has one."""
        for name in self.volume_conditions:
            column = self.bank.column(name)
            if column is not None:
                return column
        return None

    def parameter_count(self) -> int:
        """The training designs this model carries, which are its parameters.

        Same accounting as `knn_retrieval`, and deliberately not disguised: a
        construction that reports a plausible-looking weight count would be
        faking a measured column, and the cost family is where an attentive
        participant is *entitled* to catch these.
        """
        return int(np.prod(self.bank.split("train").designs.shape))

    def _match_budget(self, designs: npt.NDArray[Any], requested: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Rescale to the requested volume fraction, when the problem has one.

        Applied by every construction that wants a clean feasibility column
        while its structure is nonsense -- which is the point being made:
        feasibility is a floor, and it is cheap to stand on.
        """
        column = self._budget_column()
        if column is None:
            return designs
        return match_volume_fraction(designs, requested[:, column])

    def _shift_to_budget(self, designs: npt.NDArray[Any], requested: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Hit the volume budget by subtracting a constant, not by rescaling.

        `_match_budget` finds a *gain*, which is right for a blend or a
        retrieved structure and wrong for anything whose point is a
        perturbation: multiplying by 0.8 to shed material shrinks the
        perturbation by 0.8 as well. Measured on beams2d, that took the noise
        construction's `pixel_vendi` from 12.9 to 9.8 -- below plain retrieval,
        with the entire claim rescaled away.

        An additive offset moves the density everywhere by the same amount, so
        the budget is met exactly and the perturbation keeps its amplitude.
        Bisected because clipping makes the mean a non-linear function of the
        offset.

        Args:
            designs: `(n, ...)` density fields.
            requested: `(n, n_conds)` requested conditions.

        Returns:
            The shifted designs, clipped to `[0, 1]`.
        """
        column = self._budget_column()
        if column is None:
            return np.clip(designs, 0.0, 1.0)

        out = np.empty_like(designs, dtype=np.float64)
        for i, target in enumerate(requested[:, column]):
            low, high = -1.0, 1.0
            for _ in range(40):
                offset = 0.5 * (low + high)
                if np.clip(designs[i] - offset, 0.0, 1.0).mean() > target:
                    low = offset
                else:
                    high = offset
            out[i] = np.clip(designs[i] - 0.5 * (low + high), 0.0, 1.0)
        return out


class Ensemble2D(PlantedModel):
    """A condition-weighted average of the nearest training designs.

    The trap is a theorem rather than a trick. Squared error is minimized by the
    conditional *mean*, so a model that hedges -- averaging every design that
    plausibly answers the brief -- beats every sharp model on any column that
    measures distance to the right answer, while producing a field of
    intermediate densities that is not a manufacturable structure at all. It is
    the failure mode of every pixel-loss regression in the literature, and it is
    invisible to the columns those papers report.

    Volume-matched after averaging, so it hits the budget exactly: the blur does
    not cost it a feasibility column.

    **Not in the beams2d line-up, and the reason is worth keeping.** A trained
    pixel-loss regression is a *better* conditional-mean estimator than an
    average of twelve neighbours, and `deconv_regression` -- a real checkpoint,
    already in the bank -- beat this on every column it was built to top: `mmd`
    0.0006 against 0.0114, `pixel_paired_distance` 2.18 against 6.96, and it
    tops `pca_mmd`, `lv_mmd` and both coverage columns too. So the hedging trap
    is already set on that problem by a model somebody actually trained, which
    is a stronger version of the same lesson than a planted one. Kept here
    because a problem whose bank has no regression in it still wants the trap,
    and because a construction dropped for a measured reason should record it.
    """

    description = """Takes the training designs whose briefs are closest to yours, weights
    them by how close each brief is, and averages them into one design, then
    rescales that to your volume budget.

    Averaging is a way of hedging: the result is close to every design that
    might have been right, rather than committing to one of them."""

    algo_id = "ensemble_2d"
    conditional = True
    tuning = ("neighbours",)
    summary = "Combines the closest retrieved candidates into one design, weighted by how well each matches the brief."
    built_to = (
        "Top the distance-to-the-right-answer columns (pixel_paired_distance, cond_err) and the pixel realism "
        "columns by hedging. Averaging k designs minimizes expected squared error and produces grey densities "
        "that are not a structure -- which only the latent columns and the simulator notice."
    )
    wins = ("pixel_paired_distance", "cond_err", "mmd", "viol")
    loses = ("lv_residual", "lv_vendi", "pixel_vendi", "iog", "cog", "fog")

    neighbours: ClassVar[int] = 12

    def _sample(self, conditions: ConditionBatch, n: int) -> npt.NDArray[Any]:
        """Softmax-weight the `k` nearest training designs, then match the budget."""
        requested = self.requested(conditions, n)
        pool = self.bank.split("train")
        k = min(self.neighbours, len(pool))
        indices, distances = self.bank.neighbours(requested, split="train", k=k)

        # Bandwidth per brief rather than global: the conditions are not
        # uniformly dense, and a fixed bandwidth would return one design in the
        # crowded region and a flat average of twelve in the sparse one.
        bandwidth = np.maximum(distances.mean(axis=1, keepdims=True), 1e-12)
        weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
        weights /= weights.sum(axis=1, keepdims=True)

        picked = pool.designs[indices].astype(np.float64)
        blended = np.einsum("nk,nk...->n...", weights, picked)
        return self._match_budget(blended, requested)


class Portfolio2D(PlantedModel):
    """A maximally spread set of real training designs, rescaled to each brief.

    Every design it returns is a genuine optimal structure, so it looks real by
    construction; the set is chosen by farthest-point sampling, so it is more
    spread out than anything a conditional model produces. That is exactly the
    profile the distribution and diversity families reward -- `mmd` compares
    *sets*, and a well-spread set of real designs is the best set-level answer
    there is.

    What it never does is answer the brief. The design it hands back for a given
    request is whichever portfolio member came next, rescaled to the requested
    volume so the budget column stays clean. Only the paired columns -- each
    design against the optimum for *its own* condition -- and the simulator can
    see the difference.
    """

    description = """Returns a *portfolio* rather than a prediction: a set of training designs
    chosen to be as different from one another as possible, each rescaled to
    the volume budget its brief asked for.

    The set is built by farthest-point selection -- start from the most
    unusual design, then repeatedly add whichever candidate is least like
    everything chosen so far. Every design it gives you is a real, optimal
    structure from the dataset. Which brief each one is handed to is decided
    by volume alone."""

    algo_id = "portfolio_2d"
    conditional = True
    tuning = ("candidates",)
    summary = "Returns a spread of feasible candidate structures for the brief rather than a single prediction."
    built_to = (
        "Top the set-level columns by returning real, maximally spread training designs while ignoring which "
        "brief each one answers. On beams2d it is 1st in pixel_vendi, dpp_geometric and viol, 2nd in "
        "pca_vendi and 3rd in pca_coverage -- and 9th of 11 in pixel_paired_distance, which is the only "
        "cheap column that checks each design against the brief it was given. Separates 'matches the "
        "distribution' from 'answered the question'."
    )
    wins = ("pixel_vendi", "dpp_geometric", "pca_vendi", "pca_coverage", "viol")
    loses = ("pixel_paired_distance", "lv_paired_distance", "novelty_ratio", "iog", "cog", "fog")

    candidates: ClassVar[int] = 512
    """Training designs considered before spreading. Farthest-point sampling is
    quadratic in the pool, and a Colab CPU runtime has to do this in a second."""

    def _sample(self, conditions: ConditionBatch, n: int) -> npt.NDArray[Any]:
        """Farthest-point sample the training set, then rescale each pick to its brief."""
        requested = self.requested(conditions, n)
        pool = self.bank.split("train")
        rng = self.rng()

        take = rng.choice(len(pool), size=min(self.candidates, len(pool)), replace=False)
        flat = pool.designs[take].reshape(len(take), -1).astype(np.float64)

        # Greedy farthest-point: start from the design furthest from the mean,
        # then repeatedly add whichever candidate is furthest from everything
        # chosen so far. Deterministic given the subsample, and it maximizes
        # exactly the spread the diversity columns measure.
        chosen = [int(np.linalg.norm(flat - flat.mean(axis=0), axis=1).argmax())]
        gaps = np.linalg.norm(flat - flat[chosen[0]], axis=1)
        while len(chosen) < n:
            nxt = int(gaps.argmax())
            chosen.append(nxt)
            gaps = np.minimum(gaps, np.linalg.norm(flat - flat[nxt], axis=1))

        designs = pool.designs[take[chosen[:n]]].astype(np.float64)
        return self._match_budget(designs[self._assignment(designs, requested)], requested)

    def _assignment(self, designs: npt.NDArray[Any], requested: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Which portfolio member answers which brief, decided on volume alone.

        Rescaling cannot *add* material -- a binary design multiplied by a gain
        above one only saturates -- so handing a brief a member sparser than its
        budget produces a design that silently undershoots, and the construction
        would fail the feasibility column for a boring reason instead of passing
        it for an interesting one. Sorting both sides by volume and pairing them
        in order fixes that, and costs nothing that matters: the member still
        answers nothing about the brief except how much material it wanted.
        """
        column = self._budget_column()
        if column is None:
            return np.arange(len(designs))
        order = np.argsort(designs.reshape(len(designs), -1).mean(axis=1))
        return order[np.argsort(np.argsort(requested[:, column]))]


class CoarseToFine2D(PlantedModel):
    """Solved on a coarse grid, upsampled to the full resolution.

    Multiresolution is a real topology-optimization practice and the summary is
    a true description of what happens: the retrieved design is coarsened and
    then upsampled by a strided transposed convolution. That upsampling is also
    the one every deconvolutional decoder in the literature performs, and it has
    the pathology Odena et al. named -- uneven kernel overlap leaves a period-two
    checkerboard across the whole field.

    It was built as "looks broken, and the numbers do not care", and **that is
    false, which is what makes it worth keeping.** Measured on beams2d, `mmd`
    tracks the artifact strength closely -- 0.0198 at `artifact` 0.3, 0.0396 at
    0.6, 0.0737 at 1.0, against 0.0049 for plain retrieval -- while the parity
    modulation a person sees climbs only 1.3x to 2.3x. There is no setting
    where the checkerboard is visible and pixel realism is not already
    objecting.

    What it does instead is **run the space dissociation backwards**, and it is
    the only member of the bank that does. In pixels it ranks 8th of 11 on
    `mmd`; in the PCA subspace 4th and in the learned latent 5th. A period-two
    artifact is one coherent Fourier component, and a convolutional encoder
    with pooling largely averages it away -- so the fitted instruments are the
    ones that miss a real manufacturing pathology, and raw pixels are the space
    that catches it.

    Paired with `annealed_2d`, which fails in exactly the opposite direction
    (4th in pixels, 8th in the latent, because broadband noise scatters energy
    into every direction the encoder uses), that is a stronger statement than
    either row alone: the space is a choice, neither one dominates, and a team
    that moved wholesale to latent metrics would ship this design. Its
    obedience and legality columns -- `viol` 1st, `cond_err` 2nd -- are
    perfectly happy with it either way.
    """

    description = """Retrieves the nearest training design, coarsens it by averaging blocks of
    pixels into a lower-resolution grid, then upsamples back to full
    resolution with a strided transposed convolution -- the same operation
    that sits at the end of most image decoders. Volume-matched afterwards.

    Solving coarse and refining upward is standard practice in topology
    optimization. The upsampling step is the interesting part: where the
    kernel overlaps unevenly, neighbouring pixels receive different amounts
    of signal."""

    algo_id = "coarse_to_fine_2d"
    conditional = True
    tuning = ("coarsen", "artifact")
    summary = "Solves the design on a coarse grid and upsamples it to full resolution."
    built_to = (
        "Carry a visible checkerboard artifact -- the classic transposed-convolution pathology, and a real "
        "topology-optimization failure -- and find out which columns notice. Measured: the obedience and "
        "legality columns do not (viol 1st, cond_err 2nd of 11), and the realism columns do (mmd 8th). "
        "A design can be exactly on budget, exactly on brief, and still be something no one would build."
    )
    wins = ("viol", "cond_err", "lv_vendi")
    loses = ("mmd", "pixel_vendi", "dpp_geometric")

    coarsen: ClassVar[int] = 2
    """Downsampling factor before upsampling back.

    Must divide both design axes. It is **not** quietly reduced to one that
    does: beams2d designs are 50x100, so a declared 4 silently became 2 and two
    boards apart in the tuning log turned out to be the same numbers. A knob
    that lies about its own value costs more than the tuning it was meant to
    allow.
    """

    artifact: ClassVar[float] = 0.6
    """How much of the uneven-overlap artifact to leave in, from 0 to 1.

    Normalizing a transposed convolution per-pixel removes the checkerboard
    entirely (that is Odena et al.'s fix); normalizing by the mean overlap
    leaves all of it. This interpolates, because the two ends are both useless
    here -- no artifact is nothing to see, and all of it is an artifact `mmd`
    catches on its own, which demonstrates the opposite of the intended point.
    """

    def _sample(self, conditions: ConditionBatch, n: int) -> npt.NDArray[Any]:
        """Retrieve, coarsen, upsample with an overlapping kernel, rematch the budget.

        The grid is edge-padded up to a multiple of `coarsen` and cropped back
        afterwards, because heatconduction2d is 101x101 and a construction that
        only runs on even grids is a construction that silently leaves one of
        the three problems without its checkerboard. Padding by replication is
        what a multiresolution solver does at a boundary anyway.

        Raises:
            ValueError: If `coarsen` is not a positive integer.
        """
        if self.coarsen < 1:
            raise ValueError(f"coarsen must be a positive integer, got {self.coarsen}.")

        requested = self.requested(conditions, n)
        pool = self.bank.split("train")
        designs = pool.designs[self.bank.nearest(requested, split="train")].astype(np.float64)

        height, width = designs.shape[-2:]
        factor = self.coarsen
        pad_h, pad_w = (-height) % factor, (-width) % factor
        padded = np.pad(designs, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")

        rows, columns = padded.shape[-2] // factor, padded.shape[-1] // factor
        coarse = padded.reshape(n, rows, factor, columns, factor).mean(axis=(2, 4))

        fine = coarse
        while fine.shape[-1] < width or fine.shape[-2] < height:
            fine = _transposed_conv_up(fine, artifact=self.artifact)
        fine = fine[:, :height, :width]

        return self._match_budget(fine, requested)


class Annealed2D(PlantedModel):
    """Sampled at a fixed temperature around a retrieved design.

    Additive noise on a real optimal design cannot make it a better design, and
    every pixel-space diversity column rewards it anyway -- entropy is what they
    measure and noise is free entropy. It also reads as *inventive* to a
    memorization column, since random perturbation moves a design away from
    everything in the training set.

    The rung of the distortion ladder that matters, promoted from the control
    row into the line-up so that a team ranking on diversity and novelty picks
    it.
    """

    description = """Finds the training design closest to your brief and samples around it:
    Gaussian noise at a fixed temperature, then a constant offset so the
    volume fraction still lands exactly on budget.

    Temperature controls how far each sample strays from the retrieved
    design. The offset matters -- rescaling to fix the budget would shrink
    the perturbation too, so the noise is shifted onto the budget rather
    than scaled onto it."""

    algo_id = "annealed_2d"
    conditional = True
    tuning = ("temperature",)
    summary = "Samples at a fixed temperature around the retrieved mode for the brief."
    built_to = (
        "Look diverse, obedient and legal at once by adding Gaussian noise to a real design and offsetting it "
        "back onto the volume budget. On beams2d it is 1st in cond_err and viol and 2nd in pixel_vendi, "
        "dpp_geometric and lv_vendi -- while sitting 10th of 11 in lv_paired_distance and 9th in lv_coverage. "
        "A diversity column that rises here is measuring entropy, and a novelty column that rises here "
        "(3.8x a held-out design's distance) cannot tell invention from damage."
    )
    wins = ("pixel_vendi", "dpp_geometric", "lv_vendi", "cond_err", "viol")
    loses = ("lv_paired_distance", "lv_coverage", "iog", "cog", "fog")

    temperature: ClassVar[float] = 0.15
    """Noise scale, in design units. High enough to move the diversity columns,
    low enough that the structure is still recognisable in a figure -- a
    construction the eye rejects immediately teaches nothing about the metric."""

    def _sample(self, conditions: ConditionBatch, n: int) -> npt.NDArray[Any]:
        """Retrieve the nearest training design, perturb it, and put it back on budget.

        Keeping the budget is what makes this worth ranking, and *how* it is
        kept decides whether the construction works at all. Noise alone raises
        the mean density, because clipping at zero removes its negative half, so
        an unmatched version violates the budget on nine designs in ten and any
        team that opens the legality column has it immediately. But rescaling
        afterwards to fix that shrinks the noise along with everything else --
        measured, it took `pixel_vendi` from 12.9 down to 9.8, below plain
        retrieval, which is the whole claim gone.

        So the noise is made volume-neutral *before* it is applied: it is
        centred per design, and only added where the design is free to move in
        both directions. The perturbation keeps its full amplitude and the mean
        density is untouched, which leaves the feasibility and condition
        columns clean and the diversity columns inflated -- entropy is free, and
        only a metric that knows what a design is for can tell.
        """
        requested = self.requested(conditions, n)
        pool = self.bank.split("train")
        retrieved = pool.designs[self.bank.nearest(requested, split="train")].astype(np.float64)

        noisy = retrieved + self.rng().normal(scale=self.temperature, size=retrieved.shape)
        return self._shift_to_budget(noisy, requested)


def _transposed_conv_up(field: npt.NDArray[Any], artifact: float = 1.0) -> npt.NDArray[Any]:
    """Double a field's resolution with a stride-2, 3x3 all-ones transposed convolution.

    Every output pixel at an even index receives two input taps and every odd
    one receives a single tap, so the result carries a period-two modulation
    across both axes. Dividing by the per-pixel overlap removes it exactly --
    that is Odena et al.'s recommended fix -- and dividing by the mean overlap
    leaves all of it, so `artifact` interpolates between the two divisors.

    Args:
        field: `(n, h, w)` values.
        artifact: 0 normalizes per pixel (no checkerboard), 1 by the mean
            overlap (the full pathology).

    Returns:
        `(n, 2h + 1, 2w + 1)`, to be cropped by the caller.
    """
    n, height, width = field.shape
    shape = (n, 2 * height + 1, 2 * width + 1)
    out = np.zeros(shape, dtype=np.float64)
    overlap = np.zeros(shape, dtype=np.float64)
    for row in range(3):
        for col in range(3):
            out[:, row : row + 2 * height : 2, col : col + 2 * width : 2] += field
            overlap[:, row : row + 2 * height : 2, col : col + 2 * width : 2] += 1.0

    # The geometric interpolation keeps both ends exact: at `artifact` 0 this is
    # the per-pixel divisor, at 1 the constant mean. Edges receive fewer taps
    # than the interior, so the divisor is computed rather than assumed.
    mean_overlap = max(overlap.mean(), 1e-12)
    divisor = np.where(overlap > 0, np.maximum(overlap, 1e-12), 1.0) ** (1.0 - artifact) * mean_overlap**artifact
    return out / divisor


PLANTED_MODELS: dict[str, type[PlantedModel]] = {
    cls.algo_id: cls for cls in (Ensemble2D, Portfolio2D, CoarseToFine2D, Annealed2D)
}
"""Constructions eligible to sit in a line-up, ranked, under a name that implies a method."""
