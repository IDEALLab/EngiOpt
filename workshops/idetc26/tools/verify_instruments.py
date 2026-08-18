r"""Preflight: do the pinned autoencoders and the bank's own PLVAE actually load?

Three things go wrong silently and only this catches them.

**A deleted checkpoint.** Bank members marked `optional` skip when their package
is missing, so a deleted fingerprint shrinks the bank instead of failing. Every
`constrained_plvae_2d` member in all three problem configs was pointing at a
package deleted with the pre-rebuild pool, and the boards showed nine models
where the config listed ten.

**A width that is not the declared one.** `expected_n_active` is a spec field
nothing enforces, and the published `n_active_dims` metadata disagrees with it
on most packages. The number that matters is the one on the encoder's pruning
mask, because that is the subspace the latent columns are computed in.

**An instrument that measures itself.** If the bank's autoencoder is also the
pinned instrument, its latent columns are scored in its own latent space. This
asserts they are different packages.

Cheap -- it loads encoders and samples a handful of designs, no simulator. Run
it on the cluster rather than a laptop.

Example:
    python workshops/idetc26/tools/verify_instruments.py --problem-id beams2d
"""

from __future__ import annotations

import argparse
import traceback

from engiopt.evaluation import Evaluator
from engiopt.workshops.idetc26.config import WorkshopConfig

PROBLEMS = ("beams2d", "heatconduction2d", "photonics2d")


def check(problem_id: str, *, n_samples: int) -> bool:
    """Report on one problem's instrument pair and PLVAE bank member.

    Returns:
        True if nothing failed.
    """
    print(f"\n===== {problem_id} =====")
    ok = True
    config = WorkshopConfig.load(problem_id)
    evaluator = Evaluator.for_problem(problem_id, spec=config.spec)
    pinned = evaluator.spec.latent_instrument
    if pinned is None:
        print(f"  spec {config.spec} pins no latent instrument -- every lv_ column will be unavailable")
        return True

    from engiopt.lvae.encode import get_active_mask

    for role, fingerprint, lvae in (
        ("instrument", pinned.config_fingerprint, evaluator.latent_lvae),
        ("companion", pinned.recon_only_config_fingerprint, evaluator.latent_recon_lvae),
    ):
        if lvae is None:
            print(f"  {role:11s} {fingerprint}: NOT PINNED")
            ok = ok and role == "companion"
            continue
        width = int(get_active_mask(lvae.encoder).sum())
        declared = pinned.expected_n_active if role == "instrument" else None
        mismatch = "" if declared in (None, width) else f"  <-- spec declares {declared}"
        print(f"  {role:11s} {fingerprint}: {width} active dims, decoder={'yes' if lvae.has_decoder else 'NO'}{mismatch}")
        if not lvae.has_decoder:
            print("               lv_residual and lv_dual_gap will be unavailable without a decoder")
            ok = False

    for member in config.bank:
        if member.get("algo") != "constrained_plvae_2d":
            continue
        fingerprint = member.get("config_fingerprint")
        seed = member.get("seed", 1)
        if fingerprint in (pinned.config_fingerprint, pinned.recon_only_config_fingerprint):
            print(f"  bank member {fingerprint} IS the pinned instrument -- it would score itself")
            ok = False
        try:
            from engiopt.utils.all_generators import BUILTIN_GENERATORS

            generator = BUILTIN_GENERATORS["constrained_plvae_2d"].from_pretrained(
                evaluator.problem,
                problem_id=problem_id,
                seed=seed,
                model_source="hf",
                config_fingerprint=fingerprint,
            )
            ctx = evaluator.context_for(generator, n_samples=n_samples)
            print(f"  bank member {fingerprint}/s{seed}: loaded, sampled {len(ctx.gen_designs)} designs")
        except Exception as exc:  # noqa: BLE001 - reporting every failure beats stopping at the first
            print(f"  bank member {fingerprint}/s{seed}: FAILED {type(exc).__name__}: {str(exc)[:160]}")
            traceback.print_exc()
            ok = False
    return ok


def main() -> None:
    """Check each requested problem and exit non-zero if any failed."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem-id", nargs="*", default=list(PROBLEMS))
    ap.add_argument("--n-samples", type=int, default=4)
    args = ap.parse_args()

    results = {p: check(p, n_samples=args.n_samples) for p in args.problem_id}
    print("\n=== summary ===")
    for problem_id, ok in results.items():
        print(f"  {problem_id:20s} {'ok' if ok else 'FAILED'}")
    raise SystemExit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
