"""Execute the challenge notebook end to end and fail on the first error.

The fill-in cells ship with placeholders (`passphrase = "..."`) that a
participant replaces. This substitutes plausible answers and runs every cell, so
a broken API or a stale method name surfaces here rather than in a room of
ninety people.

    python workshops/idetc26/tools/verify_notebook.py --passphrase "..."

This is the only check that catches a notebook whose *prose* is right and whose
*calls* are stale, so it has to be run after every change to `build_notebook.py`
or to the `Case` API. It samples every suspect and runs the simulator on a
couple of designs, so it belongs on a workstation rather than in CI.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

from nbclient import NotebookClient
import nbformat

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "01_find_the_best_model.ipynb"

SUBSTITUTIONS: dict[str, str] = {}
"""Fill-in placeholders replaced before execution, as `{placeholder: answer}`.

Empty since the physics board moved to the Hub: there is no passphrase to
substitute, and every cell now runs as a participant would find it.
"""


def prepare(_passphrase: str = "") -> nbformat.NotebookNode:
    """Load the notebook and fill in the cells a participant would."""
    notebook = nbformat.read(NOTEBOOK, as_version=4)

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for placeholder, answer in SUBSTITUTIONS.items():
            cell.source = cell.source.replace(placeholder, answer)
    return notebook


def main() -> None:
    """Run every cell, reporting the first failure with its traceback."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passphrase", default="", help="Unused; kept so old invocations still work.")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    notebook = prepare(args.passphrase)
    with tempfile.TemporaryDirectory() as workdir:
        client = NotebookClient(notebook, timeout=args.timeout, resources={"metadata": {"path": workdir}})
        client.execute()

    executed = sum(1 for cell in notebook.cells if cell.cell_type == "code")
    print(f"All {executed} code cells executed without error.")


if __name__ == "__main__":
    main()
