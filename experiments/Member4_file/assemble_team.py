"""Validate or execute the team's consolidated notebook and save a local copy."""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "submission" / "DataVinci.ipynb"


def assemble(output=None, execute=False):
    document = nbformat.read(FINAL, as_version=4)
    nbformat.validate(document)
    if execute:
        NotebookClient(document, timeout=900, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute()
    path = output or ROOT / "local_artifacts" / "DataVinci_Final.ipynb"
    path = path.resolve()
    if path == FINAL:
        raise ValueError("Choose a separate path for the validated notebook copy")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        nbformat.write(document, stream)
    print(f"{'Executed' if execute else 'Validated notebook format'}: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    assemble(arguments.output, arguments.execute)
