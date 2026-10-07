"""Locate the repository's data folder."""

from pathlib import Path


def data_dir():
    """Return the repository's `data/` folder.

    Searches the current working directory and its parents, then the folders above this file
    (which lie inside the repository when `opwater` is installed in editable mode).
    """
    cwd = Path.cwd().resolve()
    for folder in (cwd, *cwd.parents, *Path(__file__).resolve().parents):
        if (folder / "data" / "networks").is_dir():
            return folder / "data"
    raise FileNotFoundError(
        "Could not find the 'data/' folder. Open the notebook from inside the repository, "
        "or install `opwater` in editable mode (`pip install -e`)."
    )


def network_file(name):
    """Return the path to an EPANET .inp file in `data/networks/`."""
    return data_dir() / "networks" / name
