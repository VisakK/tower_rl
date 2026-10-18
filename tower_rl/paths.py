"""Repository paths. Free of Isaac imports so scripts can use them before the Isaac Sim app starts."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
EPISODE_DIR = DATA_DIR / "episodes"
STATION_JSON = REPO_ROOT / "assets_m" / "tower_station_m.json"
RESULTS_DIR = REPO_ROOT / "results"


def resolve_episode(name_or_path: str, attrs: str | None = None) -> tuple[Path, Path]:
    """Episode npz + attrs json from a name in data/episodes (e.g. ``t23_ep1``) or an explicit npz path."""
    npz = Path(name_or_path)
    if npz.suffix != ".npz":
        npz = EPISODE_DIR / f"{name_or_path}.npz"
    attrs_path = Path(attrs) if attrs else npz.with_name(f"{npz.stem}_attrs.json")
    for p in (npz, attrs_path):
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found (fetch episodes with scripts/peek_hdf5_remote.py)")
    return npz, attrs_path
