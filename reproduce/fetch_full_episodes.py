"""The ten full TOWER-SimData episodes (one per variant, with images; ~8 GB) that the orientation references
(scripts/build_orient_refs.py) and the pose estimator's sequence evaluation read: data/episodes_full/data/<variant>/.
The same files and dataset revision as for student_v1_3.

  python -I reproduce/fetch_full_episodes.py
"""
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO, REVISION = "tower-benchmark/TOWER-SimData", "1a59248e44866b1593e4ac8b851aa519c013220f"
FILES = {
    "task1-2_tower_transfer_2": "episode_000054_20260909T224804Z_cef746f8.hdf5",
    "task1-3_tower_transfer_3": "episode_001022_20260909T153526Z_254a6af4.hdf5",
    "task2-2_cross_tower_transfer_2": "episode_000001_20260910T023338Z_29641aae.hdf5",
    "task2-3_cross_tower_transfer_3": "episode_000001_20260910T024604Z_2a92f2b7.hdf5",
    "task3-2_vertical_stacking_2": "episode_000001_20260910T023237Z_306148c8.hdf5",
    "task3-3_vertical_stacking_3": "episode_000001_20260910T023237Z_72e36c99.hdf5",
    "task4-2_full_tower_pushdown_2": "episode_000070_20260914T072543Z_b12dce3a.hdf5",
    "task4-3_full_tower_pushdown_3": "episode_000014_20260914T071645Z_9de9a5a9.hdf5",
    "task5-2_tower_insertion_2": "episode_000057_20260914T140503Z_a1093848.hdf5",
    "task5-3_tower_insertion_3": "episode_000003_20260914T152644Z_19091492.hdf5",
}

out = Path(__file__).resolve().parents[1] / "data" / "episodes_full"
for variant, name in FILES.items():
    p = hf_hub_download(REPO, f"data/{variant}/{name}", repo_type="dataset", revision=REVISION, local_dir=out)
    print(p, file=sys.stderr)
