#!/usr/bin/env bash
# Stage 1 (reproduce/README.md §3): downloads and generated assets; data/ and assets_m/ are not in git (~18 GB with stage 3's
# data). TOWER-Assets and the SimData metadata at the revisions used, three example episodes (tests, validation), the
# Step 1 subset (10 episodes per variant, non-image arrays read by HTTP range requests), the ten full episodes the
# orientation references read, and the meter-unit asset wrappers (absolute paths: rerun the last command after moving
# the repository). The configs Step 1 derived from the subset (configs/layouts/, configs/skills/posture_priors.yaml,
# configs/robot/nero.json) are committed and used as they are.
# Usage: reproduce/1_data.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
run "$PY" -I -c "
from huggingface_hub import snapshot_download
snapshot_download('tower-benchmark/TOWER-Assets', repo_type='dataset', local_dir='data/TOWER-Assets',
                  revision='538d8b1fb582a4c3fc8c5a2fb2bf3a82931e9a4c')
snapshot_download('tower-benchmark/TOWER-SimData', repo_type='dataset', local_dir='data/SimData-meta',
                  revision='1a59248e44866b1593e4ac8b851aa519c013220f',
                  allow_patterns=['README.md', 'schema/*', 'MANIFEST.jsonl', 'CHECKSUMS.sha256', 'CITATION.cff', 'LICENSE', 'previews/*'])
"
run "$PY" -I scripts/peek_hdf5_remote.py data/task2-3_cross_tower_transfer_3/episode_000001_20260910T024604Z_2a92f2b7.hdf5 data/episodes/t23_ep1.npz
run "$PY" -I scripts/peek_hdf5_remote.py data/task1-3_tower_transfer_3/episode_001022_20260909T153526Z_254a6af4.hdf5 data/episodes/t13_ep.npz
run "$PY" -I scripts/peek_hdf5_remote.py data/task3-3_vertical_stacking_3/episode_000001_20260910T023237Z_72e36c99.hdf5 data/episodes/t33_ep.npz
run "$PY" -I scripts/fetch_subset.py --per_variant 10 --variants task1-2_tower_transfer_2 task1-3_tower_transfer_3 \
    task2-2_cross_tower_transfer_2 task2-3_cross_tower_transfer_3 task3-2_vertical_stacking_2 task3-3_vertical_stacking_3 \
    task4-2_full_tower_pushdown_2 task4-3_full_tower_pushdown_3 task5-2_tower_insertion_2 task5-3_tower_insertion_3
run "$PY" -I reproduce/fetch_full_episodes.py
run "$PY" -I scripts/make_meter_assets.py data/TOWER-Assets assets_m station.usdz
