"""Step 2 acceptance (a): both scenes open headless and step 300 control ticks without NaN.

Each scene runs in its own process (one Isaac Sim app per process): ``pytest --isaac tests/test_scenes.py``.
"""

import json
import os
import subprocess
import sys

import pytest

from tower_rl.paths import REPO_ROOT, RESULTS_DIR

pytestmark = pytest.mark.isaac
CASES = {
    "skill_left": ["--scene", "skill", "--arm", "left", "--num_envs", "64"],
    "skill_back": ["--scene", "skill", "--arm", "back", "--num_envs", "64"],
    "station": ["--scene", "station", "--num_envs", "2"],
}


@pytest.mark.parametrize("case", list(CASES))
def test_scene_smoke(case):
    out = RESULTS_DIR / "scenes" / f"pytest_{case}.json"
    if out.exists():
        out.unlink()
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "smoke_scenes.py"), "--headless", *CASES[case], "--ticks", "300", "--out", str(out)]
    subprocess.run(cmd, cwd=REPO_ROOT, timeout=900, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=dict(os.environ))
    rep = json.loads(out.read_text())
    assert rep["nan_free"]
    if case.startswith("skill"):
        assert rep["fk_link7_err_mm_max"] < 0.05
        assert rep["hold_pinch_err_mm"][1] < 5.0      # sag-compensated hold of the pre-grasp pose
        assert rep["tower_drift_mm_max"] < 5.0        # keel-hull creep only
        assert rep["foundation_drift_mm_max"] < 0.01  # kinematic foundations
