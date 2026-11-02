"""Offline (CPU): kp_v2 + fit on an official episode's first frames, as delivered vs with the static cameras turned
180 deg; q / widths from the public dataset's first frame (same home pose); block poses compared with the dataset's.

  cd ~/Documents/tower_rl_snapshots/server_v1   # or the repo root
  CUDA_VISIBLE_DEVICES="" ~/Documents/moves/env_isaaclab/bin/python ~/Documents/tower_rl/scripts/official/orient_test.py \
      <variant> results/student/server/official_1/episodes/<variant>/<trial>_images
"""
import json, sys
from pathlib import Path
import numpy as np
import h5py
from PIL import Image
sys.path.insert(0, ".")
from tower_rl.server.student import StudentModels  # noqa: E402

variant, ep_dir = sys.argv[1], Path(sys.argv[2])
h5 = sorted(Path(f"data/episodes_full/data/{variant}").glob("*.hdf5"))[0]
with h5py.File(h5, "r") as f:
    qpos = f["observations/qpos"][0]
    ids = json.loads(f["sim"].attrs["block_ids"])
    bp0 = f["sim/block_poses"][0]
q = np.stack([qpos[0:7], qpos[8:15]])
w = np.array([qpos[7], qpos[15]])
track = json.loads(Path("data/challenge/track.json").read_text())
m = StudentModels("/home/visakii/Documents/tower_rl/results/bundles/student_v1", track, device="cpu", verify=False)
cams = ["left_back_camera", "right_front_camera", "back_wrist_camera", "left_wrist_camera"]
raw = {c: np.asarray(Image.open(ep_dir / f"00001_{c}.jpg").convert("RGB")) for c in cams}
for label, turn in (("as delivered", False), ("static cameras turned 180", True)):
    imgs = {c: (np.ascontiguousarray(np.rot90(a, 2)) if turn and c in ("left_back_camera", "right_front_camera") else a)
            for c, a in raw.items()}
    est = m.estimator()
    est.reset(variant)
    for _ in range(3):  # a few calls on the same frame (the fit's state settles)
        pos, quat, conf = est(imgs, q, w)
    print(f"== {label}: confidences " + " ".join(f"{b}:{c:.2f}" for b, c in zip(est.block_ids, conf)))
    for i, b in enumerate(est.block_ids):
        if b in ids:
            j = ids.index(b)
            d = np.linalg.norm(np.asarray(pos[i]) - bp0[j][:3]) * 1000
            print(f"   {b:22s} est {np.round(pos[i], 3)} dataset-ep0 {np.round(bp0[j][:3], 3)} diff {d:6.1f} mm")
