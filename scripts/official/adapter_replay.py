"""Plan-v1.1 §2.1 test 3: the v1.1 adapter on the saved official frames of a server run (CPU). For every episode with a
saved first frame: a fresh ``StudentAdapter`` built as the server builds it (the bundle's switcher, kp_v2 and
orientation references) gets the four first-frame images with the home joints and widths of the variant's dataset
episode (the official episodes start at home), three times (the adapter holds while unconfident on its first calls).
Reported per episode: the orientation choice, the confident movable blocks after the calls, the hold reasons, the chunk
the runtime returned; per variant: rotated / confident counts, against whether the evaluator rendered blocks
(``obs_check.py``'s edge measure).

  cd ~/Documents/tower_rl && CUDA_VISIBLE_DEVICES="" ~/Documents/moves/env_isaaclab/bin/python \\
      scripts/official/adapter_replay.py results/bundles/student_v1_1 results/student/server/official_1
"""
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

torch.set_num_threads(4)
import h5py  # noqa: E402

from tower_rl.server.student import StudentModels  # noqa: E402

CAMS = ["left_back_camera", "right_front_camera", "back_wrist_camera", "left_wrist_camera"]
BLOCKS_MIN = 0.006  # obs_check.py: back-wrist edge density above this = blocks rendered


def structure(img: np.ndarray) -> float:
    g = np.asarray(Image.fromarray(img).convert("L"), dtype=np.float32)
    h, w = g.shape
    c = g[int(0.05 * h):int(0.70 * h), int(0.25 * w):int(0.75 * w)]
    gx, gy = np.abs(np.diff(c, axis=1)), np.abs(np.diff(c, axis=0))
    return float((gx[:-1] + gy[:, :-1]).mean() / (c.mean() + 1e-6))


def home(variant: str):
    f = sorted(Path(f"data/episodes_full/data/{variant}").glob("*.hdf5"))[0]
    with h5py.File(f, "r") as h:
        qp = h["observations/qpos"][0]
    return np.stack([qp[0:7], qp[8:15]]).astype("<f4"), np.array([qp[7], qp[15]], dtype="<f4")


def main():
    bundle, run = Path(sys.argv[1]), Path(sys.argv[2])
    track = json.loads(Path("data/challenge/track.json").read_text())
    instr = {v["id"]: v["instruction"] for v in track["variants"]}
    models = StudentModels(bundle, track, device="cpu", verify=True)
    print(f"bundle {bundle.name}: orientation references {len(models.orient_refs or {})}")
    tally = defaultdict(lambda: defaultdict(int))
    for d in sorted(glob.glob(str(run / "episodes" / "*" / "*_images"))):
        d = Path(d)
        files = {c: d / f"00001_{c}.jpg" for c in CAMS}
        if not all(f.exists() for f in files.values()):
            continue
        variant = d.parent.name
        imgs = {c: np.asarray(Image.open(f).convert("RGB")) for c, f in files.items()}
        if all(int(a.max()) <= 8 for a in imgs.values()):
            continue  # the qualification check's zero images
        rendered = structure(imgs["back_wrist_camera"]) >= BLOCKS_MIN
        q, w = home(variant)
        a = models.adapter(log=lambda *x: None)
        holds, out = [], None
        for call in range(4):
            obs = {"images": imgs, "joint_positions": q, "gripper_widths_m": w, "task_variant": variant,
                   "instruction": instr.get(variant, ""), "sim_time_s": 0.1 * call, "step_index": call}
            recs = []
            a.recorder = lambda rec, im: recs.append(rec)
            out = a.infer(obs)
            holds.append(recs[-1].get("hold"))
            conf = np.asarray(recs[-1].get("conf", []))
        fixed = a.est.fixed.cpu().numpy().astype(bool)
        n_conf = int((conf[~fixed] >= 0.5).sum()) if len(conf) else 0
        o = a.orient_info or {}
        t = tally[variant]
        t["episodes"] += 1
        t["rendered"] += rendered
        t["rotated"] += bool(o.get("rotate"))
        t["by_correlation"] += o.get("how") == "correlation"
        t["confident"] += n_conf > 0
        t["confident_rendered"] += (n_conf > 0) and rendered
        t["errors"] += a.errors
        print(f"{variant[:7]} {d.name[:8]} rendered {int(rendered)} rotate {o.get('rotate')} ({o.get('how')}, "
              f"same {o.get('same')} rot180 {o.get('rot180')}) confident movable {n_conf}/{int((~fixed).sum())} "
              f"holds {holds} blind {a.blind} chunk {list(out.shape)} errors {a.errors}", flush=True)
    print("\nper variant: episodes, rendered, rotated, by correlation, confident (of rendered), errors")
    for v, t in sorted(tally.items()):
        print(f"  {v}: {t['episodes']}, {t['rendered']}, {t['rotated']}, {t['by_correlation']}, "
              f"{t['confident']} ({t['confident_rendered']} / {t['rendered']}), {t['errors']}")
    os._exit(0)


if __name__ == "__main__":
    main()
