"""Reference frames for the per-episode orientation choice of the fixed cameras (``tower_rl.perception.orient``,
Plan-v1.1 §2.1): the first frame of each variant's dataset episode (``data/episodes_full/data/<variant>/*.hdf5``),
``left_back_camera`` and ``right_front_camera``, 160 x 120 normalised grayscale, in kp_v2's convention (the dataset
orientation). Then checks the choice on every official first frame of a server run: rotate in Tasks 1-3, keep Tasks 4-5.

  python scripts/build_orient_refs.py [--out results/perception/orient_refs.npz] [--check results/student/server/official_1]
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower_rl.perception.orient import FIXED, OrientationChooser, build_refs, fallback_rotate, load_refs  # noqa: E402
from tower_rl.paths import RESULTS_DIR, REPO_ROOT  # noqa: E402


def check(run: Path, refs: dict) -> int:
    """Every episode of ``run`` with a saved first frame: the choice vs the per-task rule; returns the number wrong."""
    from PIL import Image

    wrong, n, worst = 0, 0, {}
    for d in sorted(glob.glob(str(run / "episodes" / "*" / "*_images"))):
        d = Path(d)
        files = {c: d / f"00001_{c}.jpg" for c in FIXED}
        if not all(f.exists() for f in files.values()):
            continue
        variant = d.parent.name
        imgs = {c: np.asarray(Image.open(f).convert("RGB")) for c, f in files.items()}
        if all(int(a.max()) <= 8 for a in imgs.values()):  # the qualification check's zero images (JPEG noise): the
            continue                                       # adapter holds those ("blank") without deciding
        ch = OrientationChooser(refs)
        info = ch.decide(variant, imgs)
        n += 1
        expect = fallback_rotate(variant)
        ok = info["rotate"] == expect and info["how"] == "correlation"
        wrong += not ok
        w = worst.setdefault(variant, [9.0, 9.0, 0])
        hi, lo = (info["rot180"], info["same"]) if expect else (info["same"], info["rot180"])
        w[0], w[1], w[2] = min(w[0], hi), min(w[1], hi - lo), w[2] + 1
        if not ok:
            print(f"  WRONG {variant} {d.name[:8]}: {info}")
    for v, (hi, margin, k) in sorted(worst.items()):
        print(f"  {v}: {k} episodes, right orientation corr >= {hi:.2f}, margin >= {margin:.2f}")
    print(f"checked {n} first frames, {wrong} not chosen by correlation as the per-task rule says")
    return wrong


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS_DIR / "perception" / "orient_refs.npz"))
    ap.add_argument("--check", default=None, help="a server run dir (results/student/server/<name>) to check")
    args = ap.parse_args()
    paths = sorted(glob.glob(str(REPO_ROOT / "data" / "episodes_full" / "data" / "*" / "*.hdf5")))
    refs = build_refs(paths)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **refs)
    print(f"wrote {args.out}: {len(refs)} references from {len(paths)} episodes")
    if args.check:
        sys.exit(1 if check(Path(args.check), load_refs(args.out)) else 0)


if __name__ == "__main__":
    main()
