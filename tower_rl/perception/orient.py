"""Per-episode orientation of the two fixed cameras (Plan-v1.1 §2.1; Lessons-v1 §3).

kp_v2 learned the public dataset's JPEG orientation. The official evaluator delivers Tasks 1-3's ``left_back_camera`` /
``right_front_camera`` arrays rotated 180 deg relative to it (the raw capture), and Tasks 4-5's as the dataset; the
wrist cameras always as the dataset. ``OrientationChooser`` decides once per episode, on the first call with images:
each fixed camera's delivered image and its 180 deg rotation are correlated with a stored reference frame of the
variant (the dataset episode's first frame, 160 x 120 normalised grayscale; the station and the arms at home dominate,
so it works when the evaluator renders no blocks and under the lighting cases: official run 1 gave 0.73-1.00 for the
right orientation, <= 0.06 for the wrong one). The better orientation wins if its mean correlation over the two cameras
is >= ``min_corr`` and beats the other by >= ``min_margin``; otherwise the per-task fallback (rotate Tasks 1-3, keep
Tasks 4-5). The choice is locked for the episode; ``apply`` turns the fixed cameras of every later call.

``build_refs`` writes the references from the dataset HDF5 files (``scripts/build_orient_refs.py``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FIXED = ("left_back_camera", "right_front_camera")
SIZE = (160, 120)  # (width, height) of the correlated images
FALLBACK_ROTATE = ("task1", "task2", "task3")  # official run 1: Tasks 1-3 delivered rotated, Tasks 4-5 as the dataset


def small_gray(img: np.ndarray) -> np.ndarray:
    """``img`` [H, W, 3] uint8 RGB -> [120, 160] float32, zero mean and unit variance (area-averaged: a 180 deg rotation
    of the result equals the result of the rotated image)."""
    import cv2

    g = cv2.cvtColor(np.ascontiguousarray(img, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.resize(g, SIZE, interpolation=cv2.INTER_AREA)
    return (g - g.mean()) / (g.std() + 1e-6)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float((a * b).mean())


def fallback_rotate(variant: str) -> bool:
    return any(variant.startswith(p) for p in FALLBACK_ROTATE)


class OrientationChooser:
    def __init__(self, refs: dict[str, np.ndarray] | None, min_corr: float = 0.5, min_margin: float = 0.3):
        """``refs``: ``"<variant>/<camera>"`` -> [120, 160] normalised grayscale reference (``load_refs``)."""
        self.refs = refs or {}
        self.min_corr, self.min_margin = min_corr, min_margin
        self.reset()

    def reset(self):
        self.rotate: bool | None = None  # None until decided
        self.info: dict = {}

    @property
    def decided(self) -> bool:
        return self.rotate is not None

    def decide(self, variant: str, images: dict[str, np.ndarray]) -> dict:
        """Choose and lock the orientation of the fixed cameras from this call's ``images`` (delivered arrays)."""
        same, rot, per_cam = [], [], {}
        for c in FIXED:
            ref = self.refs.get(f"{variant}/{c}")
            img = images.get(c)
            if ref is None or img is None or not np.asarray(img).any():
                continue
            g = small_gray(img)
            s, r = corr(g, ref), corr(g[::-1, ::-1], ref)
            same.append(s)
            rot.append(r)
            per_cam[c] = {"same": round(s, 3), "rot180": round(r, 3)}
        fb = fallback_rotate(variant)
        if same:
            s, r = float(np.mean(same)), float(np.mean(rot))
            best_rot = r > s
            hi, lo = (r, s) if best_rot else (s, r)
            sure = hi >= self.min_corr and hi - lo >= self.min_margin
            self.rotate = best_rot if sure else fb
            how = "correlation" if sure else "fallback"
        else:
            s = r = None
            self.rotate, how = fb, "fallback (no reference or no image)"
        self.info = {"rotate": bool(self.rotate), "how": how, "same": None if s is None else round(s, 3),
                     "rot180": None if r is None else round(r, 3), "cams": per_cam, "fallback_rotate": fb}
        return self.info

    def apply(self, images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """The images in kp_v2's convention: the fixed cameras turned 180 deg if so decided (wrist cameras unchanged)."""
        if not self.rotate:
            return images
        return {c: (np.ascontiguousarray(np.rot90(img, 2)) if c in FIXED else img) for c, img in images.items()}


def load_refs(path: str | Path | None) -> dict[str, np.ndarray]:
    if path is None or not Path(path).exists():
        return {}
    with np.load(path) as z:
        return {k: z[k].astype(np.float32) for k in z.files}


def build_refs(hdf5_paths: list[str | Path], frame: int = 0) -> dict[str, np.ndarray]:
    """``"<variant>/<camera>"`` -> reference, from each dataset episode's ``frame`` (the variant is the parent dir name;
    the stored JPEGs are in kp_v2's convention, i.e. the dataset orientation)."""
    import cv2
    import h5py

    out = {}
    for p in hdf5_paths:
        p = Path(p)
        variant = p.parent.name
        with h5py.File(p, "r") as f:
            for c in FIXED:
                raw = bytes(f[f"observations/images/{c}"][frame])
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
                out[f"{variant}/{c}"] = small_gray(img)
    return out
