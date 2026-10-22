"""The evaluator's four RGB cameras as pinhole models in the station frame. Isaac-free (numpy + the torch FK).

Image formation (measured 2026-10-20, BUILDLOG "Pose estimator v1"): the dataset JPEGs (Sep-9 and Sep-14 stations
alike) are pinhole images with the recorded intrinsics K, principal point at (cx, cy), taken from the published
station's camera poses; the two static cameras' images are delivered turned by 180 deg (the docs: the observation
arrays match the dataset JPEGs' orientation). Isaac Lab cameras ignore the principal point (aperture offsets are
unsupported) and use the mean of fx and fy, so our renders are the same views shifted by (cx - 319.5, cy - 239.5) px:
they are described by ``source="render"``.

A ``Camera`` describes an image as delivered: OpenCV axes (x right, y down, z forward), pixel centres at integer
coordinates, ``R_wc`` / ``t_wc`` the camera's orientation and centre in the station frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import torch

from tower_rl import geometry as geo
from tower_rl.config import load_yaml
from tower_rl.kinematics import ArmBase, ArmKinematics
from tower_rl.paths import STATION_JSON

CAMERA_IDS = ("left_back_camera", "right_front_camera", "left_wrist_camera", "back_wrist_camera")
STATIC_IDS = CAMERA_IDS[:2]
WRIST_ARM = {"left_wrist_camera": "left", "back_wrist_camera": "back"}
STATION_NAME = {"left_back_camera": "low", "right_front_camera": "high", "left_wrist_camera": "left_wrist",
                "back_wrist_camera": "back_wrist"}
WIDTH, HEIGHT = 640, 480
# Isaac renders: the optical axis through the image centre (pixel-centre coordinates); checked against rendered depth
RENDER_CENTRE = ((WIDTH - 1) / 2, (HEIGHT - 1) / 2)
SOURCES = ("evaluator", "render")

_USD_TO_CV = np.diag([1.0, -1.0, -1.0])  # USD camera axes (x right, y up, looking along -z) -> OpenCV axes
_TURN = np.diag([-1.0, -1.0, 1.0])  # the delivered static images are turned 180 deg about the optical axis


@dataclass
class Camera:
    name: str
    K: np.ndarray       # [3, 3] intrinsics of the delivered image
    R_wc: np.ndarray    # [3, 3] camera axes in the station frame
    t_wc: np.ndarray    # [3] camera centre in the station frame
    width: int = WIDTH
    height: int = HEIGHT
    meta: dict = field(default_factory=dict)

    def project(self, xw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixels ``[..., 2]`` and depths ``[...]`` (along the optical axis) of station-frame points ``[..., 3]``."""
        xc = (np.asarray(xw, dtype=np.float64) - self.t_wc) @ self.R_wc
        z = xc[..., 2]
        uv = xc[..., :2] / np.maximum(z, 1e-9)[..., None] * np.diag(self.K)[:2] + self.K[:2, 2]
        return uv, z

    def to_dict(self) -> dict:
        return {"name": self.name, "K": self.K.tolist(), "R_wc": self.R_wc.tolist(), "t_wc": self.t_wc.tolist(),
                "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, d: dict) -> Camera:
        return cls(d["name"], np.asarray(d["K"], dtype=np.float64), np.asarray(d["R_wc"], dtype=np.float64),
                   np.asarray(d["t_wc"], dtype=np.float64), int(d.get("width", WIDTH)), int(d.get("height", HEIGHT)))


def _quat_mat(q_wxyz) -> np.ndarray:
    return geo.quat_to_mat(torch.as_tensor(q_wxyz, dtype=torch.float64)).numpy()


def intrinsics(name: str, source: str = "evaluator") -> np.ndarray:
    """K of camera ``name`` before the static cameras' 180 deg turn."""
    assert source in SOURCES, source
    k = np.asarray(load_yaml("scene")["cameras"][STATION_NAME[name]]["K"], dtype=np.float64)
    # both renderers count the principal point from the pixel corner: (cx, cy) is (cx - 0.5, cy - 0.5) in pixel-centre
    # coordinates (dataset edges sit 0.5 px from the projected boxes otherwise; Isaac's centre 320 is 319.5)
    k[:2, 2] -= 0.5
    if source == "render":
        f = 0.5 * (k[0, 0] + k[1, 1])
        k = np.array([[f, 0.0, RENDER_CENTRE[0]], [0.0, f, RENDER_CENTRE[1]], [0.0, 0.0, 1.0]])
    return k


def _turned(k: np.ndarray, r_wc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = k.copy()
    k[0, 2] = WIDTH - 1 - k[0, 2]
    k[1, 2] = HEIGHT - 1 - k[1, 2]
    return k, r_wc @ _TURN


class CameraRig:
    """The four cameras for given joint angles: static poses from the station calibration, wrist poses from FK of the
    measured joints and the wrist mounts (both arms share one kinematic model)."""

    def __init__(self, source: str = "evaluator", station_json=STATION_JSON):
        assert source in SOURCES, source
        self.source = source
        meta = json.loads(open(station_json).read())["cameras"]
        self.kin = ArmKinematics(dtype=torch.float64)
        self.bases = {a: ArmBase(a, dtype=torch.float64) for a in ("left", "back")}
        self.static = {}
        for name in STATIC_IDS:
            c = meta[f"{STATION_NAME[name]}_camera"]
            k, r = _turned(intrinsics(name, source), _quat_mat(c["quat_wxyz_usd_camera"]) @ _USD_TO_CV)
            self.static[name] = Camera(name, k, r, np.asarray(c["pos_m"], dtype=np.float64))
        self.mount = {}
        for name in WRIST_ARM:
            c = meta[STATION_NAME[name]]
            self.mount[name] = (np.asarray(c["pose_in_gripper_base_m"], dtype=np.float64),
                                _quat_mat(c["quat_in_gripper_base_wxyz"]) @ _USD_TO_CV, intrinsics(name, source))

    def wrist(self, name: str, q7) -> Camera:
        arm = WRIST_ARM[name]
        p, r = self.kin.chain(torch.as_tensor(np.asarray(q7, dtype=np.float64).reshape(1, 7)))["gripper_base"]
        pw, qw = self.bases[arm].to_world(p, geo.mat_to_quat(r))
        r_wg = _quat_mat(qw[0])
        t, r_gc, k = self.mount[name]
        return Camera(name, k, r_wg @ r_gc, pw[0].numpy() + r_wg @ t)

    def cameras(self, q: np.ndarray) -> dict[str, Camera]:
        """All four cameras for measured joint angles ``q [2, 7]`` (left, back)."""
        q = np.asarray(q, dtype=np.float64).reshape(2, 7)
        out = dict(self.static)
        out["left_wrist_camera"] = self.wrist("left_wrist_camera", q[0])
        out["back_wrist_camera"] = self.wrist("back_wrist_camera", q[1])
        return {n: out[n] for n in CAMERA_IDS}


def camera_from_isaac(name: str, pos_w, quat_w_ros, env_origin, source: str = "render") -> Camera:
    """A camera from an Isaac Lab camera's rendered pose (``data.pos_w``, ``data.quat_w_ros``: OpenCV axes)."""
    r = _quat_mat(np.asarray(quat_w_ros, dtype=np.float64))
    k = intrinsics(name, source)
    if name in STATIC_IDS:
        k, r = _turned(k, r)
    return Camera(name, k, r, np.asarray(pos_w, dtype=np.float64) - np.asarray(env_origin, dtype=np.float64))


def deliver(name: str, img: np.ndarray) -> np.ndarray:
    """An Isaac render of camera ``name`` in the delivered orientation (static cameras turned by 180 deg)."""
    return np.ascontiguousarray(np.rot90(img, 2)) if name in STATIC_IDS else img
