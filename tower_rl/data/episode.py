"""One recorded TOWER-SimData episode: non-image arrays (npz) + HDF5 attributes (json). No Isaac imports.

Files come from ``scripts/peek_hdf5_remote.py`` / ``scripts/fetch_subset.py``: ``<name>.npz`` holds the datasets with
``/`` replaced by ``__`` and ``<name>_attrs.json`` the root attributes plus ``__group__<group>`` attribute maps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ARMS = ("left", "back")
ARM_INDEX = {"left": 0, "back": 1}  # target_mapping uses arm 0 = left, 1 = back


@dataclass
class Episode:
    """Non-image arrays and HDF5 attributes of one recorded episode."""

    arrays: dict[str, np.ndarray]
    attrs: dict
    name: str = ""
    _cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, npz_path: str | Path, attrs_path: str | Path | None = None) -> Episode:
        npz_path = Path(npz_path)
        attrs_path = Path(attrs_path) if attrs_path else npz_path.with_name(f"{npz_path.stem}_attrs.json")
        with np.load(npz_path) as z:
            arrays = {k: z[k] for k in z.files}
        return cls(arrays, json.loads(attrs_path.read_text()), name=npz_path.stem)

    # ------------------------------------------------------------------ attributes
    def root_attr(self, key: str, default=None):
        v = self.attrs.get(key, default)
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v
        return v

    def sim_attr(self, key: str):
        """Attribute of the HDF5 ``sim`` group, JSON-decoded when it is a JSON string (cached)."""
        if key not in self._cache:
            v = self.attrs["__group__sim"][key]
            try:
                v = json.loads(v) if isinstance(v, str) else v
            except json.JSONDecodeError:
                pass
            self._cache[key] = v
        return self._cache[key]

    @property
    def num_frames(self) -> int:
        return self.arrays["observations__qpos"].shape[0]

    @property
    def block_ids(self) -> list[str]:
        return self.sim_attr("block_ids")

    @property
    def block_fixed(self) -> list[bool]:
        return self.sim_attr("block_fixed_on_table")

    @property
    def provenance(self) -> dict:
        return self.sim_attr("provenance_json")

    @property
    def readback(self) -> dict:
        """Drive parameters the evaluator applied at runtime (millimetre-stage units)."""
        return self.provenance["calibration"]["runtime_readback"]

    @property
    def config(self) -> dict:
        return self.sim_attr("config_json")

    @property
    def events(self) -> list[dict]:
        return self.sim_attr("events_json")

    @property
    def motion_choices(self) -> list[dict]:
        return self.sim_attr("motion_choices_json")

    @property
    def layout_metadata(self) -> dict:
        return self.sim_attr("layout_metadata_json")

    @property
    def outcome(self) -> dict:
        return self.sim_attr("outcome_json")

    @property
    def phase_names(self) -> list[str]:
        return self.sim_attr("phase_names")

    @property
    def target_mapping(self) -> list[dict]:
        return self.sim_attr("target_mapping")

    @property
    def instruction(self) -> str:
        return self.attrs.get("instruction", "")

    @property
    def task(self) -> str:
        return self.attrs.get("public_task") or self.attrs.get("task_name", "")

    @property
    def task_name(self) -> str:
        """Generator task name, e.g. ``cross_tower_transfer_3``."""
        return self.attrs.get("task_name", "")

    @property
    def variant(self) -> str:
        """Dataset folder name, e.g. ``task2-3_cross_tower_transfer_3``."""
        return f"{self.attrs.get('public_task', '')}_{self.attrs.get('task_name', '')}"

    # ------------------------------------------------------------------ arrays (shorthands)
    @property
    def qpos_cmd(self) -> np.ndarray:
        """``leader/qpos`` [T, 16]: commanded joints + gripper widths (m), left then back."""
        return self.arrays["leader__qpos"]

    @property
    def qpos_applied(self) -> np.ndarray:
        """What the generator's simulator actually received: ``sim/controller_qpos`` where recorded (Task 4/5: while
        gripping its controller holds ~43.8 mm although ``leader/qpos`` says 0; the arm channels are identical), else
        ``leader/qpos``."""
        return self.arrays.get("sim__controller_qpos", self.arrays["leader__qpos"])

    @property
    def qpos_obs(self) -> np.ndarray:
        return self.arrays["observations__qpos"]

    @property
    def eef_obs(self) -> np.ndarray:
        """``observations/eef`` [T, 14]: link7 pose in the arm's base_link frame (xyz + xyzw) per arm."""
        return self.arrays["observations__eef"]

    @property
    def eef_cmd(self) -> np.ndarray:
        return self.arrays["leader__eef"]

    @property
    def block_poses(self) -> np.ndarray:
        """``sim/block_poses`` [T, B, 7] in the station frame (xyz + xyzw)."""
        return self.arrays["sim__block_poses"]

    @property
    def time(self) -> np.ndarray:
        return self.arrays["sim__time"]

    @property
    def success(self) -> np.ndarray:
        return self.arrays["sim__success"]

    @property
    def terminal(self) -> np.ndarray:
        return self.arrays["sim__terminal"]
