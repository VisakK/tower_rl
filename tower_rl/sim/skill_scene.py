"""Light skill scene (Step 2.2): one Nero arm, the station tabletop collider, both pads' foundations and up to six
movable blocks; no cameras, no station mesh. Import after ``AppLauncher``.

The arm sits at its station base pose, so reachability and wrist angles match the station; foundations are kinematic
(as in the evaluator). Movable blocks live in slots ``m0..m5`` that the env fills per reset; unused slots are parked
on a shelf at x = 1.35 m, out of reach of both arms. Envs do not collide with each other
(``filter_collisions``), so the spacing only matters for viewing.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCollectionCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from tower_rl.config import load_yaml
from tower_rl.scene import load_station_meta
from tower_rl.paths import STATION_JSON
from tower_rl.sim.assets import arm_cfg, block_cfg

FOUNDATION_SLOTS = ["src_base_0", "src_base_1", "src_base_2", "dst_base_0", "dst_base_1", "dst_base_2"]
MAX_MOVABLE = 6
MOVABLE_SLOTS = [f"m{i}" for i in range(MAX_MOVABLE)]
BLOCK_SLOTS = FOUNDATION_SLOTS + MOVABLE_SLOTS
PARK_X = 1.35
TABLE_SIZE = (0.96, 0.66, 0.025)   # the station's ReplayTabletopCollision: x 0..0.96, y -0.66..0, top at z = 0


def park_pose(i: int) -> list[float]:
    """Parking spot of movable slot ``i`` (on the shelf, >= 1 m from both arm bases)."""
    return [PARK_X, -0.12 - 0.08 * i, 0.0141, 1.0, 0.0, 0.0, 0.0]


def _static_box(prim_path: str, size, center, color) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(size=size, collision_props=sim_utils.CollisionPropertiesCfg(),
                                  physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.4),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=center),
    )


def movable_slots(n: int = MAX_MOVABLE) -> list[str]:
    return [f"m{i}" for i in range(n)]


def make_skill_scene_cfg(arm: str, num_envs: int, env_spacing: float = 2.5, contact_sensor: bool = True,
                         dome_light: float = 0.0, max_movable: int = MAX_MOVABLE, finger_effort: float | None = None,
                         slot_dims: dict | None = None, slot_mass: dict | None = None, collider: str = "hull") -> InteractiveSceneCfg:
    """``max_movable`` slots m0.. (Task 4: 9 for a 3-layer full tower); ``slot_dims`` / ``slot_mass`` per slot index
    (Task 5's 32 mm side blocks); ``collider`` hull | cuboid (Task 5-3); ``finger_effort`` (Task 4/5: 3 N)."""
    meta = load_station_meta(STATION_JSON)
    phys = load_yaml("physics")
    slots = movable_slots(max_movable)
    slot_dims, slot_mass = slot_dims or {}, slot_mass or {}
    blocks = {}
    for i, slot in enumerate(FOUNDATION_SLOTS):  # placeholder poses; the env writes the real ones at reset
        blocks[slot] = block_cfg(meta, f"{{ENV_REGEX_NS}}/Block_{slot}", [0.3 + 0.06 * i, -0.9, 0.015, 1, 0, 0, 0], True, phys,
                                 collider=collider)
    for i, slot in enumerate(slots):
        blocks[slot] = block_cfg(meta, f"{{ENV_REGEX_NS}}/Block_{slot}", park_pose(i), False, phys, collider=collider,
                                 dims=slot_dims.get(i), mass=slot_mass.get(i))
    # the shelf holds every parked slot (6 slots: the original 0.62 m shelf, unchanged for the grasp / place envs)
    shelf_len = 0.62 if max_movable <= MAX_MOVABLE else 0.08 * max_movable + 0.16
    attrs = {
        "table": _static_box("{ENV_REGEX_NS}/Table", TABLE_SIZE, (0.48, -0.33, -0.0125), (0.55, 0.55, 0.6)),
        "shelf": _static_box("{ENV_REGEX_NS}/Shelf", (0.2, shelf_len, 0.025),
                             (PARK_X, -0.35 if max_movable <= MAX_MOVABLE else -0.04 - shelf_len / 2, -0.0125), (0.4, 0.4, 0.45)),
        "robot": arm_cfg(arm, meta, contact_sensors=contact_sensor, finger_effort=finger_effort),
        "blocks": RigidObjectCollectionCfg(rigid_objects=blocks),
    }
    if contact_sensor:  # one sensor per finger: PhysX needs one filter match per sensor body and env
        for f in (1, 2):
            attrs[f"finger{f}_contact"] = ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/gripper_link{f}",
                filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Block_{s}" for s in slots],
                update_period=0.0, history_length=0,
            )
    if dome_light > 0:
        attrs["dome"] = AssetBaseCfg(prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(intensity=dome_light))
    cls = configclass(type("SkillSceneCfg", (InteractiveSceneCfg,), attrs))
    return cls(num_envs=num_envs, env_spacing=env_spacing, replicate_physics=True)
