"""TOWER station scene for Isaac Lab: two Nero arms with the evaluator's drive parameters plus an episode's blocks.

Isaac Lab modules can only be imported once the Isaac Sim app runs: import this module after ``AppLauncher``.

Conventions:
  * Scene units are meters. The station wrapper (``scripts/make_meter_assets.py``) converts the millimetre assets.
  * Recorded poses use ``x y z qx qy qz qw``; Isaac Lab uses ``w x y z`` quaternions.
  * A recorded arm state is 8 values ``[joint1..joint7, gripper opening width (m)]``; ``q16`` stacks left then back.
  * Simulated arms have 10 DOFs: 7 arm joints, a passive ``gripper`` joint and two fingers driven to +-width/2.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from tower_rl.data.episode import Episode  # noqa: F401  (re-exported: scripts import it from here)

PHYSICS_DT = 1.0 / 120.0
CONTROL_SUBSTEPS = 4  # recorded commands are a 30 Hz zero-order hold over 120 Hz physics
CONTROL_DT = PHYSICS_DT * CONTROL_SUBSTEPS
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
JOINT_NAMES = ARM_JOINTS + ["gripper", "gripper_joint1", "gripper_joint2"]
ARMS = ("left", "back")
ANG = 1e-6  # the evaluator's readback comes from a millimetre stage: angular quantities -> SI
LIN = 1e-3  # linear forces / velocities -> SI
# visual-only station parts that block views from outside the enclosure (no colliders)
OCCLUDERS = [
    "/World/envs/env_.*/Station/Model/StationModel/StaticCurtains",
    "/World/envs/env_.*/Station/Model/StationModel/tn__b4b08Ou03GRNY0xCb0/(StationFrameAndCabinet|UpperFrame_.*)",
]
# RTX settings recorded by the data generator (provenance_json["rendering"]); needs --enable_cameras
EVAL_RENDER_CFG = dict(
    antialiasing_mode="DLAA",
    dlss_mode=2,
    carb_settings={
        "/rtx/rendermode": "RaytracedLighting",
        "/rtx/directLighting/sampledLighting/samplesPerPixel": 16,
        "/rtx/shadows/sampleCount": 16,
        "/rtx/indirectDiffuse/fetchSampleCount": 16,
        "/rtx/reflections/sampledLighting/samplesPerPixel": 16,
        "/rtx/directLighting/domeLight/sampleCount": 16,
        "/rtx/post/motionblur/enabled": False,
    },
)


def arm_q_full(q8: torch.Tensor) -> torch.Tensor:
    """``[..., 8]`` recorded arm state -> ``[..., 10]`` simulated DOFs (passive gripper joint, fingers at +-w/2)."""
    w = q8[..., 7:8]
    return torch.cat([q8[..., :7], w, w / 2.0, -w / 2.0], dim=-1)


def actuators_from_readback(readback: dict, arm: str) -> dict[str, ImplicitActuatorCfg]:
    """Implicit actuators reproducing the evaluator's identified per-arm PD, armature and joint friction (in SI)."""
    rb = readback[arm]["readback"]

    def per_joint(key, scale, sub=None):
        vals = rb[key][0]
        return {n: float((vals[i][sub] if sub is not None else vals[i]) * scale) for i, n in enumerate(ARM_JOINTS)}

    return {
        "arm": ImplicitActuatorCfg(
            joint_names_expr=ARM_JOINTS,
            stiffness=per_joint("kp", ANG),
            damping=per_joint("kd", ANG),
            armature=per_joint("armatures", ANG),
            friction=per_joint("friction", ANG, 0),
            dynamic_friction=per_joint("friction", ANG, 1),
            viscous_friction=per_joint("friction", ANG, 2),
            effort_limit_sim=per_joint("max_efforts", ANG),
            velocity_limit_sim=per_joint("max_velocities", 1.0),
        ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["gripper_joint1", "gripper_joint2"],
            stiffness=float(rb["kp"][0][8]),
            damping=float(rb["kd"][0][8]),
            effort_limit_sim=float(rb["max_efforts"][0][8] * LIN),
            velocity_limit_sim=float(rb["max_velocities"][0][8] * LIN),
        ),
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=float(rb["max_efforts"][0][7] * LIN),
            velocity_limit_sim=float(rb["max_velocities"][0][7] * LIN),
        ),
    }


def _robot_prim(meta: dict, arm: str) -> str:
    return "{ENV_REGEX_NS}/Station" + meta["arms"][arm]["prim"][len("/TowerStation") :]


def robot_cfg(meta: dict, readback: dict, arm: str, q8_init) -> ArticulationCfg:
    """The arm already exists inside the station USD (``spawn=None``); its fixed root pose must match the station."""
    m = meta["arms"][arm]
    q = arm_q_full(torch.as_tensor(q8_init, dtype=torch.float32)).tolist()
    return ArticulationCfg(
        prim_path=_robot_prim(meta, arm),
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(m["world"]["pos_m"]),
            rot=tuple(m["world"]["quat_wxyz"]),
            joint_pos=dict(zip(JOINT_NAMES, q)),
            joint_vel={".*": 0.0},
        ),
        actuators=actuators_from_readback(readback, arm),
    )


BLOCK_SOLVER_ITERATIONS = [16, 4]  # position, velocity (scripts may override before building the scene)


def block_cfgs(meta: dict, episode: Episode, frame: int = 0) -> dict[str, RigidObjectCfg]:
    """One rigid block per recorded block id; foundation blocks are kinematic like in the evaluator."""
    poses = episode.arrays["sim__block_poses"][frame]
    cfgs = {}
    for i, bid in enumerate(episode.block_ids):
        p = poses[i]
        cfgs[bid] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Block_{bid}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=meta["block"]["wrapper"],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=bool(episode.block_fixed[i]),
                    solver_position_iteration_count=BLOCK_SOLVER_ITERATIONS[0],
                    solver_velocity_iteration_count=BLOCK_SOLVER_ITERATIONS[1],
                    max_depenetration_velocity=1.0,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=tuple(float(x) for x in p[:3]), rot=(float(p[6]), float(p[3]), float(p[4]), float(p[5]))
            ),
        )
    return cfgs


def evaluator_camera_cfgs(meta: dict, episode: Episode, scale: float = 1.0) -> dict[str, TiledCameraCfg]:
    """The evaluator's four cameras: intrinsics from the episode provenance, poses from the station/wrist calibration.

    ``cam_low`` / ``cam_high`` render the dataset's ``left_back_camera`` / ``right_front_camera`` rotated by 180 deg.
    """
    ks = episode.provenance["cameras"][0]
    mc = meta["cameras"]
    w, h = int(round(640 * scale)), int(round(480 * scale))

    def cam(prim_path, name, pos, rot, clip):
        k = ks[name]["intrinsic_matrix_k"]
        k = [k[0][0] * scale, 0.0, k[0][2] * scale, 0.0, k[1][1] * scale, k[1][2] * scale, 0.0, 0.0, 1.0]
        return TiledCameraCfg(
            prim_path=prim_path,
            data_types=["rgb"],
            width=w,
            height=h,
            offset=TiledCameraCfg.OffsetCfg(pos=tuple(pos), rot=tuple(rot), convention="opengl"),
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(intrinsic_matrix=k, width=w, height=h, clipping_range=clip),
        )

    return {
        "cam_high": cam("{ENV_REGEX_NS}/CamHigh", "high", mc["high_camera"]["pos_m"], mc["high_camera"]["quat_wxyz_usd_camera"], (0.001, 3.0)),
        "cam_low": cam("{ENV_REGEX_NS}/CamLow", "low", mc["low_camera"]["pos_m"], mc["low_camera"]["quat_wxyz_usd_camera"], (0.001, 3.0)),
        "cam_left_wrist": cam(
            _robot_prim(meta, "left") + "/gripper_base/CamWrist", "left_wrist",
            mc["left_wrist"]["pose_in_gripper_base_m"], mc["left_wrist"]["quat_in_gripper_base_wxyz"], (0.01, 3.0),
        ),
        "cam_back_wrist": cam(
            _robot_prim(meta, "back") + "/gripper_base/CamWrist", "back_wrist",
            mc["back_wrist"]["pose_in_gripper_base_m"], mc["back_wrist"]["quat_in_gripper_base_wxyz"], (0.01, 3.0),
        ),
    }


def make_scene_cfg(meta: dict, episode: Episode, num_envs: int, env_spacing: float = 3.0, extras: dict | None = None) -> InteractiveSceneCfg:
    """Station + both arms + the episode's blocks at frame 0; ``extras`` adds entities (cameras, lights) after them."""
    q0 = episode.arrays["observations__qpos"][0]
    readback = episode.readback

    @configclass
    class TowerSceneCfg(InteractiveSceneCfg):
        station = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Station", spawn=sim_utils.UsdFileCfg(usd_path=meta["wrapper"]))
        left = robot_cfg(meta, readback, "left", q0[:8])
        back = robot_cfg(meta, readback, "back", q0[8:])
        blocks = RigidObjectCollectionCfg(rigid_objects=block_cfgs(meta, episode))

    cfg_cls = TowerSceneCfg
    if extras:
        cfg_cls = configclass(type("TowerSceneExtrasCfg", (TowerSceneCfg,), dict(extras)))
    return cfg_cls(num_envs=num_envs, env_spacing=env_spacing, replicate_physics=True)


def load_station_meta(path: str | Path) -> dict:
    meta = json.loads(Path(path).read_text())
    for f in (meta["wrapper"], meta["block"]["wrapper"]):
        if not Path(f).is_file():
            raise FileNotFoundError(f"{f} missing: regenerate assets_m/ with scripts/make_meter_assets.py")
    return meta


def apply_friction(scene: InteractiveScene, finger: tuple[float, float] | None = None, block: tuple[float, float] | None = None):
    """Override static/dynamic friction of all robot shapes (``finger``) and of all blocks (``block``)."""
    if block is not None:
        blk = scene["blocks"]
        mp = blk.root_physx_view.get_material_properties()
        mp[..., 0], mp[..., 1] = block
        blk.root_physx_view.set_material_properties(mp, torch.arange(blk.root_physx_view.count, dtype=torch.int32))
    if finger is not None:
        for arm in ARMS:
            r = scene[arm]
            mp = r.root_physx_view.get_material_properties()  # [envs, shapes, 3]; finger pads dominate block contacts
            mp[..., 0], mp[..., 1] = finger
            r.root_physx_view.set_material_properties(mp, torch.arange(r.num_instances, dtype=torch.int32))


def set_arm_targets(scene: InteractiveScene, q16: torch.Tensor, env_ids: torch.Tensor | None = None):
    """Joint position targets from a ``[16]`` command (both arms) for all or the selected envs."""
    n = scene.num_envs if env_ids is None else len(env_ids)
    for i, arm in enumerate(ARMS):
        scene[arm].set_joint_position_target(arm_q_full(q16[8 * i : 8 * i + 8]).unsqueeze(0).repeat(n, 1), env_ids=env_ids)


def write_recorded_state(scene: InteractiveScene, episode: Episode, t: int, env_ids: torch.Tensor | None = None):
    """Teleport arms (state + targets) and blocks to recorded frame ``t``, with zero velocities."""
    dev = scene.device
    n = scene.num_envs if env_ids is None else len(env_ids)
    origins = scene.env_origins if env_ids is None else scene.env_origins[env_ids]
    q = torch.as_tensor(episode.arrays["observations__qpos"][t], dtype=torch.float32, device=dev)
    for i, arm in enumerate(ARMS):
        qf = arm_q_full(q[8 * i : 8 * i + 8]).unsqueeze(0).repeat(n, 1)
        scene[arm].write_joint_state_to_sim(qf, torch.zeros_like(qf), env_ids=env_ids)
        scene[arm].set_joint_position_target(qf, env_ids=env_ids)
    poses = torch.as_tensor(episode.arrays["sim__block_poses"][t], dtype=torch.float32, device=dev)
    st = torch.zeros(n, poses.shape[0], 13, device=dev)
    st[..., :3] = poses[:, :3] + origins[:, None, :]
    st[..., 3:7] = poses[:, [6, 3, 4, 5]]
    scene["blocks"].write_object_state_to_sim(st, env_ids=env_ids)
    scene.write_data_to_sim()


def settle_recorded_state(scene: InteractiveScene, sim, episode: Episode, t: int = 0, seconds: float = 1.0,
                          lift_m: float = 0.0005, env_ids: torch.Tensor | None = None):
    """Write recorded frame ``t`` with movable blocks ``lift_m`` higher and let them settle for ``seconds`` while the
    arms hold the recorded command (the generator also settles 1 s after a reset).

    Teleporting the recorded poses directly can leave the keel-hulled blocks slightly interpenetrating; PhysX then
    depenetrates them violently (8-19 mm jumps observed in Task 2-2 replays)."""
    write_recorded_state(scene, episode, t, env_ids)
    if lift_m > 0:
        blk = scene["blocks"]
        st = torch.cat([blk.data.object_link_pose_w, torch.zeros_like(blk.data.object_link_pose_w[..., :6])], -1).clone()
        movable = torch.tensor([not f for f in episode.block_fixed], device=scene.device)
        st[:, movable, 2] += lift_m
        ids = slice(None) if env_ids is None else env_ids
        blk.write_object_state_to_sim(st[ids], env_ids=env_ids)
    q = torch.as_tensor(episode.arrays["leader__qpos"][t], dtype=torch.float32, device=scene.device)
    for _ in range(int(round(seconds / PHYSICS_DT))):
        set_arm_targets(scene, q, env_ids)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(PHYSICS_DT)


def measured_q16(scene: InteractiveScene, env: int = 0) -> torch.Tensor:
    """Simulated state in the recorded layout (opening width = finger1 - finger2)."""
    out = []
    for arm in ARMS:
        q = scene[arm].data.joint_pos[env]
        out += [q[:7], (q[8] - q[9]).reshape(1)]
    return torch.cat(out)


class DelayedCommands:
    """Per-arm command delay in physics substeps (evaluator: left 5 ms, back 10 ms; 1 substep fits the data best)."""

    def __init__(self, first: torch.Tensor, delay_substeps: tuple[int, int]):
        self.delays = delay_substeps
        self.hist = [first] * (max(delay_substeps) + 1)

    def push(self, q16: torch.Tensor) -> torch.Tensor:
        """Add this substep's command and return the delayed ``[16]`` command to apply now."""
        self.hist.append(q16)
        self.hist.pop(0)
        return torch.cat([self.hist[-1 - self.delays[0]][:8], self.hist[-1 - self.delays[1]][8:]])
