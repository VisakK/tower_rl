"""Isaac Lab asset configs built from the evaluator's own assets (import after ``AppLauncher``).

* Arms: ``assets_m/nero_<arm>_station_m.usda`` (the installed robot prim, meters, base_link at the root) with the
  evaluator's runtime drive parameters from ``configs/robot/nero.json``.
* Blocks: ``assets_m/tower_block_station_m.usda`` (the station's Foundation01 prim the data generator clones: convex
  hull 0.15 x 0.0497 x 0.0283 m + keel points, 0.1056 kg, wood 0.40 / 0.23).
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg

from tower_rl.kinematics import load_model

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
JOINT_NAMES = ARM_JOINTS + ["gripper", "gripper_joint1", "gripper_joint2"]


def q_full(q7: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
    """``[..., 7]`` arm joints + ``[..., 1]`` opening width -> the 10 simulated DOFs (passive slider at 0.1 m)."""
    return torch.cat([q7, torch.full_like(width, 0.1), width / 2.0, -width / 2.0], dim=-1)


def arm_actuators(arm: str, model: dict | None = None, finger_effort: float | None = None) -> dict[str, ImplicitActuatorCfg]:
    d = (model or load_model())["drives"][arm]
    per = lambda key: dict(zip(ARM_JOINTS, d[key]))  # noqa: E731
    return {
        "arm": ImplicitActuatorCfg(
            joint_names_expr=ARM_JOINTS,
            stiffness=per("kp"), damping=per("kd"), armature=per("armature"),
            friction=per("friction_static"), dynamic_friction=per("friction_dynamic"), viscous_friction=per("friction_viscous"),
            effort_limit_sim=per("effort_limit"), velocity_limit_sim=per("velocity_limit"),
        ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["gripper_joint1", "gripper_joint2"],
            stiffness=d["finger_kp"], damping=d["finger_kd"],
            effort_limit_sim=finger_effort if finger_effort is not None else d["finger_effort_limit"],
            velocity_limit_sim=d["finger_velocity_limit"],
        ),
        # passive slider of the virtual gripper body (1 kg PhysX default mass, see configs/robot/nero.json)
        "gripper_passive": ImplicitActuatorCfg(joint_names_expr=["gripper"], stiffness=0.0, damping=0.0,
                                               effort_limit_sim=0.01, velocity_limit_sim=0.003),
    }


def arm_cfg(arm: str, meta: dict, prim_path: str = "{ENV_REGEX_NS}/Robot", q7=None, width: float = 0.1,
            contact_sensors: bool = True, model: dict | None = None, finger_effort: float | None = None) -> ArticulationCfg:
    """One Nero arm spawned from its standalone wrapper at the station base pose."""
    model = model or load_model()
    b = model["base_world"][arm]
    q7 = list(q7) if q7 is not None else [0.0, -0.8, 0.0, 1.75, 0.0, 0.0, 1.25]
    joint_pos = dict(zip(ARM_JOINTS, [float(x) for x in q7]))
    joint_pos.update({"gripper": 0.1, "gripper_joint1": width / 2, "gripper_joint2": -width / 2})
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(usd_path=meta["robot"][arm]["wrapper"], activate_contact_sensors=contact_sensors),
        init_state=ArticulationCfg.InitialStateCfg(pos=tuple(b["pos"]), rot=tuple(b["quat_wxyz"]), joint_pos=joint_pos,
                                                   joint_vel={".*": 0.0}),
        actuators=arm_actuators(arm, model, finger_effort=finger_effort),
    )


BLOCK_DIMS = (0.15, 0.05, 0.03)


def block_usd(meta: dict, collider: str = "hull") -> str:
    """Block wrapper: the station block (convex hull with keel points) or, for Task 5-3, the organizers' cuboid
    collision overlay (``assets_m/tower_block_cuboid_m.usda``)."""
    w = meta["block"]["wrapper"]
    return w if collider == "hull" else w.replace("tower_block_station_m.usda", "tower_block_cuboid_m.usda")


def block_cfg(meta: dict, prim_path: str, pose7_wxyz, kinematic: bool, phys: dict, collider: str = "hull",
              dims=None, mass: float | None = None, contact_sensors: bool = False) -> RigidObjectCfg:
    """One block (station block asset) at ``pose7_wxyz`` = [x y z qw qx qy qz]; ``dims`` scales it (Task 5's 32 mm
    side blocks; the generator scales the standard block), ``mass`` overrides the asset's 0.1056 kg."""
    bc = phys["block"]
    scale = None if dims is None else tuple(float(d) / s for d, s in zip(dims, BLOCK_DIMS))
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=block_usd(meta, collider),
            scale=scale,
            mass_props=None if mass is None else sim_utils.MassPropertiesCfg(mass=float(mass)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=kinematic,
                solver_position_iteration_count=bc["solver_position_iterations"],
                solver_velocity_iteration_count=bc["solver_velocity_iterations"],
                max_depenetration_velocity=bc["max_depenetration_velocity"],
            ),
            activate_contact_sensors=contact_sensors,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(float(x) for x in pose7_wxyz[:3]),
                                                  rot=tuple(float(x) for x in pose7_wxyz[3:7])),
    )


def physx_cfg(phys: dict) -> sim_utils.PhysxCfg:
    return sim_utils.PhysxCfg(**phys.get("physx", {}))


def sim_cfg(phys: dict, device: str = "cuda:0", render_interval: int | None = None) -> sim_utils.SimulationCfg:
    return sim_utils.SimulationCfg(
        dt=phys["dt"], render_interval=render_interval or phys["control_substeps"], device=device,
        gravity=tuple(phys["gravity"]), physx=physx_cfg(phys),
    )
