"""Extract the Nero arm model (kinematic chain, link inertials, base poses, evaluator drives) into one JSON file.

The chain is read from the meter-unit station wrapper with plain ``pxr`` (no Isaac Sim app); the drive parameters come
from an episode's runtime readback (identical in all 331 Task1-3 episodes). The JSON is what
``tower_rl.kinematics`` loads, so FK/IK/gravity at inference never need the simulator.

Usage: python scripts/extract_robot_model.py [--episode t23_ep1] [--out configs/robot/nero.json]
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pxr import Usd, UsdPhysics  # noqa: E402

from tower_rl.data.episode import Episode  # noqa: E402
from tower_rl.paths import REPO_ROOT, STATION_JSON, resolve_episode  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--episode", default="t23_ep1")
ap.add_argument("--out", default=str(REPO_ROOT / "configs" / "robot" / "nero.json"))
args = ap.parse_args()

meta = json.loads(Path(STATION_JSON).read_text())
stage = Usd.Stage.Open(meta["wrapper"])
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
FIXED = ["gripper_flange_joint", "gripper_base_joint"]
FINGERS = ["gripper_joint1", "gripper_joint2"]


def quat_wxyz(q):
    return [float(q.GetReal())] + [float(x) for x in q.GetImaginary()]


def joint_record(prim):
    j = UsdPhysics.Joint(prim)
    rec = {
        "name": prim.GetName(),
        "type": prim.GetTypeName(),
        "parent": [str(t).split("/")[-1] for t in j.GetBody0Rel().GetTargets()][0],
        "child": [str(t).split("/")[-1] for t in j.GetBody1Rel().GetTargets()][0],
        "pos0": [float(x) for x in j.GetLocalPos0Attr().Get()],
        "rot0_wxyz": quat_wxyz(j.GetLocalRot0Attr().Get()),
        "pos1": [float(x) for x in j.GetLocalPos1Attr().Get()],
        "rot1_wxyz": quat_wxyz(j.GetLocalRot1Attr().Get()),
    }
    if rec["type"] == "PhysicsRevoluteJoint":
        rj = UsdPhysics.RevoluteJoint(prim)
        rec["axis"] = rj.GetAxisAttr().Get()
        rec["limits"] = [math.radians(rj.GetLowerLimitAttr().Get()), math.radians(rj.GetUpperLimitAttr().Get())]
    elif rec["type"] == "PhysicsPrismaticJoint":
        pj = UsdPhysics.PrismaticJoint(prim)
        rec["axis"] = pj.GetAxisAttr().Get()
        rec["limits"] = [pj.GetLowerLimitAttr().Get(), pj.GetUpperLimitAttr().Get()]  # meters (wrapper-converted)
    return rec


base = meta["arms"]["left"]["prim"]  # both arms are the same asset; read the chain once
joints = [joint_record(stage.GetPrimAtPath(f"{base}/joints/{n}")) for n in ARM_JOINTS + FIXED + FINGERS]
links = {}
for name in ["base_link", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "gripper_flange", "gripper_base",
             "gripper_link1", "gripper_link2"]:
    m = UsdPhysics.MassAPI(stage.GetPrimAtPath(f"{base}/{name}"))
    links[name] = {
        "mass": float(m.GetMassAttr().Get()),
        "com": [float(x) for x in m.GetCenterOfMassAttr().Get()],
        "inertia_diag": [float(x) for x in m.GetDiagonalInertiaAttr().Get()],
        "principal_axes_wxyz": quat_wxyz(m.GetPrincipalAxesAttr().Get()),
    }

npz, attrs = resolve_episode(args.episode)
episode = Episode.load(npz, attrs)
readback = episode.readback
ANG, LIN = 1e-6, 1e-3  # millimetre-stage readback -> SI (see tower_rl.scene)
drives = {}
for arm in ("left", "back"):
    rb = readback[arm]["readback"]
    drives[arm] = {
        "kp": [v * ANG for v in rb["kp"][0][:7]],
        "kd": [v * ANG for v in rb["kd"][0][:7]],
        "armature": [v * ANG for v in rb["armatures"][0][:7]],
        "friction_static": [v[0] * ANG for v in rb["friction"][0][:7]],
        "friction_dynamic": [v[1] * ANG for v in rb["friction"][0][:7]],
        "friction_viscous": [v[2] * ANG for v in rb["friction"][0][:7]],
        "effort_limit": [v * ANG for v in rb["max_efforts"][0][:7]],
        "velocity_limit": [float(v) for v in rb["max_velocities"][0][:7]],
        "finger_kp": float(rb["kp"][0][8]),
        "finger_kd": float(rb["kd"][0][8]),
        "finger_effort_limit": float(rb["max_efforts"][0][8] * LIN),
        "finger_velocity_limit": float(rb["max_velocities"][0][8] * LIN),
    }
delays = episode.provenance["calibration"].get("command_delays_seconds")

# finger/pinch geometry recorded by the data generator (centered_grasp_geometry motion choice)
geom = next(m for m in episode.motion_choices if m.get("kind") == "centered_grasp_geometry")["finger_geometry"]
f0 = geom["fingers"][0]
out = {
    "source": {"station": meta["station"], "wrapper": meta["wrapper"], "episode": str(npz.name),
               "note": "chain/inertials from the station USD (meters); drives from the evaluator runtime readback (SI)"},
    "arm_joints": ARM_JOINTS,
    "joints": joints,
    "links": links,
    "base_world": {arm: {"pos": meta["arms"][arm]["base_link"]["pos_m"], "quat_wxyz": meta["arms"][arm]["base_link"]["quat_wxyz"]}
                   for arm in ("left", "back")},
    # the generator's "flange" is our gripper_base (pinch FK then matches its measured pinch centres to < 0.1 mm)
    "tcp": {
        "pinch_in_gripper_base": geom["pinch_center_in_flange_m"],
        "tip_depth_in_gripper_base": geom["tip_depth_m"],
        "pad_z_range_in_gripper_base": [f0["distal_min_m"][2], f0["distal_max_m"][2]],
        "pad_half_width": 0.5 * (f0["distal_max_m"][0] - f0["distal_min_m"][0]),
        "pad_thickness": f0["distal_max_m"][1] - f0["distal_min_m"][1],
        "minimum_support_clearance": geom["minimum_support_clearance_m"],
    },
    "gripper": {"width_limits": [0.0, 0.1], "finger_limits": [-0.01, 0.06]},
    # The passive prismatic `gripper` joint's body (gripper_link) has mass 0 and no collider in the USD, so PhysX gives
    # it its default 1 kg. It slides freely along gripper_base z (velocity-capped at 3 mm/s), sits at its 0.1 m limit
    # whenever the gripper points down, and dominates the wrist/elbow gravity load (the dataset's 0.085 rad back-arm
    # elbow sag only matches with it). scripts/probe_arm_statics.py checks this against PhysX.
    "virtual_gripper_link": {"mass": 1.0, "slider_pos": 0.1},
    "drives": drives,
    "command_delays_s": delays,
}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(out, indent=1))
print(f"wrote {args.out}: {len(joints)} joints, {len(links)} links")
