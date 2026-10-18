"""Smoke test: open a published TOWER task layer natively (mm stage) in Isaac Sim 5.1 and step physics.

Usage: python smoke_native_isaacsim.py <task.usda> <initial-state.json> <attrs.json with provenance readback> <out.json>
"""
import json
import sys
import time

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402
from isaacsim.core.prims import Articulation, RigidPrim  # noqa: E402
from isaacsim.core.utils.stage import get_current_stage, is_stage_loading, open_stage  # noqa: E402
from pxr import UsdGeom, UsdPhysics  # noqa: E402

usd_path, init_state_path, attrs_path, out_path = sys.argv[1:5]
report = {}

t0 = time.time()
open_stage(usd_path)
while is_stage_loading():
    simulation_app.update()
stage = get_current_stage()
mpu = UsdGeom.GetStageMetersPerUnit(stage)
report["stage_load_s"] = round(time.time() - t0, 2)
report["meters_per_unit"] = mpu
print("stage loaded in", report["stage_load_s"], "s; metersPerUnit =", mpu)

sim = SimulationContext(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 30.0, stage_units_in_meters=mpu,
                        backend="torch", device="cuda:0")
robots = Articulation("/World/Station/StationModel/InstalledRobots/Stand.*_Robot/NeroRobotWithWristCamera", name="robots")
blocks = RigidPrim("/World/Blocks/.*", name="blocks")
sim.reset()
robots.initialize()
blocks.initialize()
grav = sim.get_physics_context().get_gravity()
report["gravity"] = [list(grav[0]), grav[1]]
report["robot_prims"] = robots.prim_paths
report["num_dof"] = robots.num_dof
report["dof_names"] = robots.dof_names
report["block_prims"] = blocks.prim_paths
lim = robots.get_dof_limits()
report["dof_limits_robot0_stage_units"] = lim[0].cpu().numpy().round(4).tolist()
print("robots:", robots.count, "dof:", robots.num_dof, robots.dof_names)
print("blocks:", blocks.count)

# apply the generator's runtime drive readback (already in this mm stage's units)
prov = json.loads(json.load(open(attrs_path))["__group__sim"]["provenance_json"])
rb = prov["calibration"]["runtime_readback"]
view = robots._physics_view
order = []  # map view index -> arm name
for p in robots.prim_paths:
    order.append("left" if "StandA" in p else "back")
report["view_order"] = order


idx =torch.arange(robots.count, dtype=torch.int32)
kp = torch.tensor([rb[a]["readback"]["kp"][0] for a in order], dtype=torch.float32)
kd = torch.tensor([rb[a]["readback"]["kd"][0] for a in order], dtype=torch.float32)
arm = torch.tensor([rb[a]["readback"]["armatures"][0] for a in order], dtype=torch.float32)
mf = torch.tensor([rb[a]["readback"]["max_efforts"][0] for a in order], dtype=torch.float32)
mv = torch.tensor([rb[a]["readback"]["max_velocities"][0] for a in order], dtype=torch.float32)
fr = torch.tensor([rb[a]["readback"]["friction"][0] for a in order], dtype=torch.float32)
view.set_dof_stiffnesses(kp, idx)
view.set_dof_dampings(kd, idx)
view.set_dof_armatures(arm, idx)
view.set_dof_max_forces(mf, idx)
view.set_dof_max_velocities(mv, idx)
view.set_dof_friction_properties(fr, idx)
rb_check = {
    "kp": view.get_dof_stiffnesses()[0].numpy().tolist(),
    "friction": view.get_dof_friction_properties()[0].numpy().tolist(),
}
report["applied_readback_robot0"] = rb_check

# initial joint state from the published initial-state json (gripper width in m -> fingers +-w/2 in mm)
init = json.load(open(init_state_path))
q = np.array(init["qpos"], dtype=np.float32)
arms = {"left": q[:8], "back": q[8:]}
q_full = []
for a in order:
    j = arms[a]
    w_mm = float(j[7]) * 1000.0
    q_full.append(list(j[:7]) + [w_mm, w_mm / 2.0, -w_mm / 2.0])
q_full = torch.tensor(q_full, dtype=torch.float32, device="cuda:0")
robots.set_joint_positions(q_full)
robots.set_joint_velocities(torch.zeros_like(q_full))
robots.set_joint_position_targets(q_full)

p0, _ = blocks.get_world_poses()
p0 = p0.clone()
t1 = time.time()
n = 600  # 5 s of physics
max_speed = []
for i in range(n):
    sim.step(render=False)
    if i % 60 == 59:
        v = blocks.get_linear_velocities()
        max_speed.append(float(v.norm(dim=-1).max()))
report["sim_steps"] = n
report["wall_s_for_5s_sim"] = round(time.time() - t1, 2)
p1, _ = blocks.get_world_poses()
disp = (p1 - p0).norm(dim=-1).cpu().numpy()
report["block_displacement_mm_after_5s"] = dict(zip([p.split("/")[-1] for p in blocks.prim_paths], disp.round(3).tolist()))
report["block_max_speed_mm_s_per_0.5s"] = [round(x, 3) for x in max_speed]
qe = (robots.get_joint_positions() - q_full).abs().cpu().numpy()
report["joint_hold_abs_error_robot0"] = qe[0].round(5).tolist()
report["joint_hold_abs_error_robot1"] = qe[1].round(5).tolist()
print(json.dumps(report, indent=1))
json.dump(report, open(out_path, "w"), indent=1)
simulation_app.close()
