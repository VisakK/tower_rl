"""Block-release test: drop a block from a few mm above its goal on the destination foundation and measure where it
ends up. Rehearsals showed rare "pops" on release (impact at ~0.3 m/s, 1-2 mm penetration, the block ejected at
~0.6 m/s and landing 10-25 mm away); this measures their rate against the release height and the block's residual
motion from the grasp, on CPU PhysX (the evaluator) and on GPU PhysX (skill training).

Each env: the variant's three destination foundation blocks (kinematic, as in the evaluator) and the transfer's block
released flat-ish at ``rest_z + h`` with a random tilt, linear and angular velocity.
Usage: python scripts/drop_test.py --headless --device cpu --n 64 --heights 0 0.0015 0.003 0.005 0.008
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--variant", default="task2-2_cross_tower_transfer_2")
parser.add_argument("--transfer", type=int, default=0)
parser.add_argument("--n", type=int, default=64, help="drops per condition (= envs)")
parser.add_argument("--heights", nargs="+", type=float, default=[0.0, 0.0015, 0.003, 0.005, 0.008],
                    help="gap between the block's flat bottom and its support (m)")
parser.add_argument("--tilt_deg", type=float, default=2.0, help="uniform +- tilt about the block's width axis")
parser.add_argument("--lin_vel", type=float, default=0.03, help="residual linear velocity scale (m/s, per axis sigma)")
parser.add_argument("--ang_vel", type=float, default=0.3, help="residual angular velocity scale (rad/s, per axis sigma)")
parser.add_argument("--settle_s", type=float, default=1.5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObjectCollectionCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from tower_rl import geometry as geo  # noqa: E402
from tower_rl.config import load_yaml  # noqa: E402
from tower_rl.paths import RESULTS_DIR, STATION_JSON  # noqa: E402
from tower_rl.scene import load_station_meta  # noqa: E402
from tower_rl.sim.assets import block_cfg, sim_cfg  # noqa: E402
from tower_rl.sim.skill_scene import TABLE_SIZE, _static_box  # noqa: E402
from tower_rl.tower import load_spec, rest_heights  # noqa: E402

phys = load_yaml("physics")
meta = load_station_meta(STATION_JSON)
spec = load_spec(args.variant)
rz = rest_heights("cpu")
k = args.transfer
goal = spec.goal[k].clone()
goal[2] = rz[int(spec.transfer_dest_layer[k])]
blocks = {f"found{i}": block_cfg(meta, f"{{ENV_REGEX_NS}}/Found{i}", spec.foundation[3 + i].tolist(), True, phys) for i in range(3)}
blocks["drop"] = block_cfg(meta, "{ENV_REGEX_NS}/Drop", (goal + torch.tensor([0, 0, 0.05, 0, 0, 0, 0])).tolist(), False, phys)
attrs = {"table": _static_box("{ENV_REGEX_NS}/Table", TABLE_SIZE, (0.48, -0.33, -0.0125), (0.55, 0.55, 0.6)),
         "blocks": RigidObjectCollectionCfg(rigid_objects=blocks)}
SceneCfg = configclass(type("DropSceneCfg", (InteractiveSceneCfg,), attrs))
sim = sim_utils.SimulationContext(sim_cfg(phys, device=args.device))
scene = InteractiveScene(SceneCfg(num_envs=args.n, env_spacing=1.5, replicate_physics=True))
sim.reset()
dev = scene.device
n = args.n
blk = scene["blocks"]
di = blk.object_names.index("drop")
gen = torch.Generator(device=dev).manual_seed(args.seed)
dt = phys["dt"]
gpos = goal[:3].to(dev)
gq = goal[3:7].to(dev)
y_axis = geo.quat_apply(gq, torch.tensor([0.0, 1.0, 0.0], device=dev))
x_axis = geo.quat_apply(gq, torch.tensor([1.0, 0.0, 0.0], device=dev))
results = {}
for h in args.heights:
    th = (torch.rand(n, generator=gen, device=dev) * 2 - 1) * math.radians(args.tilt_deg)
    tx = (torch.rand(n, generator=gen, device=dev) * 2 - 1) * math.radians(args.tilt_deg) * 0.5
    q = geo.quat_mul(geo.quat_from_axis_angle(y_axis.expand(n, 3), th), gq.expand(n, 4))
    q = geo.quat_mul(geo.quat_from_axis_angle(x_axis.expand(n, 3), tx), q)
    # lift so the lowest corner of the tilted block clears the support by h (half length 0.0745, half width 0.025)
    lift = 0.0745 * th.abs().sin() + 0.025 * tx.abs().sin()
    p = gpos.expand(n, 3).clone()
    p[:, 2] += h + lift
    pose = torch.cat([p + scene.env_origins, q], -1)
    vel = torch.cat([torch.randn(n, 3, generator=gen, device=dev) * args.lin_vel,
                     torch.randn(n, 3, generator=gen, device=dev) * args.ang_vel], -1)
    blk.write_object_link_pose_to_sim(pose[:, None], object_ids=[di])
    blk.write_object_com_velocity_to_sim(vel[:, None], object_ids=[di])
    scene.write_data_to_sim()
    vz_max = torch.zeros(n, device=dev)
    zmax = torch.zeros(n, device=dev)
    for _ in range(int(round(args.settle_s / dt))):
        sim.step(render=False)
        scene.update(dt)
        v = blk.data.object_link_lin_vel_w[:, di]
        vz_max = torch.maximum(vz_max, v[:, 2])
        zmax = torch.maximum(zmax, blk.data.object_link_pos_w[:, di, 2] - gpos[2])
    fp = blk.data.object_link_pos_w[:, di] - scene.env_origins
    fq = blk.data.object_link_quat_w[:, di]
    dxy = (fp[:, :2] - gpos[:2]).norm(dim=-1)
    dyaw = geo.yaw_diff_sym(geo.yaw_from_quat(fq), geo.yaw_from_quat(gq.expand(n, 4)))
    dz = fp[:, 2] - gpos[2]
    ok = (dxy <= 0.005) & (dyaw <= math.radians(5)) & (dz.abs() <= 0.010)
    still = blk.data.object_link_lin_vel_w[:, di].norm(dim=-1) < 0.01
    pct = lambda x: [round(float(v), 2) for v in torch.quantile(x.float().cpu(), torch.tensor([0.5, 0.9, 0.99, 1.0]))]  # noqa: E731
    results[f"{h * 1000:.1f}mm"] = {"in_tol": round(float(ok.float().mean()), 4), "static": round(float(still.float().mean()), 4),
                                    "dxy_mm_p50_p90_p99_max": pct(dxy * 1000), "rebound_vz_p50_p90_p99_max": pct(vz_max),
                                    "rise_above_goal_mm_p50_p90_p99_max": pct(zmax * 1000),
                                    "pops_gt_5mm": int((dxy > 0.005).sum()), "n": n}
    print(f"[drop] h={h * 1000:.1f} mm: in tol {results[f'{h * 1000:.1f}mm']['in_tol']:.3f}, dxy p50/p90/p99/max "
          f"{results[f'{h * 1000:.1f}mm']['dxy_mm_p50_p90_p99_max']} mm, rebound vz {results[f'{h * 1000:.1f}mm']['rebound_vz_p50_p90_p99_max']}",
          flush=True)
cfg = {k2: getattr(args, k2) for k2 in ("variant", "transfer", "n", "tilt_deg", "lin_vel", "ang_vel", "settle_s", "seed", "device")}
out = Path(args.out or RESULTS_DIR / "drop_test" / f"{args.variant}_t{k}_{args.device.split(':')[0]}.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"config": cfg, "results": results}, indent=1))
print("REPORT_JSON " + json.dumps(results), flush=True)
os._exit(0)
