"""Full-station scene driven by a layout config (Steps 2.3/3/7): rehearsal, local evaluation and later rendering.

The station wrapper provides the visuals and the tabletop collider; both installed arms use the evaluator's drive
parameters; blocks follow ``configs/layouts/<variant>.yaml`` (foundations kinematic and fixed, as in the evaluator)
and a domain sample. Use CPU physics (``--device cpu``) for evaluator-faithful rehearsal: GPU PhysX makes the
keel-hulled blocks creep up to 15 mm (see BUILDLOG 2026-10-18). Import after ``AppLauncher``.
"""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCollectionCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from tower_rl import geometry as geo
from tower_rl.config import load_yaml
from tower_rl.data.layouts import Layout
from tower_rl.domain import DomainSample
from tower_rl.graph.interface import ARMS, State
from tower_rl.kinematics import ArmBase, ArmKinematics, load_model
from tower_rl.paths import STATION_JSON
from tower_rl.scene import load_station_meta
from tower_rl.sim.assets import JOINT_NAMES, arm_actuators, block_cfg, q_full
from tower_rl.tower import foundation_poses, goal_poses_with_error, load_spec, present_after, rest_heights, source_poses

OCCLUDERS = [
    "/World/envs/env_.*/Station/Model/StationModel/StaticCurtains",
    "/World/envs/env_.*/Station/Model/StationModel/tn__b4b08Ou03GRNY0xCb0/(StationFrameAndCabinet|UpperFrame_.*)",
]


def _robot_prim(meta: dict, arm: str) -> str:
    return "{ENV_REGEX_NS}/Station" + meta["arms"][arm]["prim"][len("/TowerStation"):]


def prim_name(block_id: str) -> str:
    """USD-safe prim name of a block id (Task 5 ids such as ``destination_l1_side_-1`` contain + and -)."""
    return block_id.replace("+", "p").replace("-", "m")


def block_order(layout: Layout) -> list[str]:
    """Foundations (source then destination) followed by the movable blocks in layout order."""
    f = list(layout.pads["source"]["foundation_blocks"]) + list(layout.pads["destination"]["foundation_blocks"])
    return f + sorted(layout.movable_ids, key=lambda b: (layout.layer[b], b))


def camera_cfgs(meta: dict, scale: float = 1.0, data_types=("rgb",), update_pose: bool = False) -> dict[str, TiledCameraCfg]:
    """The evaluator's four cameras (intrinsics from configs/scene.yaml, poses from the station calibration). Isaac
    ignores the principal point (images centred on the optical axis): see tower_rl.perception.cameras."""
    sc = load_yaml("scene")["cameras"]
    mc = meta["cameras"]
    w, h = int(round(640 * scale)), int(round(480 * scale))

    def cam(prim_path, name, pos, rot):
        k = sc[name]["K"]
        kk = [k[0][0] * scale, 0.0, k[0][2] * scale, 0.0, k[1][1] * scale, k[1][2] * scale, 0.0, 0.0, 1.0]
        return TiledCameraCfg(prim_path=prim_path, data_types=list(data_types), width=w, height=h,
                              update_latest_camera_pose=update_pose,
                              offset=TiledCameraCfg.OffsetCfg(pos=tuple(pos), rot=tuple(rot), convention="opengl"),
                              spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(intrinsic_matrix=kk, width=w, height=h,
                                                                                      clipping_range=tuple(sc[name]["clip_m"])))

    return {
        "cam_high": cam("{ENV_REGEX_NS}/CamHigh", "high", mc["high_camera"]["pos_m"], mc["high_camera"]["quat_wxyz_usd_camera"]),
        "cam_low": cam("{ENV_REGEX_NS}/CamLow", "low", mc["low_camera"]["pos_m"], mc["low_camera"]["quat_wxyz_usd_camera"]),
        "cam_left_wrist": cam(_robot_prim(meta, "left") + "/gripper_base/CamWrist", "left_wrist",
                              mc["left_wrist"]["pose_in_gripper_base_m"], mc["left_wrist"]["quat_in_gripper_base_wxyz"]),
        "cam_back_wrist": cam(_robot_prim(meta, "back") + "/gripper_base/CamWrist", "back_wrist",
                              mc["back_wrist"]["pose_in_gripper_base_m"], mc["back_wrist"]["quat_in_gripper_base_wxyz"]),
    }


def make_station_scene_cfg(variant: str, num_envs: int, cameras: bool = False, cam_scale: float = 1.0,
                           env_spacing: float = 3.0, extras: dict | None = None, cam_data_types=("rgb",),
                           cam_update_pose: bool = False) -> InteractiveSceneCfg:
    meta = load_station_meta(STATION_JSON)
    phys = load_yaml("physics")
    layout = Layout.load(variant)
    model = load_model()
    home = load_yaml("scene")["home_qpos"]
    attrs = {"station": AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Station", spawn=sim_utils.UsdFileCfg(usd_path=meta["wrapper"]))}
    # Task 4/5: the generator applied a 3 N finger effort limit (Tasks 1-3 data: 22 N, the recorded drive readback)
    finger_effort = layout.physics.get("finger_effort_n")
    for arm in ARMS:
        q = torch.tensor(home[arm], dtype=torch.float32)
        qf = q_full(q[:7], q[7:8]).tolist()
        b = meta["arms"][arm]["world"]
        attrs[arm] = ArticulationCfg(
            prim_path=_robot_prim(meta, arm), spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(pos=tuple(b["pos_m"]), rot=tuple(b["quat_wxyz"]),
                                                       joint_pos=dict(zip(JOINT_NAMES, qf)), joint_vel={".*": 0.0}),
            actuators=arm_actuators(arm, model, finger_effort=finger_effort),
        )
    collider = layout.physics.get("collider", "hull")
    attrs["blocks"] = RigidObjectCollectionCfg(rigid_objects={
        bid: block_cfg(meta, f"{{ENV_REGEX_NS}}/Block_{prim_name(bid)}", layout.blocks[bid], layout.fixed[bid], phys,
                       collider=collider, dims=layout.dims.get(bid), mass=layout.mass.get(bid))
        for bid in block_order(layout)})
    if cameras:
        attrs.update(camera_cfgs(meta, cam_scale, cam_data_types, cam_update_pose))
    attrs.update(extras or {})
    cls = configclass(type("StationSceneCfg", (InteractiveSceneCfg,), attrs))
    return cls(num_envs=num_envs, env_spacing=env_spacing, replicate_physics=True)


class StationWorld:
    """Runtime wrapper: domain resets, 10 Hz action rows with the evaluator's timing, and ``State`` readout."""

    def __init__(self, scene: InteractiveScene, sim, variant: str, finger_friction=(2.4, 2.0), delay_substeps=(1, 1)):
        self.scene, self.sim = scene, sim
        self.variant = variant
        self.phys = load_yaml("physics")
        self.dt = self.phys["dt"]
        self.sub = self.phys["control_substeps"]
        self.ticks = self.phys["policy_ticks"]
        self.dev = scene.device
        self.n = scene.num_envs
        self.layout = Layout.load(variant)
        self.spec = load_spec(variant, device=self.dev)
        self.rz = rest_heights(self.dev)
        self.blocks = scene["blocks"]
        self.block_ids = list(self.blocks.object_names)
        assert self.block_ids == block_order(self.layout)
        self.found_idx = list(range(6))
        self.mov_idx = [self.block_ids.index(b) for b in self.spec.movable_ids]
        self.kin = ArmKinematics(device=self.dev)
        self.bases = {a: ArmBase(a, device=self.dev) for a in ARMS}
        self.home = {a: torch.tensor(load_yaml("scene")["home_qpos"][a], dtype=torch.float32, device=self.dev) for a in ARMS}
        self.delays = delay_substeps
        self.time = 0.0
        self.step_count = 0
        self.finger_friction = finger_friction
        self._default_mass = None
        self._cmd = torch.cat([self.home["left"], self.home["back"]])[None].repeat(self.n, 1)
        self._hist = [self._cmd.clone() for _ in range(max(delay_substeps) + 1)]
        self._set_finger_friction(finger_friction)

    # ------------------------------------------------------------------ physics properties
    def _view_ids(self, env_ids: torch.Tensor, obj_ids: list[int]) -> torch.Tensor:
        o = torch.tensor(obj_ids)[:, None]
        return (o * self.n + env_ids.cpu()[None]).flatten()

    def _set_finger_friction(self, ff):
        for arm in ARMS:
            r = self.scene[arm]
            mp = r.root_physx_view.get_material_properties()
            mp[..., 0], mp[..., 1] = ff
            r.root_physx_view.set_material_properties(mp, torch.arange(r.num_instances, dtype=torch.int32))

    def _apply_block_dynamics(self, dom: DomainSample, env_ids: torch.Tensor):
        view = self.blocks.root_physx_view
        if self._default_mass is None:
            self._default_mass = view.get_masses().clone()
            self._default_inertia = view.get_inertias().clone()
        m = len(self.mov_idx)
        ids = self._view_ids(env_ids, self.mov_idx)  # object-major: obj0 envs..., obj1 envs...
        scale = dom.mass_scale[:, :m].T.reshape(-1).cpu()  # [m * n_env] matching ids
        masses = view.get_masses()
        inert = view.get_inertias()
        masses[ids] = self._default_mass[ids] * scale.reshape(masses[ids].shape)
        inert[ids] = self._default_inertia[ids] * scale[:, None]
        view.set_masses(masses, ids.to(torch.int32))
        view.set_inertias(inert, ids.to(torch.int32))
        mp = view.get_material_properties()
        fr = dom.friction[:, :m].permute(1, 0, 2).reshape(-1, 2).cpu()
        mp[ids, :, 0] = fr[:, 0:1]
        mp[ids, :, 1] = fr[:, 1:2]
        view.set_material_properties(mp, ids.to(torch.int32))

    def _write_blocks(self, poses: torch.Tensor):
        """Env-local poses [N, B, 7]; zero velocities for the movable blocks only (kinematic bodies reject them)."""
        world = poses.clone()
        world[..., :3] += self.scene.env_origins[:, None]
        self.blocks.write_object_link_pose_to_sim(world)
        mov = torch.tensor(self.mov_idx, device=self.dev)
        self.blocks.write_object_com_velocity_to_sim(torch.zeros(self.n, len(self.mov_idx), 6, device=self.dev), object_ids=mov)

    # ------------------------------------------------------------------ reset
    def reset(self, dom: DomainSample, stage: int = 0, settle_s: float = 1.0, place_err_m: float = 0.002,
              gen: torch.Generator | None = None):
        """All envs: foundations nominal, the source tower at ``stage`` (transfers before it already placed at their
        goals with a small error), arms at home with open grippers, then ``settle_s`` of physics."""
        n, dev = self.n, self.dev
        env_ids = torch.arange(n, device=dev)
        poses = torch.zeros(n, len(self.block_ids), 7, device=dev)
        poses[:, :6] = foundation_poses(self.spec, n)
        src = source_poses(self.spec, dom, self.rz)
        present = present_after(self.spec, torch.full((n,), stage, device=dev))
        placed = goal_poses_with_error(self.spec, self.rz, n, place_err_m, 1.0, gen)
        mov = src.clone()
        for k in range(stage):  # transfer k done: its block sits at its goal
            b = int(self.spec.transfer_block[k])
            mov[:, b] = placed[:, k]
        mov[..., 2] += 0.0005  # start 0.5 mm high and settle (no depenetration pops)
        poses[:, self.mov_idx] = mov
        self._write_blocks(poses)
        self._apply_block_dynamics(dom, env_ids)
        for arm in ARMS:
            q = self.home[arm]
            qf = q_full(q[:7], q[7:8])[None].repeat(n, 1)
            self.scene[arm].write_joint_state_to_sim(qf, torch.zeros_like(qf))
        self._cmd = torch.cat([self.home["left"], self.home["back"]])[None].repeat(n, 1)
        self._hist = [self._cmd.clone() for _ in range(max(self.delays) + 1)]
        for _ in range(int(round(settle_s / self.dt))):
            self._substep(self._cmd)
        self.time = 0.0
        self.step_count = 0
        self.present = present

    def reset_from_episode(self, ep, settle_s: float = 1.0):
        """All envs start from a recording's frame 0 (blocks 0.5 mm high, then settled; nominal block dynamics)."""
        n, dev = self.n, self.dev
        rec = torch.tensor(ep.block_poses[0], dtype=torch.float32, device=dev)
        poses = torch.zeros(n, len(self.block_ids), 7, device=dev)
        for i, bid in enumerate(self.block_ids):
            j = ep.block_ids.index(bid)
            p, q = geo.pose7_to_pq(rec[j])
            poses[:, i, :3] = p + torch.tensor([0.0, 0.0, 0.0 if self.layout.fixed[bid] else 0.0005], device=dev)
            poses[:, i, 3:7] = q
        self._write_blocks(poses)
        q0 = torch.tensor(ep.qpos_obs[0], dtype=torch.float32, device=dev)
        for i, arm in enumerate(ARMS):
            qf = q_full(q0[8 * i: 8 * i + 7], q0[8 * i + 7: 8 * i + 8])[None].repeat(n, 1)
            self.scene[arm].write_joint_state_to_sim(qf, torch.zeros_like(qf))
        self._cmd = torch.tensor(ep.qpos_cmd[0], dtype=torch.float32, device=dev)[None].repeat(n, 1)
        self._hist = [self._cmd.clone() for _ in range(max(self.delays) + 1)]
        for _ in range(int(round(settle_s / self.dt))):
            self._substep(self._cmd)
        self.time = 0.0
        self.step_count = 0

    # ------------------------------------------------------------------ stepping
    def _substep(self, q16: torch.Tensor):
        self._hist.append(q16)
        self._hist.pop(0)
        d_l, d_b = self.delays
        cmd = torch.cat([self._hist[-1 - d_l][:, :8], self._hist[-1 - d_b][:, 8:]], -1)
        for i, arm in enumerate(ARMS):
            c = cmd[:, 8 * i: 8 * i + 8]
            self.scene[arm].set_joint_position_target(q_full(c[:, :7], c[:, 7:8]))
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(self.dt)

    def apply_row(self, q16: torch.Tensor, on_tick=None, ticks: int | None = None):
        """Hold one action row for ``ticks`` 30 Hz ticks of 4 substeps (default 3 = the 10 Hz track; 1 = 30 Hz track);
        ``on_tick(t)`` after every tick."""
        self._cmd = q16.to(self.dev, torch.float32)
        for _ in range(self.ticks if ticks is None else ticks):
            for _ in range(self.sub):
                self._substep(self._cmd)
            self.time += self.sub * self.dt
            if on_tick is not None:
                on_tick(self.time)
        self.step_count += 1

    def render(self):
        self.sim.render()

    # ------------------------------------------------------------------ readout (env-local station frame)
    def block_pose(self):
        pos = self.blocks.data.object_link_pos_w - self.scene.env_origins[:, None]
        return pos, self.blocks.data.object_link_quat_w

    def block_speed(self):
        return self.blocks.data.object_link_lin_vel_w.norm(dim=-1)

    def joints(self):
        """``q [N,2,7]`` and opening widths ``[N,2]`` (finger1 - finger2)."""
        q = torch.stack([self.scene[a].data.joint_pos[:, :7] for a in ARMS], 1)
        w = torch.stack([self.scene[a].data.joint_pos[:, 8] - self.scene[a].data.joint_pos[:, 9] for a in ARMS], 1)
        return q, w

    def pinch_world(self, q: torch.Tensor):
        out = []
        for i, a in enumerate(ARMS):
            p, qq = self.kin.fk(q[:, i], "pinch")
            out.append(self.bases[a].to_world(p, qq)[0])
        return torch.stack(out, 1)

    def state(self, env: int, instruction: str = "") -> State:
        pos, quat = self.block_pose()
        q, w = self.joints()
        return State(t=self.time, step=self.step_count, variant=self.variant, instruction=instruction,
                     q=q[env].cpu().numpy(), width=w[env].cpu().numpy(), block_ids=self.block_ids,
                     block_pos=pos[env].cpu().numpy(), block_quat=quat[env].cpu().numpy(),
                     block_conf=np.ones(len(self.block_ids)),
                     privileged={"block_speed": self.block_speed()[env].cpu().numpy()})
