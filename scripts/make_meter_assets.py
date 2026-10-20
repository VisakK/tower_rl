"""Build meter-unit wrapper layers around the (millimetre) TOWER station so Isaac Lab can load it.

Nothing in the downloaded assets is modified; the wrappers only reference them and add overrides.

Usage: python make_meter_assets.py <TOWER-Assets root> <out dir> <station file name> [--gripper-friction S D | --no-gripper-material]
"""
import argparse
import json
import os

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

ap = argparse.ArgumentParser()
ap.add_argument("assets_root")
ap.add_argument("out_dir")
ap.add_argument("station", choices=["station.usdz", "aliyun-rtx-training.usdz"])
ap.add_argument("--gripper-friction", nargs=2, type=float, default=None,
                help="static/dynamic friction of a finger material re-scoped inside the reference (default: none,"
                     " i.e. fingers keep the composed default material like a plain reference does)")
args = ap.parse_args()
args.out_dir = os.path.abspath(args.out_dir)  # the JSON records absolute wrapper paths

MM = 0.001
station_path = os.path.abspath(os.path.join(args.assets_root, "station", args.station))
os.makedirs(args.out_dir, exist_ok=True)
tag = "station" if args.station == "station.usdz" else "aliyun"
ARMS = {"left": "StandA_Robot", "back": "StandB_Robot"}
# prismatic joints are authored in world-space millimetres; everything else is unit-free or overridden at runtime
PRISMATIC = ("gripper", "gripper_joint1", "gripper_joint2")


def new_stage(path, default_prim):
    if os.path.exists(path):
        os.remove(path)
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{default_prim}")
    stage.SetDefaultPrim(root.GetPrim())
    return stage


# ---------------------------------------------------------------- station wrapper
src = Usd.Stage.Open(station_path)
out_station = os.path.join(args.out_dir, f"tower_{tag}_m.usda")
st = new_stage(out_station, "TowerStation")
model = UsdGeom.Xform.Define(st, "/TowerStation/Model")
model.GetPrim().GetReferences().AddReference(station_path, "/PNP_DUALD1200_STATION")
model.AddScaleOp().Set(Gf.Vec3f(MM, MM, MM))
st.OverridePrim("/TowerStation/Model/ReplayTaskBlocks").SetActive(False)

# Prims with !resetXformStack! ignore every parent transform (our mm->m scale and Isaac Lab's env offsets).
# Replace the reset with the equivalent local matrix: new_local = intended_world * parent_world^-1.
src_xc = UsdGeom.XformCache()
unreset = []
for p in src.Traverse():
    if p.IsA(UsdGeom.Xformable) and UsdGeom.Xformable(p).GetResetXformStack():
        local = src_xc.GetLocalToWorldTransform(p) * src_xc.GetLocalToWorldTransform(p.GetParent()).GetInverse()
        dst = st.OverridePrim(str(p.GetPath()).replace("/PNP_DUALD1200_STATION", "/TowerStation/Model", 1))
        x = UsdGeom.Xformable(dst)
        x.ClearXformOpOrder()
        x.AddTransformOp(opSuffix="towerUnreset").Set(local)
        unreset.append(str(p.GetPath()))

if args.gripper_friction is not None:
    mat = UsdShade.Material.Define(st, "/TowerStation/Model/GripperPhysicsMaterial")
    mapi = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    mapi.CreateStaticFrictionAttr(args.gripper_friction[0])
    mapi.CreateDynamicFrictionAttr(args.gripper_friction[1])
    mapi.CreateRestitutionAttr(0.0)

report = {"station": station_path, "wrapper": out_station, "unreset_prims": unreset, "arms": {}}
for arm, stand in ARMS.items():
    src_base = f"/PNP_DUALD1200_STATION/StationModel/InstalledRobots/{stand}/NeroRobotWithWristCamera"
    dst_base = f"/TowerStation/Model/StationModel/InstalledRobots/{stand}/NeroRobotWithWristCamera"
    for j in PRISMATIC:
        sj = UsdPhysics.PrismaticJoint(src.GetPrimAtPath(f"{src_base}/joints/{j}"))
        o = st.OverridePrim(f"{dst_base}/joints/{j}")
        oj = UsdPhysics.PrismaticJoint(o)
        oj.CreateLowerLimitAttr(sj.GetLowerLimitAttr().Get() * MM)
        oj.CreateUpperLimitAttr(sj.GetUpperLimitAttr().Get() * MM)
        sd = UsdPhysics.DriveAPI.Get(sj.GetPrim(), "linear")
        od = UsdPhysics.DriveAPI.Apply(o, "linear")
        od.CreateTargetPositionAttr(sd.GetTargetPositionAttr().Get() * MM)
        od.CreateMaxForceAttr(sd.GetMaxForceAttr().Get() * MM)  # kg*mm/s^2 -> N
        pos = sj.GetPrim().GetAttribute("state:linear:physics:position")
        if pos and pos.HasAuthoredValue():
            o.CreateAttribute("state:linear:physics:position", Sdf.ValueTypeNames.Float).Set(pos.Get() * MM)
        mjv = sj.GetPrim().GetAttribute("physxJoint:maxJointVelocity")
        if mjv and mjv.HasAuthoredValue():
            o.CreateAttribute("physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float).Set(mjv.Get() * MM)
    if args.gripper_friction is not None:
        for link in ("gripper_link1", "gripper_link2"):
            o = st.OverridePrim(f"{dst_base}/{link}")
            o.CreateRelationship("material:binding:physics").SetTargets([Sdf.Path("/TowerStation/Model/GripperPhysicsMaterial")])
    # robot root pose in the wrapper (meters) for Isaac Lab init_state (fixed base => reset writes this pose)
    rj = src.GetPrimAtPath(f"{src_base}/root_joint")
    rel = {r.GetName(): [str(t) for t in r.GetTargets()] for r in rj.GetRelationships()}
    report["arms"][arm] = {"prim": dst_base, "root_joint_rels": rel}

st.GetRootLayer().Save()

# compute world poses in the composed meter stage
cst = Usd.Stage.Open(out_station)
xc = UsdGeom.XformCache()
for arm, stand in ARMS.items():
    base = report["arms"][arm]["prim"]
    for link in ("world", "base_link"):
        m = xc.GetLocalToWorldTransform(cst.GetPrimAtPath(f"{base}/{link}"))
        t = m.ExtractTranslation()
        r = m.RemoveScaleShear().ExtractRotationQuat()
        s = Gf.Vec3d(*(m.GetRow3(i).GetLength() for i in range(3)))
        report["arms"][arm][link] = {"pos_m": [round(x, 6) for x in t],
                                     "quat_wxyz": [round(r.GetReal(), 7)] + [round(x, 7) for x in r.GetImaginary()],
                                     "world_scale": [round(x, 6) for x in s]}
    lim = {}
    for j in PRISMATIC:
        pj = UsdPhysics.PrismaticJoint(cst.GetPrimAtPath(f"{base}/joints/{j}"))
        lim[j] = [pj.GetLowerLimitAttr().Get(), pj.GetUpperLimitAttr().Get()]
    report["arms"][arm]["prismatic_limits_m"] = lim
    b = UsdShade.MaterialBindingAPI(cst.GetPrimAtPath(f"{base}/gripper_link1"))
    mat, _ = b.ComputeBoundMaterial("physics")
    report["arms"][arm]["finger_physics_material"] = str(mat.GetPath()) if mat else None
cams = {}
for name in ("low_camera", "high_camera"):
    p = cst.GetPrimAtPath(f"/TowerStation/Model/CameraAlignment/{name}")
    m = xc.GetLocalToWorldTransform(p)
    r = m.RemoveScaleShear().ExtractRotationQuat()
    c = UsdGeom.Camera(p)
    cams[name] = {"pos_m": [round(x, 6) for x in m.ExtractTranslation()],
                  "quat_wxyz_usd_camera": [round(r.GetReal(), 7)] + [round(x, 7) for x in r.GetImaginary()],
                  "focal_length": c.GetFocalLengthAttr().Get(), "h_aperture": c.GetHorizontalApertureAttr().Get(),
                  "v_aperture": c.GetVerticalApertureAttr().Get(), "clipping_range": list(c.GetClippingRangeAttr().Get())}
for arm, stand in ARMS.items():
    wpath = (f"/TowerStation/Model/StationModel/InstalledRobots/{stand}/NeroRobotWithWristCamera/gripper_base/WristCameraMount/"
             "camera_stand_link/d435_camera_link/camera_link/camera_color_frame/camera_color_optical_frame/wrist_camera")
    p = cst.GetPrimAtPath(wpath)
    local = UsdGeom.Xformable(p).GetLocalTransformation()
    parent_w = xc.GetLocalToWorldTransform(p.GetParent())
    gb = xc.GetLocalToWorldTransform(cst.GetPrimAtPath(f"/TowerStation/Model/StationModel/InstalledRobots/{stand}/NeroRobotWithWristCamera/gripper_base"))
    rel = xc.GetLocalToWorldTransform(p) * gb.GetInverse()
    c = UsdGeom.Camera(p)
    rq = rel.RemoveScaleShear().ExtractRotationQuat()
    cams[f"{arm}_wrist"] = {"path": wpath, "local_xform_ops": [op.GetOpName() for op in UsdGeom.Xformable(p).GetOrderedXformOps()],
                            "pose_in_gripper_base_m": [round(x, 6) for x in rel.ExtractTranslation()],
                            "quat_in_gripper_base_wxyz": [round(rq.GetReal(), 7)] + [round(x, 7) for x in rq.GetImaginary()],
                            "focal_length": c.GetFocalLengthAttr().Get(), "h_aperture": c.GetHorizontalApertureAttr().Get(),
                            "v_aperture": c.GetVerticalApertureAttr().Get(), "clipping_range": list(c.GetClippingRangeAttr().Get())}
report["cameras"] = cams
tbl = cst.GetPrimAtPath("/TowerStation/Model/ReplayTabletopCollision")
bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"]).ComputeWorldBound(tbl).ComputeAlignedRange()
report["tabletop_collider_aabb_m"] = [list(bb.GetMin()), list(bb.GetMax())]

# ---------------------------------------------------------------- standalone arm wrappers (light skill scene)
# Each references one installed robot prim directly. Its parent stand carries the x1000 scale, so the robot's own
# frames are already in meters; the local rotateXYZ that orients it on the stand is cleared so the wrapper's root is
# the base_link frame and Isaac Lab's init_state places it (at the station base pose from report["arms"]).
report["robot"] = {}
for arm, stand in ARMS.items():
    out_robot = os.path.join(args.out_dir, f"nero_{arm}_{tag}_m.usda")
    rs = new_stage(out_robot, "Nero")
    rp = rs.GetPrimAtPath("/Nero")
    rp.GetReferences().AddReference(station_path, f"/PNP_DUALD1200_STATION/StationModel/InstalledRobots/{stand}/NeroRobotWithWristCamera")
    rx = UsdGeom.Xformable(rp)
    rx.ClearXformOpOrder()
    rx.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
    rx.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))  # the referenced prim authors quatd/double3
    rx.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1, 1, 1))
    src_base = f"/PNP_DUALD1200_STATION/StationModel/InstalledRobots/{stand}/NeroRobotWithWristCamera"
    for j in PRISMATIC:  # same unit conversion as in the station wrapper
        sj = UsdPhysics.PrismaticJoint(src.GetPrimAtPath(f"{src_base}/joints/{j}"))
        o = rs.OverridePrim(f"/Nero/joints/{j}")
        oj = UsdPhysics.PrismaticJoint(o)
        oj.CreateLowerLimitAttr(sj.GetLowerLimitAttr().Get() * MM)
        oj.CreateUpperLimitAttr(sj.GetUpperLimitAttr().Get() * MM)
        sd = UsdPhysics.DriveAPI.Get(sj.GetPrim(), "linear")
        od = UsdPhysics.DriveAPI.Apply(o, "linear")
        od.CreateTargetPositionAttr(sd.GetTargetPositionAttr().Get() * MM)
        od.CreateMaxForceAttr(sd.GetMaxForceAttr().Get() * MM)
        pos = sj.GetPrim().GetAttribute("state:linear:physics:position")
        if pos and pos.HasAuthoredValue():
            o.CreateAttribute("state:linear:physics:position", Sdf.ValueTypeNames.Float).Set(pos.Get() * MM)
        mjv = sj.GetPrim().GetAttribute("physxJoint:maxJointVelocity")
        if mjv and mjv.HasAuthoredValue():
            o.CreateAttribute("physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float).Set(mjv.Get() * MM)
    rs.GetRootLayer().Save()
    crs = Usd.Stage.Open(out_robot)
    lim = {j: [UsdPhysics.PrismaticJoint(crs.GetPrimAtPath(f"/Nero/joints/{j}")).GetLowerLimitAttr().Get(),
               UsdPhysics.PrismaticJoint(crs.GetPrimAtPath(f"/Nero/joints/{j}")).GetUpperLimitAttr().Get()] for j in PRISMATIC}
    l1 = UsdGeom.XformCache().GetLocalToWorldTransform(crs.GetPrimAtPath("/Nero/link1")).ExtractTranslation()
    report["robot"][arm] = {"wrapper": out_robot, "prismatic_limits_m": lim, "link1_pos_m": [round(x, 6) for x in l1]}

# ---------------------------------------------------------------- block wrapper (the exact prim the generator clones)
out_block = os.path.join(args.out_dir, f"tower_block_{tag}_m.usda")
bs = new_stage(out_block, "Block")
bp = bs.GetPrimAtPath("/Block")
bp.GetReferences().AddReference(station_path, "/PNP_DUALD1200_STATION/ReplayTaskBlocks/Foundation01")
x = UsdGeom.Xformable(bp)
x.ClearXformOpOrder()
x.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
x.AddOrientOp().Set(Gf.Quatf(1, 0, 0, 0))
x.AddScaleOp().Set(Gf.Vec3f(1, 1, 1))
bp.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(False)
bs.GetRootLayer().Save()
cbs = Usd.Stage.Open(out_block)
bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(cbs.GetPrimAtPath("/Block")).ComputeAlignedRange()
wood = UsdPhysics.MaterialAPI(cbs.GetPrimAtPath("/Block/PhysicsMaterials/Wood"))
report["block"] = {"wrapper": out_block, "aabb_m": [list(bb.GetMin()), list(bb.GetMax())],
                   "mass": UsdPhysics.MassAPI(cbs.GetPrimAtPath("/Block")).GetMassAttr().Get(),
                   "friction_static_dynamic": [wood.GetStaticFrictionAttr().Get(), wood.GetDynamicFrictionAttr().Get()]}
with open(os.path.join(args.out_dir, f"tower_{tag}_m.json"), "w") as f:
    json.dump(report, f, indent=1)
print(json.dumps(report, indent=1))
