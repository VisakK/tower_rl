"""Dump transforms/scales along key paths, joint limits and gripper details of a station USD."""
import sys

from pxr import Gf, Usd, UsdGeom, UsdPhysics

stage = Usd.Stage.Open(sys.argv[1])
R = "/PNP_DUALD1200_STATION"
xc = UsdGeom.XformCache()


def show(path):
    p = stage.GetPrimAtPath(path)
    if not p:
        print("MISSING", path)
        return
    ops = [(op.GetOpName(), op.Get()) for op in UsdGeom.Xformable(p).GetOrderedXformOps()] if p.IsA(UsdGeom.Xformable) else []
    w = xc.GetLocalToWorldTransform(p)
    s = Gf.Vec3d(*(w.GetRow3(i).GetLength() for i in range(3)))
    print(f"{path}\n    type={p.GetTypeName()} instanceable={p.IsInstanceable()} schemas={list(p.GetAppliedSchemas())}")
    for n, v in ops:
        print(f"    {n} = {v}")
    print(f"    world_t={tuple(round(x, 3) for x in w.ExtractTranslation())} world_scale={tuple(round(x, 5) for x in s)}")


for path in [
    f"{R}",
    f"{R}/StationModel",
    f"{R}/StationModel/InstalledRobots",
    f"{R}/StationModel/InstalledRobots/StandA_Robot",
    f"{R}/StationModel/InstalledRobots/StandA_Robot/NeroRobotWithWristCamera",
    f"{R}/StationModel/InstalledRobots/StandA_Robot/NeroRobotWithWristCamera/world",
    f"{R}/StationModel/InstalledRobots/StandA_Robot/NeroRobotWithWristCamera/base_link",
    f"{R}/StationModel/InstalledRobots/StandA_Robot/NeroRobotWithWristCamera/link7",
    f"{R}/StationModel/InstalledRobots/StandA_Robot/NeroRobotWithWristCamera/gripper_base",
    f"{R}/StationModel/InstalledRobots/StandA_Robot/NeroRobotWithWristCamera/gripper_link1",
    f"{R}/StationModel/InstalledRobots/StandB_Robot",
    f"{R}/StationModel/InstalledRobots/StandB_Robot/NeroRobotWithWristCamera",
    f"{R}/StationModel/InstalledRobots/StandB_Robot/NeroRobotWithWristCamera/base_link",
    f"{R}/TabletopPlacement",
    f"{R}/ReplayTabletopCollision",
    f"{R}/ReplayTaskBlocks",
    f"{R}/ReplayTaskBlocks/Foundation01",
    f"{R}/ReplayTaskBlocks/RotatedBase01",
    f"{R}/ReplayTaskBlocks/TowerL2_01",
    f"{R}/CameraAlignment",
    f"{R}/CameraAlignment/low_camera",
    f"{R}/CameraAlignment/high_camera",
    f"{R}/LifterControls/left_joint8",
    f"{R}/LifterControls/back_joint8",
    f"{R}/StationModel/StationLighting/TopWhitePlaneLight",
]:
    show(path)

print("\n== Foundation01 subtree")
for p in Usd.PrimRange(stage.GetPrimAtPath(f"{R}/ReplayTaskBlocks/Foundation01")):
    attrs = {a.GetName(): a.Get() for a in p.GetAttributes() if a.HasAuthoredValue() and not a.GetName().startswith(("inputs:", "outputs:", "info:", "primvars:st", "points", "normals", "faceVertex", "extent"))}
    print(" ", p.GetPath(), p.GetTypeName(), list(p.GetAppliedSchemas()), {k: v for k, v in attrs.items() if len(str(v)) < 200})
    if p.IsA(UsdGeom.Mesh):
        m = UsdGeom.Mesh(p)
        pts = m.GetPointsAttr().Get()
        print("     mesh points:", len(pts), "faces:", len(m.GetFaceVertexCountsAttr().Get()), "extent:", m.GetExtentAttr().Get())

print("\n== joint limits / gripper (StandA)")
base = f"{R}/StationModel/InstalledRobots/StandA_Robot/NeroRobotWithWristCamera"
for p in Usd.PrimRange(stage.GetPrimAtPath(base)):
    if p.IsA(UsdPhysics.Joint):
        a = {x.GetName(): x.Get() for x in p.GetAttributes() if x.HasAuthoredValue()}
        keep = {k: v for k, v in a.items() if any(s in k for s in ("Limit", "limit", "axis", "physxJoint", "physics:jointEnabled", "excludeFromArticulation", "localPos", "localRot"))}
        print(" ", p.GetPath().name, p.GetTypeName(), keep)

print("\n== all attributes on gripper joints")
for j in ("gripper", "gripper_joint1", "gripper_joint2"):
    p = stage.GetPrimAtPath(f"{base}/joints/{j}")
    print(" ", j, list(p.GetAppliedSchemas()))
    for x in p.GetAttributes():
        if x.HasAuthoredValue():
            print("     ", x.GetName(), "=", x.Get())
    for r in p.GetRelationships():
        print("      rel", r.GetName(), r.GetTargets())

print("\n== any mimic / tendon / gear schemas anywhere?")
for p in stage.Traverse():
    sch = [s for s in p.GetAppliedSchemas() if any(k in s for k in ("Mimic", "Tendon", "Gear", "Rack"))]
    if sch or p.GetTypeName() in ("PhysxPhysicsGearJoint", "PhysxPhysicsRackAndPinionJoint"):
        print(" ", p.GetPath(), p.GetTypeName(), sch)

print("\n== LifterControls subtree")
for p in Usd.PrimRange(stage.GetPrimAtPath(f"{R}/LifterControls")):
    print(" ", p.GetPath(), p.GetTypeName(), {a.GetName(): a.Get() for a in p.GetAttributes() if a.HasAuthoredValue() and "xformOp" not in a.GetName()})

print("\n== tabletop collision cube attrs")
p = stage.GetPrimAtPath(f"{R}/ReplayTabletopCollision")
for x in p.GetAttributes():
    if x.HasAuthoredValue():
        print("   ", x.GetName(), "=", x.Get())
for r in p.GetRelationships():
    print("    rel", r.GetName(), r.GetTargets())

print("\n== material bindings to gripper physics material")
for p in stage.Traverse():
    for r in p.GetRelationships():
        if "material:binding" in r.GetName() and any("__DualReplayGripperPhysicsMaterial" in str(t) for t in r.GetTargets()):
            print(" ", p.GetPath(), r.GetName())
