"""Step 2.1: dump the station structure that the scenes depend on to docs/stage_tree.txt (plain pxr, no Kit).

Reads the meter-unit wrapper (``assets_m/tower_station_m.usda``) so every pose is in meters in the station frame.
Usage: python -I scripts/dump_stage_tree.py [--out docs/stage_tree.txt]
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

from tower_rl.paths import REPO_ROOT, STATION_JSON  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "stage_tree.txt"))
args = ap.parse_args()
meta = json.loads(Path(STATION_JSON).read_text())
stage = Usd.Stage.Open(meta["wrapper"])
src = Usd.Stage.Open(meta["station"])
xc = UsdGeom.XformCache()
bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
lines = []


def w(s=""):
    lines.append(s)


def pose(p):
    m = xc.GetLocalToWorldTransform(p)
    t = m.ExtractTranslation()
    r = m.RemoveScaleShear().ExtractRotationQuat()
    q = [r.GetReal(), *r.GetImaginary()]
    yaw = math.degrees(math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
    return f"pos=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}) quat_wxyz=({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}) yaw={yaw:.1f}deg"


w(f"source station: {meta['station']}")
w(f"meter wrapper:  {meta['wrapper']}")
w(f"source metersPerUnit={UsdGeom.GetStageMetersPerUnit(src)} kilogramsPerUnit={UsdPhysics.GetStageKilogramsPerUnit(src)} "
  f"upAxis={UsdGeom.GetStageUpAxis(src)}; wrapper scale 0.001 -> meters")
w(f"unreset prims (resetXformStack replaced by explicit matrices): {meta['unreset_prims']}")
w()
w("== hierarchy (depth <= 4 below /TowerStation/Model; meshes and materials omitted)")
root = stage.GetPrimAtPath("/TowerStation/Model")
base_depth = len(root.GetPath().pathString.split("/"))
for p in Usd.PrimRange(root):
    d = len(p.GetPath().pathString.split("/")) - base_depth
    if d > 4 or p.GetTypeName() in ("Mesh", "Material", "Shader", "GeomSubset"):
        continue
    if any(s in p.GetPath().pathString for s in ("/visuals", "/collisions", "/Looks", "UrdfVisual")):
        continue
    tags = [s.replace("Physics", "").replace("API", "") for s in p.GetAppliedSchemas() if "Physics" in s or "Physx" in s][:4]
    w("  " * d + f"{p.GetName()} [{p.GetTypeName()}]" + (f" {tags}" if tags else ""))

w()
w("== articulation roots")
for p in stage.Traverse():
    if p.HasAPI(UsdPhysics.ArticulationRootAPI):
        w(f"  {p.GetPath()}")
for arm, a in meta["arms"].items():
    w(f"\n== arm '{arm}': {a['prim']}")
    w(f"  base_link world: pos={a['base_link']['pos_m']} quat_wxyz={a['base_link']['quat_wxyz']}")
    for j in stage.GetPrimAtPath(a["prim"] + "/joints").GetChildren():
        line = f"  joint {j.GetName():22s} {j.GetTypeName():24s}"
        if j.GetTypeName() == "PhysicsRevoluteJoint":
            rj = UsdPhysics.RevoluteJoint(j)
            line += f" limits(rad)=({math.radians(rj.GetLowerLimitAttr().Get()):.4f}, {math.radians(rj.GetUpperLimitAttr().Get()):.4f})"
        elif j.GetTypeName() == "PhysicsPrismaticJoint":
            pj = UsdPhysics.PrismaticJoint(j)
            line += f" limits(m)=({pj.GetLowerLimitAttr().Get():.4f}, {pj.GetUpperLimitAttr().Get():.4f})"
        w(line)
    for link in ("gripper_link1", "gripper_link2"):
        mat, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(f"{a['prim']}/{link}")).ComputeBoundMaterial("physics")
        w(f"  {link} physics material (composed): {mat.GetPath() if mat else None}")
    srcp = a["prim"].replace("/TowerStation/Model", "/PNP_DUALD1200_STATION")
    rel = src.GetPrimAtPath(f"{srcp}/gripper_link1").GetRelationship("material:binding:physics")
    w(f"  gripper_link1 physics material (source, outside defaultPrim): {[str(t) for t in rel.GetTargets()] if rel else None}")

w("\n== physics materials (source station)")
for p in src.Traverse():
    if p.HasAPI(UsdPhysics.MaterialAPI):
        m = UsdPhysics.MaterialAPI(p)
        comb = p.GetAttribute("physxMaterial:frictionCombineMode")
        w(f"  {p.GetPath()}: static={m.GetStaticFrictionAttr().Get()} dynamic={m.GetDynamicFrictionAttr().Get()} "
          f"restitution={m.GetRestitutionAttr().Get()} combine={comb.Get() if comb and comb.HasAuthoredValue() else 'default(average)'}")
mat_root = src.GetPrimAtPath("/__DualReplayGripperPhysicsMaterial")
if mat_root:
    m = UsdPhysics.MaterialAPI(mat_root)
    w(f"  /__DualReplayGripperPhysicsMaterial: static={m.GetStaticFrictionAttr().Get()} dynamic={m.GetDynamicFrictionAttr().Get()}")

w("\n== task blocks (ReplayTaskBlocks; inactive in the wrapper, read from the source in meters)")
blocks = src.GetPrimAtPath("/PNP_DUALD1200_STATION/ReplayTaskBlocks")
sxc = UsdGeom.XformCache()
sbb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
for c in blocks.GetChildren() if blocks else []:
    m = sxc.GetLocalToWorldTransform(c)
    t = m.ExtractTranslation() * 0.001
    r = m.RemoveScaleShear().ExtractRotationQuat()
    ext = sbb.ComputeWorldBound(c).ComputeAlignedRange().GetSize() * 0.001
    kin = c.GetAttribute("physics:kinematicEnabled")
    w(f"  {c.GetName():16s} pos=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}) quat_wxyz=({r.GetReal():.4f}, {', '.join(f'{x:.4f}' for x in r.GetImaginary())}) "
      f"aabb=({ext[0]:.4f}, {ext[1]:.4f}, {ext[2]:.4f}) kinematic={kin.Get() if kin else None}")
w(f"  block wrapper: {meta['block']}")

w("\n== cameras")
for p in stage.Traverse():
    if p.IsA(UsdGeom.Camera) and "Looks" not in p.GetPath().pathString:
        c = UsdGeom.Camera(p)
        w(f"  {p.GetPath()}\n     {pose(p)} focal={c.GetFocalLengthAttr().Get():.3f} h_ap={c.GetHorizontalApertureAttr().Get():.3f}")

w("\n== tabletop collider")
w(f"  ReplayTabletopCollision aabb (m): {meta['tabletop_collider_aabb_m']}")
w("\n== physics scene (source)")
for p in src.Traverse():
    if p.IsA(UsdPhysics.Scene):
        w(f"  {p.GetPath()}")
        for a in p.GetAttributes():
            if a.HasAuthoredValue():
                w(f"    {a.GetName()} = {a.Get()}")
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text("\n".join(lines) + "\n")
print(f"wrote {args.out} ({len(lines)} lines)")
