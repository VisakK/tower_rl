"""Static (no Kit) structural inspection of a TOWER station USD with plain pxr."""
import sys
from collections import Counter, defaultdict

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

path = sys.argv[1]
stage = Usd.Stage.Open(path, load=Usd.Stage.LoadAll)
root = stage.GetRootLayer()
print("== layer:", root.identifier)
print("defaultPrim:", stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else None)
print("metersPerUnit:", UsdGeom.GetStageMetersPerUnit(stage), "upAxis:", UsdGeom.GetStageUpAxis(stage))
print("kilogramsPerUnit:", UsdPhysics.GetStageKilogramsPerUnit(stage))
print("customLayerData:", dict(root.customLayerData))
print("sublayers:", list(root.subLayerPaths))

prims = list(stage.Traverse(Usd.TraverseInstanceProxies()))
print("prim count (incl. instance proxies):", len(prims))
print("type counts:", Counter(p.GetTypeName() for p in prims).most_common(25))
schema_counts = Counter()
for p in prims:
    for s in p.GetAppliedSchemas():
        schema_counts[s] += 1
print("applied schemas:", schema_counts.most_common(60))

print("\n== top-level hierarchy (depth<=3)")
for p in prims:
    d = len(p.GetPath().pathString.split("/")) - 1
    if d <= 3:
        print("  " * d + f"{p.GetName()} [{p.GetTypeName()}]" + (" (instanceable)" if p.IsInstanceable() else ""))

print("\n== physics scenes")
for p in prims:
    if p.IsA(UsdPhysics.Scene):
        print(p.GetPath())
        for a in p.GetAttributes():
            if a.HasAuthoredValue():
                print("   ", a.GetName(), "=", a.Get())

print("\n== articulation roots")
for p in prims:
    if p.HasAPI(UsdPhysics.ArticulationRootAPI):
        print(p.GetPath(), p.GetTypeName())
        for a in p.GetAttributes():
            if a.HasAuthoredValue() and ("physx" in a.GetName() or "physics" in a.GetName()):
                print("   ", a.GetName(), "=", a.Get())

print("\n== joints")
for p in prims:
    if p.IsA(UsdPhysics.Joint):
        j = UsdPhysics.Joint(p)
        b0 = j.GetBody0Rel().GetTargets()
        b1 = j.GetBody1Rel().GetTargets()
        info = {}
        for a in p.GetAttributes():
            n = a.GetName()
            if a.HasAuthoredValue() and any(k in n for k in ("limit", "drive", "axis", "physxJoint", "maxJointVelocity", "armature", "friction", "excludeFromArticulation", "jointEnabled", "physics:gearing", "mimic", "Mimic", "state")):
                info[n] = a.Get()
        print(f"{p.GetPath()} [{p.GetTypeName()}] body0={[str(t) for t in b0]} body1={[str(t) for t in b1]}")
        print("    schemas:", list(p.GetAppliedSchemas()))
        for k, v in sorted(info.items()):
            print("   ", k, "=", v)

print("\n== rigid bodies")
for p in prims:
    if p.HasAPI(UsdPhysics.RigidBodyAPI):
        attrs = {a.GetName(): a.Get() for a in p.GetAttributes() if a.HasAuthoredValue() and ("physics:" in a.GetName() or "physx" in a.GetName())}
        print(p.GetPath(), {k: v for k, v in attrs.items() if k in ("physics:kinematicEnabled", "physics:rigidBodyEnabled", "physics:mass", "physics:density", "physics:centerOfMass", "physics:diagonalInertia")})

print("\n== colliders (approximation counts)")
approx = Counter()
coll_examples = defaultdict(list)
for p in prims:
    if p.HasAPI(UsdPhysics.CollisionAPI):
        a = p.GetAttribute("physics:approximation")
        v = a.Get() if a and a.HasAuthoredValue() else ("<none:" + p.GetTypeName() + ">")
        en = p.GetAttribute("physics:collisionEnabled").Get() if p.GetAttribute("physics:collisionEnabled") else None
        approx[(v, en)] += 1
        if len(coll_examples[(v, en)]) < 6:
            coll_examples[(v, en)].append(str(p.GetPath()))
for k, n in approx.most_common():
    print(k, n)
    for e in coll_examples[k]:
        print("     e.g.", e)

print("\n== physics materials")
for p in prims:
    if p.HasAPI(UsdPhysics.MaterialAPI):
        attrs = {a.GetName(): a.Get() for a in p.GetAttributes() if a.HasAuthoredValue() and ("physics:" in a.GetName() or "physx" in a.GetName())}
        print(p.GetPath(), attrs)

print("\n== cameras")
for p in prims:
    if p.IsA(UsdGeom.Camera):
        c = UsdGeom.Camera(p)
        xf = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        print(p.GetPath(), "focal", c.GetFocalLengthAttr().Get(), "hAp", c.GetHorizontalApertureAttr().Get(),
              "vAp", c.GetVerticalApertureAttr().Get(), "clip", c.GetClippingRangeAttr().Get(),
              "world_t", tuple(round(x, 2) for x in xf.ExtractTranslation()))

print("\n== lights")
for p in prims:
    if "Light" in p.GetTypeName():
        print(p.GetPath(), p.GetTypeName())

print("\n== render material shader ids / mdl")
ids = Counter()
for p in prims:
    if p.GetTypeName() == "Shader":
        a = p.GetAttribute("info:id")
        m = p.GetAttribute("info:mdl:sourceAsset")
        ids[(a.Get() if a and a.HasAuthoredValue() else None, str(m.Get()) if m and m.HasAuthoredValue() else None)] += 1
for k, n in ids.most_common(20):
    print(k, n)

print("\n== composition: external asset paths referenced")
deps = set()
for layer in stage.GetUsedLayers():
    deps.add(layer.identifier)
for d in sorted(deps):
    print("  used layer:", d)
