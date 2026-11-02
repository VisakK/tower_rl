"""Freeze a student bundle (student plan Steps 10.1 and 13): a teacher bundle's graphs / skills / configs, the switcher
export, the task specs, the pose estimator and the student configs, with a manifest (git commit, dirty flag, file
hashes), a tar.gz and its sha256 (the server's ``artifact_identity``).

Usage: python scripts/freeze_student_bundle.py --teacher results/bundles/teacher_t45_final \
           --switcher results/student/models/<name>/switcher.pt --name student_v1 \
           [--perception results/perception/models/kp_v2/model_last.pt] [--skill <skill>_<arm>=<file.pt>]...
The runtime loads it as a teacher bundle (``--bundle results/bundles/student_v1``) with
``--student results/bundles/student_v1/student/switcher.pt``.

A bundle derived from a student bundle (``student_v1_1``, Plan-v1.1 §3.1): ``--teacher`` the student bundle,
``--switcher <it>/student/switcher.pt``, ``--perception <it>/perception/kp_v2.pt``, the v1.1 graph flags through
``--graph_set`` and the orientation references through ``--extra perception/orient_refs.npz=<file>``.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower_rl.data.layouts import ALL_VARIANTS  # noqa: E402
from tower_rl.graph.graph import load_graph  # noqa: E402
from tower_rl.paths import REPO_ROOT, RESULTS_DIR  # noqa: E402
from tower_rl.student.spec import TaskSpec  # noqa: E402

STUDENT_CONFIGS = ["student/switcher.yaml", "student/collect.yaml", "student/noise.yaml"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True, help="teacher bundle dir (graphs/, skills/, configs/)")
    ap.add_argument("--switcher", required=True, help="switcher export (.pt with its .pt.json)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--perception", default=str(RESULTS_DIR / "perception" / "models" / "kp_v2" / "model_last.pt"))
    ap.add_argument("--skill", action="append", default=[], help="<skill>_<arm>=<TorchScript .pt> replacing or adding to the "
                                                                 "teacher's (e.g. insert_bc_left=<.pt>: Step 9c)")
    ap.add_argument("--graph_set", default="{}", help='JSON merged into every graph\'s top level, e.g. {"distilled": true} '
                                                      '(Step 9c) or {"gripper_closed_width": 0.0425} (Step 9b)')
    ap.add_argument("--graph_set_variant", action="append", default=[], help="<variant>=<JSON>: merged into that variant's "
                                                                             "graph after --graph_set (per-variant flags)")
    ap.add_argument("--extra", action="append", default=[], help="<path in the bundle>=<file>: an extra file, e.g. "
                                                                 "perception/orient_refs.npz=results/perception/orient_refs.npz")
    ap.add_argument("--note", default="", help="free text for the manifest (what changed)")
    args = ap.parse_args()
    teacher = Path(args.teacher)
    out = RESULTS_DIR / "bundles" / args.name
    if out.exists():
        raise SystemExit(f"{out} exists")
    shutil.copytree(teacher / "graphs", out / "graphs")
    shutil.copytree(teacher / "skills", out / "skills")
    shutil.copytree(teacher / "configs", out / "configs")
    for spec in args.skill:
        key, path = spec.split("=", 1)
        shutil.copy2(path, out / "skills" / f"{key}.pt")
        for meta in (Path(path).with_suffix(".json"), Path(path + ".json")):  # SkillPolicy.load reads <name>.json
            if meta.exists():
                shutil.copy2(meta, out / "skills" / f"{key}.json")
                break
    gset = json.loads(args.graph_set)
    per_variant = {}
    for spec in args.graph_set_variant:
        v, js = spec.split("=", 1)
        per_variant[v] = json.loads(js)
    if gset or per_variant:
        for g in sorted((out / "graphs").glob("*.json")):
            d = json.loads(g.read_text())
            d.update(gset)
            d.update(per_variant.get(g.stem, {}))
            g.write_text(json.dumps(d, indent=1))
    (out / "student" / "specs").mkdir(parents=True)
    shutil.copy2(args.switcher, out / "student" / "switcher.pt")
    shutil.copy2(args.switcher + ".json", out / "student" / "switcher.pt.json")
    for v in ALL_VARIANTS:  # the task specs as the runtime builds them from this bundle's graphs
        spec = TaskSpec.from_graph(load_graph(v, out / "graphs" / f"{v}.json"))
        (out / "student" / "specs" / f"{v}.json").write_text(json.dumps(
            {"variant": v, "family": spec.family, "goals": [{"k": g.k, "type": g.type, "block": g.block, "arm": g.arm,
                                                             "stages": list(g.stages), **g.geometry} for g in spec.goals],
             "parks_before": {str(k): p["arm"] for k, p in spec.parks_before.items()},
             "end_park": spec.end_park["arm"] if spec.end_park else None}, indent=1))
    (out / "perception").mkdir()
    shutil.copy2(args.perception, out / "perception" / "kp_v2.pt")
    for spec in args.extra:
        dst, src = spec.split("=", 1)
        (out / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out / dst)
    for c in STUDENT_CONFIGS:
        src = REPO_ROOT / "configs" / c
        if src.exists():
            dst = out / "configs" / c
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip())
    files = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}
    sw = json.loads((out / "student" / "switcher.pt.json").read_text())
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_commit": commit, "git_dirty": dirty,
                "teacher": str(teacher.resolve()), "switcher": {"source": str(Path(args.switcher).resolve()),
                                                                 "features": sw.get("features"), "sha256": sw.get("sha256")},
                "perception": str(Path(args.perception).resolve()), "replaced_skills": args.skill, "graph_set": gset,
                "graph_set_variant": per_variant, "extra": args.extra, "note": args.note, "files": files}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    tar = out.with_suffix(".tar.gz")
    with tarfile.open(tar, "w:gz") as tf:
        tf.add(out, arcname=out.name)
    digest = sha256(tar)
    tar.with_suffix(".sha256").write_text(f"{digest}  {tar.name}\n")
    print(json.dumps({"bundle": str(out), "tar": str(tar), "sha256": digest}, indent=1))


if __name__ == "__main__":
    main()
