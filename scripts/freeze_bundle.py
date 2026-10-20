"""Step 7.7: freeze the teacher into bundle/ (graphs + TorchScript skills + configs) and record its sha256.

Usage: python scripts/freeze_bundle.py --skill grasp_lift:left=<run dir> --skill grasp_lift:back=<run dir> \
           [--skill place_release:left=<run dir> ...] [--out bundle]
Skills without a checkpoint fall back to the executor's planned controllers (recorded in the manifest).
The executor loads graphs/ and skills/ from the bundle; set TOWER_CONFIG_DIR=<bundle>/configs to read configs from it.
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

from tower_rl.paths import REPO_ROOT  # noqa: E402
from tower_rl.skills.export import export  # noqa: E402

CONFIGS = ["physics.yaml", "scene.yaml", "tiers.yaml", "robot/nero.json", "skills/grasp_geometry.yaml",
           "skills/posture_priors.yaml", "skills/grasp_lift.yaml", "skills/place_release.yaml", "layouts"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skill", action="append", default=[], help="<skill>:<arm>=<run dir or checkpoint>")
    ap.add_argument("--graphs", default=str(REPO_ROOT / "tower_rl" / "graph" / "graphs"))
    ap.add_argument("--out", default=str(REPO_ROOT / "bundle"))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "skills").mkdir(parents=True)
    shutil.copytree(args.graphs, out / "graphs")
    for c in CONFIGS:
        src = REPO_ROOT / "configs" / c
        dst = out / "configs" / c
        dst.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)
    skills = {}
    for spec in args.skill:
        key, path = spec.split("=", 1)
        skill, arm = key.split(":")
        dst = out / "skills" / f"{skill}_{arm}.pt"
        export(path, dst)
        skills[f"{skill}_{arm}"] = {"source": str(Path(path).resolve()), "sha256": sha256(dst)}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip())
    files = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_commit": commit, "git_dirty": dirty, "skills": skills,
                "planned_fallbacks": sorted({f"{s}_{a}" for s in ("grasp_lift", "place_release") for a in ("left", "back")} - set(skills)),
                "files": files}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    tar = out.with_suffix(".tar.gz")
    with tarfile.open(tar, "w:gz") as tf:
        tf.add(out, arcname=out.name)
    digest = sha256(tar)
    tar.with_suffix(".sha256").write_text(f"{digest}  {tar.name}\n")
    print(json.dumps({"bundle": str(out), "tar": str(tar), "sha256": digest, "skills": list(skills),
                      "planned_fallbacks": manifest["planned_fallbacks"]}, indent=1))


if __name__ == "__main__":
    main()
