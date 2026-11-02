"""Compare a frozen bundle with a reference manifest (reproduce/manifests/<name>.json = the MANIFEST.json of the bundle
as built for the challenge), file by file (sha256), in four groups:
  settings  graphs/ and student/specs/: what the runtime loads besides the models; must be identical
  configs   configs/: copies of the repository's configs/ taken when the teacher bundle was frozen. The runtime reads
            the repository's configs/ (TOWER_CONFIG_DIR is set only by rehearse_many.py --bundle_configs), so a
            difference here shows a config edited since the original freeze, not a different policy
  metadata  the JSON beside each model (source paths, training statistics)
  weights   .pt / .npz: identical only when the same checkpoints are frozen. A TorchScript re-export of the same
            checkpoint has equal weights but different bytes; retrained models differ
Exit status 1 when a settings file differs or is missing.

  python reproduce/compare_bundle.py results/bundles/student_v1_3 reproduce/manifests/student_v1_3.json
"""
import hashlib
import json
import sys
from pathlib import Path

GROUPS = ("settings", "configs", "metadata", "weights")


def group(rel):
    if rel.endswith((".pt", ".npz")):
        return "weights"
    if rel.startswith("configs/"):
        return "configs"
    if rel.startswith(("skills/", "reach/", "student/switcher")):
        return "metadata"
    return "settings"


def main():
    bundle, ref = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())["files"]
    have = {str(p.relative_to(bundle)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(bundle.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"}
    out = {g: {"identical": [], "different": [], "missing": [], "extra": []} for g in GROUPS}
    for rel, h in ref.items():
        out[group(rel)]["missing" if rel not in have else "identical" if have[rel] == h else "different"].append(rel)
    for rel in sorted(set(have) - set(ref)):
        out[group(rel)]["extra"].append(rel)
    for g, d in out.items():
        print(f"{g:9s} " + ", ".join(f"{s} {len(v)}" for s, v in d.items()))
        for s in ("different", "missing", "extra"):
            for rel in d[s]:
                print(f"    {s}: {rel}")
    bad = out["settings"]["different"] + out["settings"]["missing"]
    print("settings identical to the reference" if not bad else f"SETTINGS DIFFER: {len(bad)} files")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
