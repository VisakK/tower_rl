"""YAML configs under configs/ and result paths keyed by config hash and date (build file rule 5)."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path

import yaml

from tower_rl.paths import REPO_ROOT, RESULTS_DIR

# a frozen teacher bundle carries its own configs: TOWER_CONFIG_DIR=<bundle>/configs
CONFIG_DIR = Path(os.environ.get("TOWER_CONFIG_DIR", REPO_ROOT / "configs"))


def load_yaml(name: str | Path) -> dict:
    """``load_yaml('physics')`` reads configs/physics.yaml; paths are used as given."""
    p = Path(name)
    if not p.suffix:
        p = CONFIG_DIR / f"{name}.yaml"
    return yaml.safe_load(p.read_text())


def deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def config_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


def result_path(kind: str, cfg, tag: str = "", ext: str = ".json") -> Path:
    """``results/<kind>/<YYYY-MM-DD>_<hash>[_tag].json``."""
    day = datetime.date.today().isoformat()
    name = f"{day}_{config_hash(cfg)}" + (f"_{tag}" if tag else "") + ext
    p = RESULTS_DIR / kind / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
