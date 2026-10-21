"""Step 1: write configs/layouts/<variant>.yaml from the dataset metadata and draw top-down layout plots.

Usage: python -I scripts/build_layouts.py [--meta data/meta_tasks123.jsonl] [--plots results/layouts]
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from tower_rl.data.layouts import (BLOCK_SIZE, METADATA_JSONL, METADATA_T45_JSONL, VARIANTS, VARIANTS_T4, VARIANTS_T5,  # noqa: E402
                                   build_layout, build_layout_t45, load_metadata, write_layout, yaw_of_wxyz)
from tower_rl.paths import RESULTS_DIR  # noqa: E402


def footprint(pose, scale=1.0):
    x, y = pose[0], pose[1]
    yaw = yaw_of_wxyz(pose[3:7])
    hx, hy = BLOCK_SIZE[0] / 2 * scale, BLOCK_SIZE[1] / 2 * scale
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + c * dx - s * dy, y + s * dx + c * dy) for dx, dy in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))]


def plot(layout, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    colors = {0: "0.6", 1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    for ax, title in zip(axes, ("initial layout (nominal)", "goals (dashed) and transfers")):
        ax.set_title(f"{layout['variant']}: {title}", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(0.0, 0.62)
        ax.set_ylim(-0.66, -0.12)
        ax.grid(alpha=0.3)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        for side, pad in layout["pads"].items():
            ax.plot(*pad["center"][:2], "k+", ms=10)
            ax.annotate(side, pad["center"][:2], textcoords="offset points", xytext=(0, -95 if side == "destination" else 90),
                        ha="center", fontsize=8)
        blocks = sorted(layout["blocks"].items(), key=lambda kv: kv[1]["layer"])
        for bid, b in blocks:
            if ax is axes[1] and not b["fixed"]:
                continue
            shrink = 1.0 - 0.12 * b["layer"]  # upper layers drawn slightly smaller so lower ones stay visible
            ax.add_patch(Polygon(footprint(b["pose"], shrink), closed=True, fc=colors.get(b["layer"], "tab:red"), ec="k", alpha=0.55))
            if not b["fixed"]:
                ax.annotate(bid.replace("source_", ""), b["pose"][:2], fontsize=6, ha="center")
    for k, t in enumerate(layout["transfers"]):
        g = t["goal"]
        shrink = 1.0 - 0.12 * t["dest_layer"]
        axes[1].add_patch(Polygon(footprint(g, shrink), closed=True, fill=False, ls="--", ec=colors.get(t["dest_layer"], "tab:red"), lw=1.5))
        s = layout["blocks"][t["block"]]["pose"]
        axes[1].annotate("", xy=g[:2], xytext=s[:2], arrowprops=dict(arrowstyle="->", color="tab:purple" if t["arm"] == "back" else "tab:cyan", lw=1))
        axes[1].annotate(f"{k + 1}:{t['arm']}", ((s[0] + g[0]) / 2, (s[1] + g[1]) / 2 + 0.008 * k), fontsize=7)
    for k, p in enumerate(layout.get("pushes", [])):  # Task 4: push arrows from each block along its direction
        s = layout["blocks"][p["block"]]["pose"]
        d = p["direction"]
        axes[1].annotate("", xy=(s[0] + 0.2 * d[0], s[1] + 0.2 * d[1]), xytext=s[:2],
                         arrowprops=dict(arrowstyle="->", color="tab:purple" if p["arm"] == "back" else "tab:cyan", lw=1))
        axes[1].annotate(f"{k + 1}:{p['arm']}", (s[0] + 0.1 * d[0], s[1] + 0.1 * d[1] + 0.006 * k), fontsize=7)
    for ax in axes:  # arm bases (station frame)
        ax.plot(-0.1165, -0.288, "s", color="tab:cyan")
        ax.plot(0.381, -0.774, "s", color="tab:purple")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--plots", default=str(RESULTS_DIR / "layouts"))
    ap.add_argument("--t45", action="store_true", help="Task 4/5 variants from data/meta_tasks45.jsonl (Tasks 1-3 untouched)")
    args = ap.parse_args()
    meta = load_metadata(args.meta or (METADATA_T45_JSONL if args.t45 else METADATA_JSONL))
    Path(args.plots).mkdir(parents=True, exist_ok=True)
    for v in (VARIANTS_T4 + VARIANTS_T5 if args.t45 else VARIANTS):
        layout = build_layout_t45(v, meta[v]) if args.t45 else build_layout(v, meta[v])
        p = write_layout(layout)
        plot(layout, Path(args.plots) / f"{v}_topdown.png")
        print(f"[layouts] {v}: {layout['num_episodes']} episodes, {len(layout['transfers'])} transfers, "
              f"{len(layout.get('pushes', []))} pushes, max block std {layout['jitter']['max_block_std_mm']} mm -> {p}", flush=True)


if __name__ == "__main__":
    main()
