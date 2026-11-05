"""The block diagrams of the system write-ups (``Teacher-v1.MD``, ``Student-v1.MD`` in ~/Documents/corl_challenge), drawn
with matplotlib as SVG (no Graphviz on this machine). Coordinates are inches; every figure is one function below.

  python scripts/doc_figures.py --out ~/Documents/corl_challenge/figures [--only teacher_overview ...] [--png <dir>]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams.update({"svg.fonttype": "path", "font.family": "DejaVu Sans", "font.size": 9})

# fill, edge per kind of box
KIND = {
    "sim": ("#e8eef7", "#4a6fa5"),        # simulator / shared infrastructure
    "teacher": ("#fdebd3", "#c07a1e"),    # teacher-only logic
    "student": ("#dcefe6", "#2f8a5f"),    # student-only logic
    "learned": ("#f3dcef", "#9b3d8c"),    # trained networks
    "data": ("#ffffff", "#777777"),       # datasets, checkpoints, logs
    "external": ("#ece6f6", "#6a4fa3"),   # the official evaluator and the network
    "note": ("#fffbe6", "#b59a1a"),       # remarks, known gaps
    "ok": ("#e3f4e1", "#3d8b37"),
    "plain": ("#f6f6f6", "#888888"),
}
NAMES = {"sim": "simulator / shared", "teacher": "teacher logic", "student": "student logic", "learned": "trained network",
         "data": "data / checkpoint", "external": "official evaluator", "note": "known gap / remark", "plain": "park / other"}


class Diagram:
    def __init__(self, w: float, h: float, title: str | None = None):
        self.fig, self.ax = plt.subplots(figsize=(w, h))
        self.fig.patch.set_facecolor("white")
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(0, h)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.boxes: dict[str, tuple[float, float, float, float]] = {}
        if title:
            self.ax.text(0.15, h - 0.12, title, fontsize=12, fontweight="bold", va="top", ha="left")

    def group(self, x, y, w, h, label, color="#999999", fill="#fafafa"):
        """A dashed background frame with its label at the top left."""
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fill, ec=color,
                                         lw=1.0, ls=(0, (4, 3)), zorder=0))
        self.ax.text(x + 0.12, y + h - 0.08, label, fontsize=8.5, color=color, fontweight="bold", va="top", ha="left", zorder=1)

    def box(self, key, x, y, w, h, title, body=None, kind="plain", title_size=9.5, body_size=7.6, dashed=False, align="center"):
        fc, ec = KIND[kind]
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc=fc, ec=ec,
                                         lw=1.6 if kind == "learned" else 1.2, ls=(0, (3, 2)) if dashed else "-", zorder=2))
        self.boxes[key] = (x, y, w, h)
        if body:
            self.ax.text(x + w / 2, y + h - 0.09, title, fontsize=title_size, fontweight="bold", ha="center", va="top", zorder=3,
                         linespacing=1.15)
            title_h = (title.count("\n") + 1) * title_size / 72 * 1.42
            tx = x + w / 2 if align == "center" else x + 0.1
            self.ax.text(tx, y + h - 0.13 - title_h, body, fontsize=body_size, ha=align, va="top",
                         linespacing=1.25, zorder=3, color="#222222")
        else:
            self.ax.text(x + w / 2, y + h / 2, title, fontsize=title_size, fontweight="bold", ha="center", va="center", zorder=3,
                         linespacing=1.15)

    def anchor(self, key, side, t=0.5):
        x, y, w, h = self.boxes[key]
        return {"left": (x, y + t * h), "right": (x + w, y + t * h), "top": (x + t * w, y + h), "bottom": (x + t * w, y)}[side]

    def arrow(self, a, b, label=None, sa="right", sb="left", ta=0.5, tb=0.5, rad=0.0, color="#333333", dashed=False,
              label_pos=0.5, label_dx=0.0, label_dy=0.04, label_size=7.3, both=False, lw=1.2):
        p = self.anchor(a, sa, ta) if isinstance(a, str) else a
        q = self.anchor(b, sb, tb) if isinstance(b, str) else b
        self.ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=10, lw=lw, color=color,
                                          connectionstyle=f"arc3,rad={rad}", ls=(0, (4, 2)) if dashed else "-", zorder=4,
                                          shrinkA=1, shrinkB=1))
        if label:
            mx, my = p[0] + (q[0] - p[0]) * label_pos + label_dx, p[1] + (q[1] - p[1]) * label_pos + label_dy
            self.ax.text(mx, my, label, fontsize=label_size, ha="center", va="bottom", color=color, zorder=5, linespacing=1.15,
                         bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))

    def path(self, points, label=None, color="#333333", dashed=False, label_at=None, label_size=7.3):
        """An orthogonal arrow through the given points (the last segment carries the head)."""
        for p, q in zip(points[:-2], points[1:-1]):
            self.ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=1.2, ls=(0, (4, 2)) if dashed else "-", zorder=4)
        self.ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=10, lw=1.2, color=color,
                                          ls=(0, (4, 2)) if dashed else "-", zorder=4, shrinkA=0, shrinkB=1))
        if label:
            lx, ly = label_at if label_at else points[len(points) // 2]
            self.ax.text(lx, ly, label, fontsize=label_size, ha="center", va="bottom", color=color, zorder=5, linespacing=1.15,
                         bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))

    def text(self, x, y, s, size=7.6, color="#333333", ha="left", va="top", weight="normal", style="normal"):
        self.ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, fontstyle=style, zorder=5,
                     linespacing=1.25)

    def legend(self, x, y, kinds, step=1.75):
        for i, k in enumerate(kinds):
            fc, ec = KIND[k]
            self.ax.add_patch(FancyBboxPatch((x + i * step, y), 0.22, 0.16, boxstyle="round,pad=0,rounding_size=0.03", fc=fc,
                                             ec=ec, lw=1.2, zorder=2))
            self.ax.text(x + i * step + 0.3, y + 0.08, NAMES[k], fontsize=7.3, va="center", ha="left", color="#333333")

    def save(self, path: Path, png: Path | None = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.08, facecolor="white")
        if png is not None:
            png.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(png / (path.stem + ".png"), dpi=90, bbox_inches="tight", pad_inches=0.08, facecolor="white")
        plt.close(self.fig)
        print(f"wrote {path}")


FIGURES = {}


def figure(name):
    def deco(fn):
        FIGURES[name] = fn
        return fn
    return deco


# ---------------------------------------------------------------------------------------------------------------- teacher


@figure("teacher_overview")
def teacher_overview():
    d = Diagram(14.5, 8.0, "Teacher v1: privileged state -> task-graph executor -> skill nodes -> 10 Hz joint rows")
    d.group(0.15, 2.1, 4.45, 5.3, "Isaac Lab station scene (CPU PhysX, evaluator drives)", color="#4a6fa5", fill="#f6f8fc")
    d.box("dom", 0.35, 5.95, 4.05, 1.05, "DomainSampler (tiers D0 / D1)",
          "seeded per episode; D1: one of 10 cases (mass, friction,\nshifted / irregular tower, lighting, combined)\nnothing from the sample reaches the policy", kind="sim", body_size=7.2)
    d.box("world", 0.35, 4.0, 4.05, 1.6, "StationWorld + PhysX",
          "2 Nero arms (the evaluator's PD gains), 6 foundations,\n2-9 movable blocks, tabletop (no floor below)\ndt 1/120 s; a row is held 0.1 s = 12 substeps;\n1-substep command delay", kind="sim", body_size=7.2)
    d.box("trk", 0.35, 2.3, 4.05, 1.35, "Operation trackers (30 Hz)",
          "our reconstruction of grasp / place / task success\n5 mm / 5 deg, settle 1 s; structure 10 mm (Tasks 4, 5)\nsuccess -> early stop; 210 s -> failure", kind="sim", body_size=7.2)
    d.arrow("dom", "world", "reset: poses, mass, friction", sa="bottom", sb="top", label_dx=0.0, label_dy=-0.02)
    d.arrow("world", "trk", "truth every tick", sa="bottom", sb="top", label_dy=-0.02)
    d.box("src", 4.95, 5.95, 2.3, 1.05, "Pose source", "sim truth (teacher)\nkp_v2 cameras | noise", kind="teacher", body_size=7.2)
    d.box("state", 4.95, 4.0, 2.3, 1.6, "State (per call)",
          "t, step, q [2,7] rad,\nwidths [2] m, block\npos / quat / conf [B]\n(privileged: unread)", kind="teacher", body_size=7.2)
    d.arrow("world", "state", "queue empty", sa="right", sb="left", label_dy=0.02)
    d.arrow("src", "state", sa="bottom", sb="top")
    d.group(7.55, 2.1, 3.4, 5.3, "Executor (EnvRun, one per env)", color="#c07a1e", fill="#fffaf3")
    d.box("graph", 7.75, 5.95, 3.0, 1.05, "Task graph JSON",
          "linear node list per variant\n(layout -> graph builder; retries 2)", kind="teacher", body_size=7.2)
    d.box("node", 7.75, 4.65, 3.0, 0.95, "Node pointer -> make_node",
          "arm-switch queue | clearance park\n| next graph node", kind="teacher", body_size=7.2)
    d.box("est", 7.75, 3.45, 3.0, 0.9, "Rest-pose estimates",
          "medians of 10 samples (conf >= 0.5)\nheld block = pinch x offset", kind="teacher", body_size=7.2)
    d.box("adv", 7.75, 2.3, 3.0, 0.85, "_advance (verdicts)",
          "ok -> next; failed / timeout -> retry,\nskip, arm switch", kind="teacher", body_size=7.2)
    d.arrow("graph", "node", sa="bottom", sb="top")
    d.arrow("state", "est", sa="right", sb="left", ta=0.2, tb=0.6)
    d.arrow("est", "node", "targets", sa="top", sb="bottom", ta=0.7, tb=0.7, label_dx=0.32, label_dy=-0.12)
    d.path([(7.75, 2.72), (7.65, 2.72), (7.65, 5.12), (7.75, 5.12)], color="#c07a1e", dashed=True)
    d.text(7.0, 3.0, "pointer", size=6.8, color="#c07a1e")
    d.group(11.25, 2.1, 3.1, 5.3, "Skill nodes", color="#9b3d8c", fill="#fdf8fc")
    d.box("planned", 11.45, 5.45, 2.7, 1.55, "Planned",
          "transit: IK via-points, <= 16 rows\npark: joint interpolation\nTask 5 grasp, insert, push-in:\nclosed-loop controllers", kind="teacher", body_size=7.0)
    d.box("ppo", 11.45, 3.75, 2.7, 1.4, "PPO (rsl_rl), per arm",
          "grasp_lift 65-D (Tasks 1-3)\nplace_release 73-D\npush 65-D (Task 4)\ndeterministic mean, 1 row / call", kind="learned", body_size=7.0)
    d.box("ctl", 11.45, 2.3, 2.7, 1.1, "Controllers",
          "DLS IK + gravity-sag\ncompensation; gripper rate limit", kind="sim", body_size=7.0)
    d.arrow("node", "ppo", "node.step(State)", sa="right", sb="left", ta=0.5, tb=0.75, label_dy=0.03)
    d.arrow("ppo", "ctl", sa="bottom", sb="top")
    d.path([(14.15, 6.2), (14.28, 6.2), (14.28, 2.85), (14.15, 2.85)], color="#333333")
    d.arrow("ppo", "adv", "status + info", sa="left", sb="right", ta=0.12, tb=0.6, label_dx=0.0, label_dy=-0.3)
    d.path([(12.8, 2.3), (12.8, 1.7), (0.25, 1.7), (0.25, 4.8), (0.35, 4.8)], "chunk [H <= 16, 16]: per arm 7 joint targets (rad) + gripper width (m)",
           color="#4a6fa5", label_at=(6.6, 1.72))
    d.box("rep", 4.95, 0.2, 2.6, 1.05, "Rehearsal reports", "feedback.md, episodes.json,\nmetrics.json (rehearse.py)", kind="data", body_size=7.0)
    d.box("edit", 7.95, 0.2, 2.9, 1.05, "Graph edit loop (offline)", "LLM edits of graph JSON and\nskill YAMLs; kept if better", kind="teacher", body_size=7.0, dashed=True)
    d.arrow("trk", "rep", sa="right", sb="top", ta=0.15, tb=0.3, rad=-0.25)
    d.arrow("rep", "edit")
    d.path([(10.85, 0.72), (11.05, 0.72), (11.05, 6.45), (10.75, 6.45)], color="#c07a1e", dashed=True)
    d.legend(11.35, 0.85, ["sim", "teacher"], step=1.55)
    d.legend(11.35, 0.45, ["learned", "data"], step=1.55)
    return d


@figure("teacher_call_loop")
def teacher_call_loop():
    d = Diagram(12.7, 5.9, "Executor: one 10 Hz call (EnvRun.act)")
    d.box("rb", 0.3, 4.05, 2.3, 1.15, "run_batch", "an env's row queue is empty:\nState = world.state(e)\n(+ observe / noise)", kind="sim", body_size=7.2)
    d.box("first", 2.95, 4.05, 2.3, 1.15, "First call only", "reset the arm and gripper\ncontrollers to measured q", kind="teacher", body_size=7.2)
    d.box("obs", 5.6, 4.05, 2.35, 1.15, "observe(State)", "every block pose feeds\nits rest-pose estimate", kind="teacher", body_size=7.2)
    d.box("pick", 8.3, 4.05, 4.2, 1.15, "Pick the node", "1. head of the arm-switch queue\n2. else a clearance park (other arm within 0.18 m)\n3. else make_node(graph[i])", kind="teacher", body_size=7.2)
    d.arrow("rb", "first")
    d.arrow("first", "obs")
    d.arrow("obs", "pick")
    d.box("step", 8.3, 2.05, 4.2, 1.35, "node.step(State)", "guards first: timeout, dropped, structure, shoved\nobservation -> policy or planner -> targets\nIK + sag compensation -> rows", kind="learned", body_size=7.2)
    d.box("status", 5.6, 2.05, 2.35, 1.35, "Terminal status?", "ok / failed / timeout\n-> _advance (pointer)\nnone: keep the node", kind="teacher", body_size=7.2)
    d.box("rows", 2.95, 2.05, 2.3, 1.35, "Rows yet?", "no: pick the next node\n(at most 64 steps)\nyes: return them", kind="teacher", body_size=7.2)
    d.box("chunk", 0.3, 2.05, 2.3, 1.35, "Chunk [H, 16]", "H <= 16: transit, park\nH = 1: contact nodes\n30 Hz: x3 interpolated", kind="sim", body_size=7.2)
    d.arrow("pick", "step", sa="bottom", sb="top", ta=0.3, tb=0.3)
    d.arrow("step", "status", sa="left", sb="right")
    d.arrow("status", "rows", sa="left", sb="right")
    d.arrow("rows", "chunk", "yes", sa="left", sb="right")
    d.path([(4.1, 3.4), (4.1, 3.72), (11.6, 3.72), (11.6, 4.05)], "no rows", color="#c07a1e", label_at=(7.0, 3.73))
    d.box("apply", 0.3, 0.2, 6.0, 1.35, "Runner -> PhysX", "validate 1 <= H <= 16 and finite; queue the rows\neach row held 0.1 s (3 ticks x 4 substeps of 1/120 s)\ntrackers update every tick; next call when the queue empties", kind="sim", body_size=7.2)
    d.arrow("chunk", "apply", sa="bottom", sb="top", ta=0.5, tb=0.2)
    d.path([(0.3, 0.87), (0.15, 0.87), (0.15, 4.62), (0.3, 4.62)], color="#4a6fa5")
    d.text(6.7, 1.4, "Measured: 26-30 ms per call mean, <= 0.34 s max on CPU (BUILDLOG Step 8.1 round 2).\nThe executor never ends an episode: after park_<arm>_end it holds still.", size=7.4, color="#555555")
    return d


@figure("teacher_node_lifecycle")
def teacher_node_lifecycle():
    d = Diagram(12.8, 6.7, "A node's lifecycle and the failure edges (_advance)")
    d.box("created", 0.25, 4.6, 2.6, 1.4, "Created", "make_node(spec); targets\ncomputed once from rest\nestimates (pre-grasp, goal,\nslot, tower axis)", kind="teacher", body_size=7.2)
    d.box("running", 3.2, 4.6, 3.0, 1.4, "Running", "one or more rows per call\nper-row aborts: dropped, empty\ngripper, structure > 10 mm,\nshoved > 40 mm, unreachable", kind="teacher", body_size=7.2)
    d.box("ok", 6.6, 5.3, 1.8, 0.7, "OK", kind="ok")
    d.box("fail", 6.6, 4.35, 1.8, 0.8, "FAILED or\nTIMEOUT", kind="note", title_size=8.8)
    d.box("next", 8.8, 5.3, 3.75, 0.7, "next graph node (a park always advances)", kind="ok", title_size=8.0)
    d.arrow("created", "running")
    d.arrow("running", "ok", sa="right", sb="left", ta=0.75)
    d.arrow("running", "fail", sa="right", sb="left", ta=0.3)
    d.arrow("ok", "next")
    chain = [("d1", "on_fail = skip (Task 5 insert, push_in),\nor the block fell off the table -> skip"),
             ("d2", "a place missed by <= 15 mm, own arm cannot\nreach, the other can (once) -> arm-switch queue"),
             ("d3", "tries > 2 (3 attempts per transfer) -> skip:\nopen, detach, go to the next transfer"),
             ("d4", "the arm still holds the block -> retry from\nthe pre-place / pre-insert transit"),
             ("d5", "else: open and restart the transfer at its\npre-grasp transit (a re-grasp)")]
    for i, (k, t) in enumerate(chain):
        d.box(k, 8.8, 4.25 - i * 0.98, 3.75, 0.8, t, kind="teacher" if i < 4 else "teacher", title_size=7.4)
        if i:
            d.arrow(chain[i - 1][0], k, "no", sa="bottom", sb="top", label_dx=0.18, label_dy=-0.1, label_size=6.8)
    d.arrow("fail", "d1", sa="right", sb="left")
    d.box("budget", 0.25, 2.3, 5.95, 1.65, "Retry budget and what it buys",
          "3 attempts per transfer; failures of all its nodes count; FAILED = TIMEOUT\narm switch at most once per transfer; afterwards the next transfer\nmeasured: the arm-switch re-grasp timed out 206 of 217 times;\nre-grasp after a timeout 17 / 131, after a drop 20 / 38\nTask 4 pushes retry; Task 5 insert and push-in never retry", kind="data", body_size=7.2)
    d.box("end", 0.25, 0.3, 5.95, 1.65, "After the last node",
          "the executor holds still; the runner ends the episode on tracker\nsuccess (early stop), at 210 s, or on a policy error\nno check runs between nodes or after the last one: Task 4 failures\nwith every node ok, and dragged neighbours, idle to the limit", kind="note", body_size=7.2)
    return d


def _chain(d, keys, x0, y, w, h, gap, labels, kinds):
    for i, (k, lab, kind) in enumerate(zip(keys, labels, kinds)):
        d.box(k, x0 + i * (w + gap), y, w, h, lab, kind=kind, title_size=7.6)
        if i:
            d.arrow(keys[i - 1], k)


@figure("teacher_task_graphs")
def teacher_task_graphs():
    d = Diagram(14.5, 8.3, "Task graphs: one linear node list per variant (graph.py); failure edges live in the executor")
    # transfer
    d.group(0.15, 5.35, 14.2, 2.35, "Tasks 1-3: transfer, cross transfer, vertical stacking (per operation K: 4 nodes)", color="#c07a1e", fill="#fffaf3")
    keys = ["a1", "a2", "a3", "a4"]
    _chain(d, keys, 0.4, 6.35, 2.25, 0.8, 0.35,
           ["tK_to_pregrasp\ntransit, 6 cm above", "tK_grasp\nPPO grasp_lift, lift 6 cm", "tK_to_preplace\ntransit, held, 0.08 m/s", "tK_place\nPPO place_release, 5 mm"],
           ["teacher", "learned", "teacher", "learned"])
    d.path([(9.75, 6.35), (9.75, 6.05), (1.5, 6.05), (1.5, 6.35)], "failed: open and restart at tK_to_pregrasp (3 attempts)", color="#a05a00", dashed=True, label_at=(5.6, 5.88))
    d.box("tvar", 10.65, 5.5, 3.55, 1.95, "Operations, arms, parks",
          "1-2 (18 nodes): t0, t1 back -> park_back_2\n      -> t2, t3 left -> park_left_end\n1-3 (27): left pair, back pair, left pair\n2-2 (10): t0 left -> park -> t1 back\n2-3 (15): back, left, back\n3-2 (9): back, back; 3-3 (13): back x3", kind="plain", body_size=7.0, align="left")
    # push-down
    d.group(0.15, 2.95, 14.2, 2.2, "Task 4: full-tower push-down (per block: 2 nodes; top layer first, slot 2 -> 0)", color="#c07a1e", fill="#fffaf3")
    _chain(d, ["p1", "p2"], 0.4, 3.85, 2.25, 0.8, 0.35, ["pK_to_prepush\ntransit, 100 mm behind", "pK_push\nPPO push (9a: 67-D)"], ["teacher", "learned"])
    d.path([(4.3, 3.85), (4.3, 3.55), (1.5, 3.55), (1.5, 3.85)], "failed: retry the push", color="#a05a00", dashed=True, label_at=(2.9, 3.38))
    d.box("pnote", 5.65, 3.15, 4.75, 1.7, "Verdicts", "ok: perceived departure (>= 0.16 m from the axis, or\n>= 0.09 m out on the table), or lost_departed\nfailed: structure (a tower block moved > 10 mm\nsince this node started), shoved (> 40 mm)\nstalled >= 9 cm out: planned push finishes it", kind="plain", body_size=7.0, align="left")
    d.box("pvar", 10.65, 3.15, 3.55, 1.7, "Blocks, arms, parks", "4-2 (14 nodes): p0-p2 back (+y,\n   l2_2, l2_1, l2_0) -> park_back_3 ->\n   p3-p5 left (+x) -> park_left_end\n4-3 (21): left L3 x3, back L2 x3,\n   left L1 x3", kind="plain", body_size=7.0, align="left")
    # insertion
    d.group(0.15, 0.2, 14.2, 2.55, "Task 5: tower insertion (per insert: 6 nodes)", color="#c07a1e", fill="#fffaf3")
    _chain(d, ["i1", "i2", "i3", "i4", "i5", "i6"], 0.4, 1.4, 2.05, 0.85, 0.25,
           ["iK_to_pregrasp\ntransit, 4 cm", "iK_grasp\nplanned, ramped lift", "iK_to_preinsert\nheld, 0.17 m out", "iK_insert\nplanned, 1 mm / row", "iK_to_prepushin\ntransit", "iK_push_in\nplanned, to centre"],
           ["teacher", "teacher", "teacher", "teacher", "teacher", "teacher"])
    d.text(0.4, 1.25, "grasp failed: restart at iK_to_pregrasp (3 attempts);   insert / push_in: on_fail skip (no re-insert, no re-push)\n"
           "distilled: true (student_v1) swaps the grasp, insert and push-in for BC networks; the verdicts stay", size=7.2, color="#a05a00")
    d.box("ivar", 0.4, 0.3, 13.75, 0.5, "5-2 (7 nodes): i0 left -> park_left_end      5-3 (14): i0 back (gap below) -> park_back_1 -> i1 left -> park_left_end", kind="plain", title_size=7.6)
    d.legend(10.65, 7.75, ["teacher", "learned"], step=1.55)
    return d


@figure("teacher_skill_training")
def teacher_skill_training():
    d = Diagram(14.5, 7.5, "PPO skill training in Isaac Lab (rsl_rl): one policy per skill and arm")
    d.box("cfg", 0.25, 5.75, 2.45, 1.05, "configs/skills/<skill>.yaml", "+ --override: rewards, reset\nmix, noise, PPO settings", kind="data", body_size=7.0, title_size=8.6)
    d.box("reset", 0.25, 2.25, 2.45, 3.15, "Reset (per env)", "(variant, stage) draw\ntier mix: D0 25 %,\nD1 25 %, D2 50 %\nblock mass, block and\nfinger friction, finger\neffort, tower poses\nobservation-noise biases\nstart pose by IK from a\nmixture of start modes", kind="sim", body_size=7.0)
    d.box("env", 3.05, 4.55, 4.2, 2.25, "Skill env (DirectRLEnv)", "light skill scene: one arm at its station base,\ntable, 6 foundations, movable slots, finger\ncontact sensors; 4096 envs (push: 3072)\nGPU PhysX, dt 1/120 s, decimation 12 -> 10 Hz\nepisode 150 steps (push 80)", kind="sim", body_size=7.0)
    d.box("obsa", 7.6, 5.6, 3.1, 1.2, "Actor observation", "noisy block poses + proprioception\n(obs.py: the same functions the\nexecutor calls at test time)", kind="teacher", body_size=7.0)
    d.box("obsc", 7.6, 4.1, 3.1, 1.2, "Critic observation", "actor + truth: contact forces, mass,\nblock / finger friction, joint\nvelocity, cmd - q, true poses", kind="note", body_size=7.0)
    d.box("ppo", 11.05, 4.1, 3.2, 2.7, "PPO (rsl_rl 3.1.2)", "actor MLP 256-256-128, ELU\ncritic MLP 512-256-128\nrunning observation normalizers\n24 steps per env per iteration\n5 epochs x 4 minibatches\ngamma 0.99, lambda 0.95, clip 0.2\nadaptive lr (KL 0.016)", kind="learned", body_size=7.0)
    d.box("act", 11.05, 2.25, 3.2, 1.45, "Action (10 Hz)", "delta pose, 6-D (arm base frame)\n+ gripper logit (grasp, place)\nclamp, EMA 0.3, distance scale", kind="learned", body_size=7.0)
    d.box("dec", 7.6, 2.25, 3.1, 1.45, "Decode -> joint targets", "apply_delta (max lead 3 cm)\nDLS IK + posture prior\nsag compensation (nominal mass)\ngripper rate limit", kind="sim", body_size=7.0)
    d.box("phys", 3.05, 2.25, 4.2, 1.45, "PhysX step, rewards, dones", "12 substeps, 1-substep delay\nrewards and terminations from the true state\nand the finger contact sensors", kind="sim", body_size=7.0)
    d.arrow("cfg", "env", sa="right", sb="left", tb=0.75)
    d.arrow("reset", "env", sa="right", sb="left", ta=0.85, tb=0.25)
    d.arrow("env", "obsa", sa="right", sb="left", ta=0.75)
    d.arrow("env", "obsc", sa="right", sb="left", ta=0.25)
    d.arrow("obsa", "ppo", sa="right", sb="left", tb=0.78)
    d.arrow("obsc", "ppo", sa="right", sb="left", tb=0.25)
    d.arrow("ppo", "act", "actor", sa="bottom", sb="top", label_dx=0.28, label_dy=-0.1)
    d.arrow("act", "dec", sa="left", sb="right")
    d.arrow("dec", "phys", sa="left", sb="right")
    d.arrow("phys", "env", "next observation", sa="top", sb="bottom", label_dx=0.75, label_dy=-0.12)
    d.box("ckpt", 11.4, 0.2, 2.85, 1.1, "model_*.pt", "results/skills/<skill>_<arm>/<run>/\n+ config.json, TensorBoard", kind="data", body_size=6.9)
    d.box("exp", 8.45, 0.2, 2.6, 1.1, "export.py", "TorchScript actor + normalizer\n(deterministic mean) + .json", kind="teacher", body_size=6.9)
    d.box("ev", 5.55, 0.2, 2.55, 1.1, "evaluate_skill.py", "GPU, 500 episodes per tier,\nWilson intervals per case", kind="sim", body_size=6.9)
    d.box("fb", 2.85, 0.2, 2.35, 1.1, "freeze_bundle", "bundle/skills/ + MANIFEST\n(sha256 per file)", kind="data", body_size=6.9)
    d.box("reh", 0.25, 0.2, 2.25, 1.1, "CPU rehearsal", "station scene, the\nevaluator's physics", kind="sim", body_size=6.9)
    d.path([(14.25, 4.6), (14.38, 4.6), (14.38, 1.6), (12.8, 1.6), (12.8, 1.3)], color="#9b3d8c")
    d.arrow("ckpt", "exp", sa="left", sb="right")
    d.arrow("exp", "ev", sa="left", sb="right")
    d.arrow("ev", "fb", sa="left", sb="right")
    d.arrow("fb", "reh", sa="left", sb="right")
    d.legend(0.3, 7.05 - 0.05, ["sim", "teacher", "learned", "note", "data"], step=1.85)
    return d


# ---------------------------------------------------------------------------------------------------------------- student


@figure("student_overview")
def student_overview():
    d = Diagram(14.5, 6.3, "Student v1 deployed: one 10 Hz call from the official evaluator to a joint chunk")
    d.box("ev", 0.25, 3.45, 2.85, 2.2, "Official evaluator", "organizers' Isaac Sim station,\nCPU PhysX, scorer (unpublished)\n10 tasks x 10 episodes\n210 s simulated per episode;\nsimulated time pauses\nduring a call", kind="external", body_size=7.1)
    d.box("net", 3.5, 4.25, 2.35, 1.2, "Cloudflare quick tunnel", "wss://<name>.trycloudflare.com\n-> ws://127.0.0.1:9001", kind="external", body_size=6.9, title_size=8.8)
    d.text(3.5, 4.1, "request: 4 RGB 480 x 640, q [2,7],\nwidths [2], sim time, step\nreply: infer_result (the chunk)", size=6.9, color="#6a4fa3")
    d.box("srv", 6.25, 3.45, 3.0, 2.2, "Policy server", "tower-infer-v1 handler (kit)\nmetadata, reset, infer;\none SessionModel per session;\na global lock serializes calls;\nlogs per call and per episode", kind="student", body_size=7.1)
    d.box("adp", 9.65, 3.45, 2.4, 2.2, "StudentAdapter", "parse the observation\nholds: blank, unconfident\n(first 3 calls), error,\nbad_chunk\nclip32 to the track bounds", kind="student", body_size=7.1)
    d.box("est", 12.45, 3.45, 1.85, 2.2, "Pose\nestimator", "kp_v2 (GPU)\n+ multi-view fit\n+ tracker\n86 ms", kind="learned", body_size=7.1)
    d.arrow("ev", "net", sa="right", sb="left", ta=0.72, both=True)
    d.arrow("net", "srv", sa="right", sb="left", both=True)
    d.arrow("srv", "adp")
    d.arrow("adp", "est", "images", label_dy=0.03)
    d.box("st", 12.45, 1.15, 1.85, 1.6, "State", "block pos / quat /\nconf [B]; q, widths,\nt; no privileged\nfields", kind="student", body_size=7.0)
    d.box("run", 8.15, 1.15, 3.9, 1.6, "Option runtime (CPU, ~9 ms)", "switcher bc_r2 at option ends\nnodes: planned transit / park; PPO grasp,\nplace, push (9a); distilled Task 5 nets\n1 row (contact) or <= 16 rows (transit)", kind="student", body_size=7.0)
    d.box("chk", 5.0, 1.15, 2.75, 1.6, "Chunk", "float32 [H, 16], H <= 16\nper arm: 7 joint targets\n(rad) + gripper width (m)", kind="student", body_size=7.0)
    d.arrow("est", "st", sa="bottom", sb="top")
    d.arrow("st", "run", sa="left", sb="right")
    d.arrow("run", "chk", sa="left", sb="right")
    d.arrow("chk", "srv", "infer_result", sa="top", sb="bottom", ta=0.7, tb=0.3, label_dx=-0.55, label_dy=-0.1)
    d.arrow("run", "est", "focus block", sa="top", sb="bottom", ta=0.93, tb=0.15, dashed=True, color="#2f8a5f", label_dx=-0.15, label_dy=-0.05)
    d.text(0.25, 2.7, "Per call, measured (BUILDLOG Step 11):\nserver 93 ms (estimator 86, runtime 9),\np99 147 ms; 10 sessions at once 0.95 s.\nLimits: 180 s per call, 7200 s per episode.\nThe evaluator rejects out-of-bounds chunks.", size=7.3, color="#444444")
    d.text(0.25, 0.75, "Reset: SessionModel.start = estimator reset + runtime built for the variant (~65 ms).", size=7.3, color="#444444")
    d.legend(8.15, 0.35, ["external", "student", "learned"], step=1.85)
    return d


@figure("student_perception")
def student_perception():
    d = Diagram(14.5, 6.8, "Pose estimator per call and session: kp_v2 network -> multi-view fit -> tracker")
    d.box("img", 0.25, 4.45, 2.3, 1.75, "Inputs", "4 RGB 480 x 640\n(static views turned 180 deg)\nq [2,7], widths [2]\nfocus block (runtime)", kind="external", body_size=7.1)
    d.box("net", 2.9, 4.45, 2.6, 1.75, "kp_v2 network", "ResNet-34 U-Net, per view\n4 maps: corner, edge,\nmask (unused), hidden\nbf16, ~40 % of a call", kind="learned", body_size=7.1)
    d.box("pk", 5.85, 4.45, 2.4, 1.75, "Detections", "corner peaks: NMS 5x5,\nthreshold 0.3, <= 400\nedge maxima along the\nnormals; sub-pixel", kind="student", body_size=7.1)
    d.box("cam", 8.6, 4.45, 2.55, 1.75, "Camera rig", "static: calibrated poses\nwrist: FK of measured q\nevaluator model (principal\npoint) vs render model", kind="student", body_size=7.1)
    d.box("pri", 11.5, 4.45, 2.8, 1.75, "Priors per block", "previous estimate (+ velocity)\nheld: pinch x offset\nat rest: anchor (0.5 mm)\ncall 0: nominal + search", kind="student", body_size=7.1)
    d.arrow("img", "net")
    d.arrow("net", "pk")
    d.box("vis", 11.5, 2.2, 2.8, 1.75, "Visibility", "ray cast against the other\nblocks' priors, masked by\nthe hidden map (arms and\nfingers are not modelled)", kind="student", body_size=7.1)
    d.box("fit", 8.0, 2.2, 3.15, 1.75, "Multi-view fit", "Gauss-Newton, 6 DoF per block,\n4 views jointly; Huber -> Tukey\n8 corners + 152 edge points\nfoundations: z pinned, yaw only", kind="student", body_size=7.1)
    d.box("rec", 4.65, 2.2, 3.0, 1.75, "Score + recovery", "inliers, ratio = score / expected\nre-search +-24 mm, +-15 deg,\n+-100 mm along; wide search\n(table, layer, leaning)", kind="student", body_size=7.1)
    d.box("trk", 0.25, 2.2, 4.05, 1.75, "Tracker", "low-pass at rest, anchoring, held offsets,\nfoundation freeze after 10 fits, coasting\n(x0.8, <= 5 calls), ghosts, the quaternion\ntwin nearest the previous output", kind="student", body_size=7.1)
    d.arrow("pk", "fit", sa="bottom", sb="top", ta=0.6, tb=0.25)
    d.arrow("cam", "fit", sa="bottom", sb="top", ta=0.4, tb=0.75)
    d.arrow("pri", "vis", sa="bottom", sb="top")
    d.arrow("vis", "fit", sa="left", sb="right")
    d.arrow("fit", "rec", sa="left", sb="right")
    d.arrow("rec", "trk", sa="left", sb="right")
    d.box("out", 0.25, 0.25, 3.3, 1.5, "Output [B blocks]", "pos [B,3] m, quat [B,4] wxyz\nconf 1 / 0.6 / 0.3 / 0\nB = 8 to 15 incl. foundations", kind="student", body_size=7.1)
    d.box("cons", 3.9, 0.25, 4.3, 1.5, "Consumers", "student runtime: tokens and skill nodes\nteacher executor (the camera teacher)\nlogs -> noise-model fit (v1, v2)", kind="student", body_size=7.1)
    d.box("acc", 8.55, 0.25, 5.75, 1.5, "Measured (kp_v2)", "at rest p50 / p90 / p99 0.18 / 0.62 / 1.08 mm; held p90 0.10-0.19 mm\nnearly hidden Task 5 insert: p90 7-18 mm; jumps to the empty\nslot below at conf 1.0; identity swaps; conf is not calibrated", kind="note", body_size=7.1)
    d.arrow("trk", "out", sa="bottom", sb="top", ta=0.4)
    d.arrow("out", "cons")
    d.text(4.65, 2.05, "The tracker's state (poses, velocities, held offsets, anchors) is the next call's prior.", size=7.0, color="#2f8a5f")
    return d


@figure("student_perception_training")
def student_perception_training():
    d = Diagram(14.5, 5.2, "Estimator training data and training (Pose-estimator-v1.MD)")
    d.group(0.15, 2.85, 10.75, 1.85, "Our renders", color="#4a6fa5", fill="#f6f8fc")
    d.box("tt", 0.35, 3.05, 2.4, 1.3, "teacher_t45_final", "rollouts, station scene,\nCPU physics, all ten\nvariants, D0 + D1", kind="teacher", body_size=7.0)
    d.box("rnd", 3.05, 3.05, 2.6, 1.3, "Isaac renders", "4 cameras, RGB + depth,\n4 RTX subframes, light\nrandomized; p 0.1 per row", kind="sim", body_size=7.0)
    d.box("lbl", 5.95, 3.05, 2.6, 1.3, "Labels", "poses, render camera model,\ncorner visibility (depth),\nid maps (254 = arm-hidden)", kind="student", body_size=7.0)
    d.box("r1", 8.85, 3.05, 1.85, 1.3, "r1 / r1_val", "28,227 / 7,372\nframes", kind="data", body_size=7.0)
    d.group(0.15, 0.8, 10.75, 1.85, "Organizer frames (the evaluator's renderer)", color="#6a4fa3", fill="#faf8fd")
    d.box("hdf", 0.35, 1.0, 2.4, 1.3, "TOWER-SimData", "HDF5 episodes, read over\nHTTP range requests", kind="external", body_size=7.0)
    d.box("fe", 3.05, 1.0, 2.6, 1.3, "fetch + correct", "frames of 73 episodes;\nfoundation poses corrected\nby image registration", kind="student", body_size=7.0)
    d.box("l2", 5.95, 1.0, 2.6, 1.3, "label_dataset_frames", "Isaac depth of the recorded\nstate, evaluator camera\nmodel, id maps", kind="student", body_size=7.0, title_size=8.6)
    d.box("ds", 8.85, 1.0, 1.85, 1.3, "dataset\ntrain / val", "2,800 / 720\nframes", kind="data", body_size=7.0)
    for a, b in (("tt", "rnd"), ("rnd", "lbl"), ("lbl", "r1"), ("hdf", "fe"), ("fe", "l2"), ("l2", "ds")):
        d.arrow(a, b)
    d.box("tr", 11.2, 1.0, 3.1, 3.7, "train_perception", "organizer frames drawn x3\nwarp, photometric, JPEG,\nblur, noise, 512 x 384 crops\nGaussian targets: corner\n1.5 px, edge 1 px\nquality focal + BCE losses\nAdamW, cosine, batch 12\nkp_v1: 25k iterations\nkp_v2: +12k (45 min)\n-> kp_v2.pt (98.8 MB)", kind="learned", body_size=7.0)
    d.arrow("r1", "tr", sa="right", sb="left", tb=0.72)
    d.arrow("ds", "tr", sa="right", sb="left", tb=0.25)
    d.text(0.25, 0.6, "Evaluation: single frames (eval_perception), whole episodes at 10 Hz (eval_perception_sequence), closed loop (rehearse.py --perception).", size=7.3, color="#444444")
    return d


@figure("student_runtime")
def student_runtime():
    d = Diagram(13.6, 6.9, "Option runtime (StudentRun.act): one call, State in, chunk out")
    d.box("state", 0.25, 4.85, 2.5, 1.3, "State", "block pos [B,3] m, quat [B,4],\nconf [B] (1 / 0.6 / 0.3 / 0),\nq [2,7] rad, widths [2] m, t", kind="student", body_size=7.0)
    d.box("rt", 3.1, 4.85, 3.0, 1.3, "Rest estimates + tokens", "robust rest poses, held block from\nthe pinch, IK reachability;\nTokenTracker: 1 s history, speeds,\ncalls since seen, fallen flag", kind="student", body_size=7.0)
    d.box("dp", 6.45, 4.85, 2.2, 1.3, "Decision point?", "first call, option end,\nor every 2 s while idle", kind="student", body_size=7.0)
    d.box("sw", 9.0, 4.85, 1.85, 1.3, "Switcher", "bc_r2 (CPU)\n1-5 ms", kind="learned", body_size=7.0)
    d.box("rules", 11.2, 4.85, 2.2, 1.3, "Mechanical rules", "masks: invalid -> idle\nopen rule for a held\nblock not delivered", kind="student", body_size=7.0)
    d.arrow("state", "rt")
    d.arrow("rt", "dp")
    d.arrow("dp", "sw", "yes", label_dy=0.03)
    d.arrow("sw", "rules", label_dy=0.03)
    d.box("mem", 0.25, 2.55, 2.5, 1.7, "Memory", "per goal: tries, attempts,\nlast option, status, why (11),\nalt state; per arm: holding,\nmoved since park; option_log", kind="student", body_size=7.0)
    d.box("ne", 3.1, 2.55, 3.0, 1.7, "Node end", "ok / failed / timeout + info\n(pos_err_mm, along_mm, ...)\na failed node aborts its option;\nan empty-handed arm opens", kind="student", body_size=7.0)
    d.box("nodes", 6.45, 2.55, 4.4, 1.7, "make_node -> node classes", "Transit, Park: planned IK, open loop, <= 16 rows\nLearnedGrasp / LearnedPlace: PPO (65-D / 73-D)\nPush: PPO 9a (67-D); distilled Task 5 (68-D)\nIdle: 2 s; contact + idle: 1 row per call", kind="student", body_size=7.0)
    d.box("spec", 11.2, 2.55, 2.2, 1.7, "TaskSpec.\noption_nodes", "two node specs per\nstage (transit +\ncontact); parks", kind="student", body_size=7.0)
    d.arrow("rules", "spec", "Decision", sa="bottom", sb="top", label_dx=0.45, label_dy=-0.1)
    d.arrow("spec", "nodes", sa="left", sb="right")
    d.path([(7.55, 4.85), (7.55, 4.25)], color="#333333")
    d.text(7.65, 4.68, "no: step the running node", size=7.0, color="#333333")
    d.arrow("nodes", "ne", "status", sa="left", sb="right", label_dy=0.03)
    d.arrow("ne", "mem", sa="left", sb="right")
    d.arrow("mem", "rt", "next tokens", sa="top", sb="bottom", ta=0.8, tb=0.2, dashed=True, color="#2f8a5f", label_dx=0.25, label_dy=-0.05)
    d.box("safe", 6.45, 0.3, 2.85, 1.55, "_safe", "non-finite chunk -> hold row\nbeyond joint / gripper limits\nby > 1e-6 -> clipped, counted", kind="student", body_size=7.0)
    d.box("chunk", 9.65, 0.3, 3.75, 1.55, "Chunk -> adapter", "float32 [H, 16]; per arm 7 sag-compensated\njoint targets + width; H <= 16 (transit,\npark), 1 (contact, idle); clip32 in the adapter", kind="student", body_size=7.0)
    d.arrow("nodes", "safe", "rows", sa="bottom", sb="top", ta=0.3, tb=0.5, label_dx=0.25, label_dy=-0.1)
    d.arrow("safe", "chunk")
    d.box("guard", 0.25, 0.3, 5.85, 1.55, "Guards (inherited from the executor, plus one)", "auto_park the other arm when its pinch is within 0.18 m of the transit target\nlivelock guard: skip a re-park of an arm that has not moved since its park\nno abort inside an option: decisions only at option ends", kind="note", body_size=7.0, dashed=True)
    return d


@figure("student_switcher")
def student_switcher():
    d = Diagram(13.6, 7.4, "Switcher (bc_r2): tokens -> Transformer -> masked (option, goal, arm)")
    d.text(0.2, 7.0, "7 options: acquire, deliver, push_off, insert, push_home, park, idle.   Features v1: no dynamics, force or contact features.", size=7.6, color="#555555")
    tok = [("glob", "global token  x1  [16]", "variant (10), family (3), time left,\nK / 9, decisions / 50", 0.85),
           ("arm", "arm tokens  x2  [27]", "pinch position rel. pads, rot6d, width,\nclosed command, holding, at home, moved\nsince park, last option (7), time since it", 1.05),
           ("blk", "block tokens  x16  [38] + mask", "role, position rel. pads, yaw (cos / sin 2 psi),\ntilt, confidence level, calls since seen, held\nby, displacement + 1 s speed, goal offset,\npush travel, slot coordinates, bound, fallen", 1.3),
           ("goal", "goal tokens  x9  [88] + mask", "type, k (one-hot), prescribed arm, the bound\nblock's token, goal pose rel. pad; history:\nattempts per arm, tries, last option, status,\nwhy (11), alt state; reachable per arm", 1.3)]
    y = 6.55
    for k, t, b, h in tok:
        y -= h
        d.box(k, 0.2, y, 3.45, h, t, b, kind="student", body_size=6.9)
        y -= 0.18
    d.box("norm", 4.05, 3.95, 1.9, 1.6, "Normaliser", "per-feature mean / std\nof training tokens,\nstd floor 0.1,\nclamp +-8 (exported)", kind="student", body_size=6.9)
    d.box("proj", 6.25, 3.95, 1.9, 1.6, "Projection", "per-type linear\nto d = 128\n+ type embedding\n28 tokens, pad mask", kind="learned", body_size=6.9)
    d.box("enc", 8.45, 3.8, 1.95, 1.9, "Transformer\nencoder", "pre-LN, 3 layers,\n4 heads, FF 384,\nGELU, dropout 0\n669,707 parameters", kind="learned", body_size=6.9)
    for k in ("glob", "arm", "blk", "goal"):
        d.arrow(k, "norm", sa="right", sb="left")
    d.arrow("norm", "proj")
    d.arrow("proj", "enc")
    heads = [("h_opt", "option head", "MLP on global -> 7", "learned"),
             ("h_goal", "goal pointer", "q(global, option) . goals -> 9", "learned"),
             ("h_arm", "arm head", "(global, option, goal) -> 2", "learned"),
             ("h_prog", "progress head", "P(goal satisfied) per goal;\ncomputed, read nowhere", "note"),
             ("h_val", "value head", "no loss yet (Step 12)", "note")]
    for i, (k, t, b, kind) in enumerate(heads):
        d.box(k, 10.8, 6.0 - i * 0.98, 2.25, 0.82, t, b, kind=kind, body_size=6.8, dashed=kind == "note", title_size=8.6)
        d.arrow("enc", k, sa="right", sb="left", ta=0.9 - i * 0.2)
    d.box("mask", 4.05, 0.3, 3.0, 1.45, "Action masks", "valid_opt [7]: the family's options\nvalid_goal [7, 9]: 0 <= k < K\nvalid_arm [7, 9, 2]: deliver and\ninsert only by the holding arm", kind="student", body_size=6.9)
    d.box("dec", 7.4, 0.3, 2.9, 1.45, "Masked argmax", "option -> goal -> arm\n(-1e9 fills, deterministic)\npark -> park(None, arm)\nno goal for a goal option -> idle", kind="student", body_size=6.9)
    d.box("out", 10.8, 0.3, 2.25, 1.45, "Decision", "(option, goal, arm)\n-> the runtime, which\nre-applies the masks", kind="student", body_size=6.9)
    d.arrow("mask", "dec")
    d.arrow("dec", "out")
    for yy in (6.41, 5.43, 4.45):
        d.ax.plot([13.05, 13.22], [yy, yy], color="#9b3d8c", lw=1.2, zorder=4)
    d.path([(13.22, 6.41), (13.22, 1.98), (8.85, 1.98), (8.85, 1.75)], color="#9b3d8c")
    d.text(9.0, 2.27, "option, goal, arm logits", size=6.8, color="#9b3d8c")
    return d


@figure("student_training")
def student_training():
    d = Diagram(14.6, 8.5, "Student training pipeline (Steps 0-11): from the teacher and the oracle to the bundle")
    d.group(0.15, 5.35, 14.3, 2.45, "Decisions: the switcher, distilled from the state oracle (BC + DAgger)", color="#2f8a5f", fill="#f5fbf8")
    d.box("tg", 0.35, 5.6, 1.95, 1.75, "Teacher graphs\n+ skills", "teacher_t45_final:\n65-D pushes,\nplanned Task 5", kind="teacher", body_size=7.0)
    d.box("orc", 2.6, 5.6, 2.2, 1.75, "Oracle", "graph mode = executor\n(G3a 480 / 480)\nstate mode: truth labels\n(G3b 447 vs 446 / 480)", kind="teacher", body_size=7.0)
    d.box("roll", 5.1, 5.6, 2.35, 1.75, "Isaac CPU rollouts", "rehearse_many, 4 envs\nsim poses + noise v1\nexploration, injection\noracle relabels online", kind="sim", body_size=7.0)
    d.box("r0", 7.75, 5.6, 1.45, 1.75, "r0", "3,000 ep.\n50,457 dec.\noracle acts\neps 0.1", kind="data", body_size=7.0)
    d.box("bc0", 9.45, 6.6, 1.15, 0.75, "bc_r0", kind="learned")
    d.box("r1", 9.45, 5.6, 1.15, 0.75, "r1", kind="data")
    d.box("bc1", 10.85, 6.6, 1.15, 0.75, "bc_r1", kind="learned")
    d.box("r2", 10.85, 5.6, 1.15, 0.75, "r2", kind="data")
    d.box("bc2", 12.3, 5.6, 2.0, 1.75, "bc_r2", "6,000 ep., 84,273 dec.\n404 vs oracle 405 / 480\nrecovery 55.5 vs 59.3 %\n(G8 passed)", kind="learned", body_size=7.0)
    d.arrow("tg", "orc")
    d.arrow("orc", "roll")
    d.arrow("roll", "r0")
    d.arrow("r0", "bc0", sa="right", sb="left", ta=0.8)
    d.arrow("bc0", "r1", "beta 0.5", sa="bottom", sb="top", label_dx=0.38, label_dy=-0.09)
    d.arrow("r1", "bc1", sa="right", sb="left", ta=0.8, tb=0.2)
    d.arrow("bc1", "r2", "beta 0", sa="bottom", sb="top", label_dx=0.33, label_dy=-0.09)
    d.arrow("r2", "bc2", sa="right", sb="left", tb=0.22)
    d.text(7.75, 5.52, "BC from scratch on all rounds: CE(option, goal, arm), x5 non-nominal, + 0.5 BCE(progress)", size=6.9, color="#555555")
    d.group(0.15, 3.65, 6.9, 1.5, "Estimator noise model", color="#4a6fa5", fill="#f6f8fc")
    d.box("logs", 0.35, 3.85, 1.95, 1.0, "camera-loop\nkp_v2 logs", kind="data", title_size=8.5)
    d.box("fit", 2.6, 3.85, 1.95, 1.0, "fit_estimator\n_noise", kind="student", title_size=8.5)
    d.box("nz", 4.85, 3.85, 2.0, 1.0, "v1 (switcher)\nv2 (9a)", kind="data", title_size=8.5)
    d.arrow("logs", "fit")
    d.arrow("fit", "nz")
    d.path([(5.85, 4.85), (5.85, 5.25), (6.2, 5.25), (6.2, 5.6)], color="#4a6fa5", dashed=True)
    d.text(0.35, 3.78, "G5 (noise matches the camera loop): not passed with v2; v3 to do", size=6.9, color="#a05a00")
    d.group(0.15, 0.2, 9.1, 3.25, "Skills: fine-tuned and distilled", color="#9b3d8c", fill="#fcf7fb")
    d.box("pp", 0.35, 2.15, 1.95, 1.0, "push v3 / v1c\n(teacher PPO)", kind="learned", title_size=8.3)
    d.box("wid", 2.6, 2.15, 1.95, 1.0, "widen 65 -> 67\n+ seen, since_seen", kind="student", title_size=8.0)
    d.box("ppo", 4.85, 2.15, 2.0, 1.0, "PPO under v2\ntarget losses", kind="sim", title_size=8.3)
    d.box("p9a", 7.15, 2.15, 1.9, 1.0, "9a pushes", "left 98.8 %, back 95.4 %\nTask 4 53 vs 84 / 96", kind="learned", body_size=6.8)
    d.arrow("pp", "wid")
    d.arrow("wid", "ppo")
    d.arrow("ppo", "p9a")
    d.path([(5.85, 3.85), (5.85, 3.15)], color="#4a6fa5", dashed=True)
    d.box("t5", 0.35, 0.45, 1.95, 1.2, "planned Task 5\ncontrollers", kind="teacher", title_size=8.3)
    d.box("dart", 2.6, 0.45, 1.95, 1.2, "DART collection", "Isaac CPU, sim poses\n1,200 ep., 328k rows", kind="sim", body_size=6.8)
    d.box("trd", 4.85, 0.45, 2.0, 1.2, "train_distill", "MLP 68-256-256-256-7\nper skill and arm", kind="student", body_size=6.8)
    d.box("dn", 7.15, 0.45, 1.9, 1.2, "distilled Task 5", "grasp, insert, push-in\nG9c 83 vs 78 / 96", kind="learned", body_size=6.8)
    d.arrow("t5", "dart")
    d.arrow("dart", "trd")
    d.arrow("trd", "dn")
    d.text(0.35, 2.05, "9b grip fine-tunes (finger effort 3 / 22 N): trained, not in student_v1; grip matrix not re-run", size=6.8, color="#a05a00")
    d.box("kp", 9.5, 3.85, 2.2, 1.0, "kp_v2 estimator", "Pose-estimator-v1.MD", kind="learned", body_size=6.8)
    d.box("frz", 9.5, 1.75, 2.2, 1.6, "freeze_student\n_bundle", "push9a + bc_r2 +\ndistilled v2 + kp_v2;\ngraphs: distilled true", kind="student", body_size=6.8)
    d.box("v1", 12.0, 2.35, 2.35, 1.0, "student_v1", "tar sha256 01ab88da...", kind="data", body_size=6.8)
    d.box("g10", 12.0, 1.0, 2.35, 1.0, "G10 camera loop", "135 vs teacher 131 / 160", kind="sim", body_size=6.8)
    d.box("srv", 12.0, 0.2, 2.35, 0.6, "server: G11a, official run", kind="external", title_size=7.6)
    d.arrow("bc2", "frz", sa="bottom", sb="right", ta=0.3, tb=0.85)
    d.arrow("kp", "frz", sa="bottom", sb="top")
    d.arrow("p9a", "frz", sa="right", sb="left", tb=0.6)
    d.arrow("dn", "frz", sa="right", sb="left", tb=0.2)
    d.arrow("frz", "v1", sa="right", sb="left", ta=0.62)
    d.arrow("v1", "g10", sa="bottom", sb="top")
    d.arrow("g10", "srv", sa="bottom", sb="top")
    for i, k in enumerate(["teacher", "student", "learned", "sim", "data"]):
        d.legend(12.15, 4.95 - i * 0.27, [k])
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--png", default=None, help="also write PNG previews to this directory")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    png = Path(args.png).expanduser() if args.png else None
    for name, fn in FIGURES.items():
        if args.only and name not in args.only:
            continue
        sub = "teacher-v1" if name.startswith("teacher") else "student-v1"
        fn().save(out / sub / f"{name}.svg", png)


if __name__ == "__main__":
    main()
