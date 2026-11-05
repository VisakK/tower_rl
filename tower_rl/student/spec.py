"""Task specs (student plan Step 1): a variant's goals and option node templates, built from its graph JSON.

A graph (``tower_rl/graph/graphs/<variant>.json`` or a bundle's copy) lists, per transfer k in the prescribed order,
the nodes of each stage of its family, plus a park whenever the next transfer uses the other arm and one at the end:

  transfer   acquire   = t{k}_to_pregrasp, t{k}_grasp          deliver   = t{k}_to_preplace, t{k}_place
  pushdown   push_off  = p{k}_to_prepush, p{k}_push
  insertion  acquire   = i{k}_to_pregrasp, i{k}_grasp          insert    = i{k}_to_preinsert, i{k}_insert
             push_home = i{k}_to_prepushin, i{k}_push_in

A ``TaskSpec`` holds one ``Goal`` per transfer with its stages' node specs, the graph's parks and its top-level
settings. ``option_nodes(decision)`` returns the node specs an option runs (deep copies; an arm other than the
prescribed one gets the executor's arm-switch naming: ``_alt`` ids, ``first_of_transfer`` false). The graph's node
list is recovered exactly by concatenating the options in prescribed order with the parks reinserted
(``graph_nodes``). Pure python; the runtime uses ``spec.graph`` as the executor's ``g``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

from tower_rl.graph.graph import load_graph
from tower_rl.graph.interface import ARMS

OPTIONS = ("acquire", "deliver", "push_off", "insert", "push_home", "park", "idle")
FAMILIES = ("transfer", "pushdown", "insertion")
FAMILY_STAGES = {"transfer": ("acquire", "deliver"), "pushdown": ("push_off",),
                 "insertion": ("acquire", "insert", "push_home")}
GOAL_TYPES = ("place", "push_off", "insert")
GOAL_TYPE = {"transfer": "place", "pushdown": "push_off", "insertion": "insert"}
PREFIX = {"transfer": "t", "pushdown": "p", "insertion": "i"}
TRANSPORT = ("deliver", "insert")  # stages that carry the held block to its goal
WHY = ("none", "ok", "dropped", "unreachable", "structure", "shoved", "misplaced", "stalled", "lost", "timeout", "other")
STAGE_NODES = {  # stage -> ((skill, transit target), id suffix) of its nodes, in order
    "acquire": ((("transit", "pregrasp"), "to_pregrasp"), (("grasp_lift", None), "grasp")),
    "deliver": ((("transit", "preplace"), "to_preplace"), (("place_release", None), "place")),
    "push_off": ((("transit", "prepush"), "to_prepush"), (("push", None), "push")),
    "insert": ((("transit", "preinsert"), "to_preinsert"), (("insert", None), "insert")),
    "push_home": ((("transit", "prepushin"), "to_prepushin"), (("push_in", None), "push_in")),
}
STAGE_OF = {key: st for st, nodes in STAGE_NODES.items() for key, _ in nodes}
GEOMETRY = ("dest_layer", "goal_in_pad", "goal_z", "direction", "side_blocks", "layer")
ALT_PARK_TIMEOUT_S = 10.0  # the executor's inserted parks (arm switch, arm_clear)


def node_key(n: dict) -> tuple:
    return n["skill"], (n.get("target") if n["skill"] == "transit" else None)


@dataclass(frozen=True)
class Decision:
    """One switcher / oracle decision. ``goal``: the goal index (a park: the index its id carries, None at the end);
    ``park``: "graph" (the park between transfers), "end", "alt" (the arm-switch sequence) or "auto"; ``alt``: an item
    of the oracle's arm-switch sequence (graph mode runs it without the arm_clear check, as the executor does)."""
    option: str
    goal: int | None = None
    arm: str | None = None
    park: str | None = None
    alt: bool = False

    def __post_init__(self):
        assert self.option in OPTIONS, self.option
        assert self.arm is None or self.arm in ARMS, self.arm


@dataclass
class Goal:
    k: int
    type: str                      # place | push_off | insert
    block: str
    arm: str                       # the prescribed arm (demonstrations)
    stages: dict[str, list[dict]]  # stage -> the graph's node specs, family order
    geometry: dict = field(default_factory=dict)  # dest_layer, goal_in_pad, goal_z, direction, side_blocks, layer

    def __getattr__(self, name):
        if name in GEOMETRY:
            return self.__dict__["geometry"].get(name)
        raise AttributeError(name)

    def on_fail(self, stage: str) -> str:
        """The failure policy of the stage's contact node (graph ``on_fail``, default retry)."""
        return self.stages[stage][-1].get("on_fail", "retry")


class TaskSpec:
    """Goals, option node templates and settings of one variant (from its graph dict, kept as ``graph``)."""

    def __init__(self, graph: dict):
        self.graph = graph
        self.variant = graph["variant"]
        self.family = graph.get("family", "transfer")
        assert self.family in FAMILIES, self.family
        self.stages = FAMILY_STAGES[self.family]
        self.retries = graph["retries"]
        self.chunk = graph["chunk"]
        self.foundation = graph["foundation"]
        self.movable = graph["movable"]
        self.tower = graph.get("tower")
        self.perception = graph.get("perception")
        self.gripper_closed_width = graph.get("gripper_closed_width")
        self.grasp_planned = graph.get("grasp_planned", False)
        self.rate_hz = graph.get("rate_hz", 10)
        self.arm_switch_m = graph.get("arm_switch_m", 0.015)
        self.goals: list[Goal] = []
        self.parks_before: dict[int, dict] = {}  # goal k -> the graph's park preceding its first node
        self.end_park: dict | None = None
        self._build()

    # ------------------------------------------------------------------ construction and validation
    def _build(self):
        nodes = self.graph["nodes"]
        by_goal: dict[int, dict[str, list[dict]]] = {}
        order: list[tuple[int, str]] = []  # (goal, stage) in node order
        for i, n in enumerate(nodes):
            if n["skill"] == "park":
                continue
            key = node_key(n)
            if key not in STAGE_OF:
                raise ValueError(f"{self.variant}: node {n['id']} ({key}) belongs to no stage")
            st = STAGE_OF[key]
            if st not in self.stages:
                raise ValueError(f"{self.variant}: node {n['id']} has stage {st}, not one of the {self.family} family")
            k = n["transfer"]
            by_goal.setdefault(k, {}).setdefault(st, []).append(n)
            if not order or order[-1] != (k, st):
                order.append((k, st))
        ks = sorted(by_goal)
        if ks != list(range(len(ks))):
            raise ValueError(f"{self.variant}: transfers {ks} are not 0..K-1")
        expect = [(k, st) for k in ks for st in self.stages]
        if order != expect:
            raise ValueError(f"{self.variant}: stages out of the family order: {order}")
        for k in ks:
            stages = by_goal[k]
            for st in self.stages:
                got = [(node_key(n), n["id"]) for n in stages[st]]
                want = [(key, f"{PREFIX[self.family]}{k}_{suffix}") for key, suffix in STAGE_NODES[st]]
                if got != want:
                    raise ValueError(f"{self.variant}: goal {k} stage {st}: nodes {got}, expected {want}")
            members = [n for st in self.stages for n in stages[st]]
            blocks = {n["block"] for n in members}
            arms = {n["arm"] for n in members}
            if len(blocks) != 1 or len(arms) != 1:
                raise ValueError(f"{self.variant}: goal {k} mixes blocks {blocks} or arms {arms}")
            geo = {}
            for f in GEOMETRY:
                for n in members:
                    if f in n:
                        geo[f] = n[f]
                        break
            self.goals.append(Goal(k, GOAL_TYPE[self.family], blocks.pop(), arms.pop(), stages, geo))
        self._parks(nodes)

    def _parks(self, nodes: list[dict]):
        """Parks only between transfers that switch arms (parking the previous arm) and once at the end."""
        K = len(self.goals)
        for i, n in enumerate(nodes):
            if n["skill"] != "park":
                continue
            prev = next((m for m in reversed(nodes[:i]) if m["skill"] != "park"), None)
            nxt = next((m for m in nodes[i + 1:] if m["skill"] != "park"), None)
            if nxt is None:
                if i != len(nodes) - 1 or self.end_park is not None:
                    raise ValueError(f"{self.variant}: end park {n['id']} is not the last node")
                if n["transfer"] != K or (prev is not None and prev["arm"] != n["arm"]):
                    raise ValueError(f"{self.variant}: end park {n['id']} does not park the last transfer's arm")
                self.end_park = n
                continue
            k = nxt["transfer"]
            ok = prev is not None and prev["arm"] == n["arm"] and nxt["arm"] != n["arm"] and n["transfer"] == k \
                and nxt.get("first_of_transfer") and nodes[i + 1] is nxt and k not in self.parks_before
            if not ok:
                raise ValueError(f"{self.variant}: park {n['id']} is not between transfers that switch arms")
            self.parks_before[k] = n
        for k in range(1, K):
            if self.goals[k].arm != self.goals[k - 1].arm and k not in self.parks_before:
                raise ValueError(f"{self.variant}: no park between goals {k - 1} and {k}, which switch arms")

    # ------------------------------------------------------------------ queries
    @classmethod
    def from_graph(cls, graph: dict) -> TaskSpec:
        return cls(graph)

    @classmethod
    def load(cls, variant: str, bundle: str | Path | None = None, graph_set: dict | None = None) -> TaskSpec:
        """The graph from ``bundle``/graphs if it has the variant (as ``Executor.from_bundle``), else the repo's;
        ``graph_set`` is merged into its top level (``rehearse.py --graph_set``)."""
        gp = Path(bundle) / "graphs" / f"{variant}.json" if bundle else None
        graph = load_graph(variant, gp if gp is not None and gp.exists() else None)
        graph.update(graph_set or {})
        return cls(graph)

    @property
    def K(self) -> int:
        return len(self.goals)

    @property
    def options(self) -> tuple[str, ...]:
        """The option types of this family (park and idle always)."""
        return tuple(self.stages) + ("park", "idle")

    def park_timeout(self) -> float:
        p = self.end_park or next(iter(self.parks_before.values()), None)
        return p["timeout_s"] if p else 12.0

    def park_spec(self, arm: str, k: int | None, kind: str = "graph") -> dict:
        """A park node spec named as the executor names it: ``park_{arm}_{k}`` (between transfers; the graph's own when
        it has one), ``park_{arm}_end``, ``t{k}_alt_park_{arm}`` (arm-switch sequence) or ``auto_park_{arm}``."""
        if kind == "graph":
            p = self.parks_before.get(k)
            if p is not None and p["arm"] == arm:
                return copy.deepcopy(p)
            return {"id": f"park_{arm}_{k}" if k is not None else f"park_{arm}", "skill": "park", "arm": arm,
                    "transfer": k if k is not None else self.K, "timeout_s": self.park_timeout(), "first_of_transfer": False}
        if kind == "end":
            p = self.end_park
            if p is not None and p["arm"] == arm:
                return copy.deepcopy(p)
            return {"id": f"park_{arm}_end", "skill": "park", "arm": arm, "transfer": self.K, "timeout_s": self.park_timeout(),
                    "first_of_transfer": False}
        if kind == "alt":
            return {"id": f"t{k}_alt_park_{arm}", "skill": "park", "arm": arm, "transfer": k, "timeout_s": ALT_PARK_TIMEOUT_S}
        if kind == "auto":
            return {"id": f"auto_park_{arm}", "skill": "park", "arm": arm, "transfer": k, "timeout_s": ALT_PARK_TIMEOUT_S}
        raise ValueError(kind)

    def option_nodes(self, d: Decision) -> list[dict]:
        """The node specs option ``d`` runs: deep copies of its stage's nodes with ``arm`` replaced (another arm than the
        prescribed one: ids with ``_alt``, ``first_of_transfer`` false, as ``EnvRun._schedule_arm_switch``); a park's
        single node; nothing for idle."""
        if d.option == "idle":
            return []
        if d.option == "park":
            return [self.park_spec(d.arm, d.goal, d.park or "graph")]
        g = self.goals[d.goal]
        if d.option not in g.stages:
            raise ValueError(f"{self.variant}: option {d.option} is not a stage of the {self.family} family")
        nodes = copy.deepcopy(g.stages[d.option])
        arm = d.arm or g.arm
        if arm != g.arm:
            nodes = [dict(n, id=n["id"] + "_alt", arm=arm, first_of_transfer=False) for n in nodes]
        return nodes

    def graph_nodes(self) -> list[dict]:
        """The graph's node list rebuilt from the options in prescribed order with the parks reinserted."""
        out = []
        for g in self.goals:
            if g.k in self.parks_before:
                p = self.parks_before[g.k]
                out.append(self.park_spec(p["arm"], g.k, "graph"))
            for st in self.stages:
                out += self.option_nodes(Decision(st, g.k, g.arm))
        if self.end_park is not None:
            out.append(self.park_spec(self.end_park["arm"], None, "end"))
        return out

    def goal_of_block(self, bid: str) -> int | None:
        return next((g.k for g in self.goals if g.block == bid), None)
