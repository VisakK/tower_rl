"""Student task specs (plan Step 1): every graph, in the repo and in the teacher bundle, round-trips through its spec."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tower_rl.config import load_yaml
from tower_rl.data.layouts import ALL_VARIANTS, VARIANTS, Layout
from tower_rl.graph.executor import Executor
from tower_rl.graph.graph import load_graph
from tower_rl.graph.interface import State, Status
from tower_rl.paths import RESULTS_DIR
from tower_rl.student.spec import FAMILY_STAGES, Decision, TaskSpec

BUNDLE = RESULTS_DIR / "bundles" / "teacher_t45_final"


def _graphs():
    out = [(v, "repo", load_graph(v)) for v in ALL_VARIANTS]
    if (BUNDLE / "graphs").is_dir():
        out += [(v, "bundle", load_graph(v, BUNDLE / "graphs" / f"{v}.json")) for v in ALL_VARIANTS]
    return out


@pytest.mark.parametrize("variant,src,graph", _graphs(), ids=lambda x: x if isinstance(x, str) else "")
def test_graph_round_trip(variant, src, graph):
    before = copy.deepcopy(graph)
    spec = TaskSpec.from_graph(graph)
    assert spec.graph_nodes() == graph["nodes"]  # every node, every field, in order
    assert graph == before  # building the spec does not touch the graph
    # every non-park node belongs to exactly one (goal, stage); stages in family order
    staged = [n for g in spec.goals for st in spec.stages for n in g.stages[st]]
    assert len(staged) == sum(n["skill"] != "park" for n in graph["nodes"])
    assert len({id(n) for n in staged}) == len(staged)
    assert tuple(spec.goals[0].stages) == FAMILY_STAGES[spec.family]
    # goals in the layout's prescribed order, with its blocks and arms
    lay = Layout.load(variant)
    plan = lay.pushes if spec.family == "pushdown" else lay.transfers
    assert [(g.block, g.arm) for g in spec.goals] == [(t.block, t.arm) for t in plan]
    # top-level settings copied from the graph
    assert spec.retries == graph["retries"] and spec.chunk == graph["chunk"] and spec.movable == graph["movable"]
    assert spec.arm_switch_m == graph.get("arm_switch_m", 0.015)
    assert spec.family == graph.get("family", "transfer")
    assert spec.gripper_closed_width == graph.get("gripper_closed_width")
    assert spec.grasp_planned == graph.get("grasp_planned", False)


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_option_nodes(variant):
    spec = TaskSpec.load(variant)
    for g in spec.goals:
        other = next(a for a in ("left", "back") if a != g.arm)
        for st in spec.stages:
            own = spec.option_nodes(Decision(st, g.k, g.arm))
            assert own == g.stages[st] and all(a is not b for a, b in zip(own, g.stages[st]))  # copies
            own[0]["id"] = "changed"
            assert g.stages[st][0]["id"] != "changed"
            alt = spec.option_nodes(Decision(st, g.k, other))
            assert [n["id"] for n in alt] == [n["id"] + "_alt" for n in g.stages[st]]
            assert all(n["arm"] == other and n["first_of_transfer"] is False for n in alt)
            assert [{k: v for k, v in n.items() if k not in ("id", "arm", "first_of_transfer")} for n in alt] == \
                   [{k: v for k, v in n.items() if k not in ("id", "arm", "first_of_transfer")} for n in g.stages[st]]
    assert spec.option_nodes(Decision("idle")) == []
    with pytest.raises(ValueError):
        spec.option_nodes(Decision("push_off" if spec.family != "pushdown" else "deliver", 0, "left"))


def test_park_specs():
    spec = TaskSpec.load("task1-3_tower_transfer_3")
    g = spec.graph["nodes"]
    assert spec.park_spec("left", 2, "graph") == next(n for n in g if n["id"] == "park_left_2")
    assert spec.park_spec("left", None, "end") == g[-1] and g[-1]["id"] == "park_left_end"
    assert spec.park_spec("back", 1, "graph")["id"] == "park_back_1"  # none in the graph: same naming, the graph's timeout
    assert spec.park_spec("back", 1, "graph")["timeout_s"] == 12.0
    assert spec.park_spec("left", 3, "alt") == {"id": "t3_alt_park_left", "skill": "park", "arm": "left", "transfer": 3,
                                                 "timeout_s": 10.0}
    assert spec.park_spec("back", 4, "auto")["id"] == "auto_park_back"
    assert spec.option_nodes(Decision("park", 2, "left")) == [spec.park_spec("left", 2, "graph")]
    assert sorted(spec.parks_before) == [2, 4] and spec.K == 6


def test_load_from_bundle_and_graph_set():
    v = "task2-2_cross_tower_transfer_2"
    spec = TaskSpec.load(v, BUNDLE, graph_set={"gripper_closed_width": 0.0425, "retries": 1})
    assert spec.gripper_closed_width == 0.0425 and spec.retries == 1 and spec.graph["retries"] == 1
    assert spec.graph_nodes() == load_graph(v)["nodes"]
    spec = TaskSpec.load(v, "/nonexistent_bundle")  # no graph in the bundle: the repo's (as Executor.from_bundle)
    assert spec.graph == load_graph(v)


def test_bad_graphs_are_rejected():
    g = load_graph("task2-2_cross_tower_transfer_2")
    bad = copy.deepcopy(g)
    bad["nodes"] = [n for n in bad["nodes"] if n["id"] != "park_left_1"]  # arm switch without a park
    with pytest.raises(ValueError):
        TaskSpec.from_graph(bad)
    bad = copy.deepcopy(g)
    bad["nodes"][1], bad["nodes"][2] = bad["nodes"][2], bad["nodes"][1]  # deliver's transit inside acquire
    with pytest.raises(ValueError):
        TaskSpec.from_graph(bad)
    bad = copy.deepcopy(g)
    bad["nodes"][1]["block"] = "source_l1_center"  # a goal acting on two blocks
    with pytest.raises(ValueError):
        TaskSpec.from_graph(bad)


def _state(variant: str) -> State:
    lay = Layout.load(variant)
    ids = list(lay.blocks)
    home = load_yaml("scene")["home_qpos"]
    return State(t=0.0, step=0, variant=variant, instruction="", q=np.array([home["left"][:7], home["back"][:7]]),
                 width=np.array([0.1, 0.1]), block_ids=ids, block_pos=np.stack([lay.blocks[b][:3] for b in ids]),
                 block_quat=np.stack([lay.blocks[b][3:7] for b in ids]), block_conf=np.ones(len(ids)))


class _Failed:
    def __init__(self, info):
        self.info = info


@pytest.mark.parametrize("variant", [v for v in VARIANTS if not v.startswith(("task4", "task5"))])
def test_alt_options_match_the_executor_arm_switch(variant):
    """The spec's alt park + other-arm acquire / deliver + park equal the node list the executor queues."""
    spec = TaskSpec.load(variant)
    for g in spec.goals:
        ex = Executor(variant)
        ex.reset(1, variant, [0])
        run = ex.runs[0]
        run._pregrasp_reachable = lambda arm, bid, st, own=g.arm: arm != own
        run.i = next(j for j, n in enumerate(run.g["nodes"]) if n["skill"] == "place_release" and n["transfer"] == g.k)
        run.node = _Failed({"place_xy_mm": 5.0})
        run._advance(Status.FAILED, _state(variant))
        other = next(a for a in ("left", "back") if a != g.arm)
        want = [spec.park_spec(g.arm, g.k, "alt")] + spec.option_nodes(Decision("acquire", g.k, other)) + \
            spec.option_nodes(Decision("deliver", g.k, other)) + [spec.park_spec(other, g.k, "alt")]
        assert json.loads(json.dumps(run.alt_queue)) == json.loads(json.dumps(want))


def test_bundle_graphs_exist():
    if not BUNDLE.is_dir():
        pytest.skip("teacher bundle not present")
    assert all((BUNDLE / "graphs" / f"{v}.json").is_file() for v in ALL_VARIANTS)
    assert Path(BUNDLE / "MANIFEST.json").is_file()
