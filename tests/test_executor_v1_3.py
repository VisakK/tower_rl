"""v1.3 executor additions (opt-in): the place factory's switch to the planned place after an off-centre grip (graph
"offcentre_place"). The node classes are stubbed: the switch is pure logic; the nodes themselves run in rehearsals."""

import tower_rl.graph.executor as ex


def test_offcentre_place_switch(monkeypatch):
    class Stub:
        def __init__(self, *a, **k):
            self.offcentre = False

    class Planned(Stub):
        pass

    class Learned(Stub):
        pass

    monkeypatch.setattr(ex, "PlannedPlaceNode", Planned)
    monkeypatch.setattr(ex, "LearnedPlaceNode", Learned)

    class Obj:
        pass

    run = Obj()
    run.ex = Obj()
    spec = {"skill": "place_release", "arm": "left", "block": "b"}
    oc = {"offcentre_place": {"min_mm": 4.0}}
    cases = [({}, 9.0, True, Learned, False),        # flag off: the learned place, whatever the grip
             (oc, 3.9, True, Learned, False),        # under the threshold
             (oc, -4.0, True, Planned, True),        # at the threshold, either sign
             (oc, 6.5, True, Planned, True),
             (oc, None, True, Learned, False),       # no attached block: no switch
             (oc, 6.5, False, Planned, False)]       # no learned policy: the old fallback, not marked off-centre
    for g, along, has_pol, want, marked in cases:
        run.g = g
        run.ex.skills = {("place_release", "left"): object()} if has_pol else {}
        monkeypatch.setattr(ex, "held_along_mm", lambda r, s, arm, bid, v=along: v)
        node = ex._make_node(run, spec, None)
        assert type(node) is want and node.offcentre == marked, (g, along, has_pol)
