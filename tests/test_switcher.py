"""Switcher network (student plan Step 7): masked decoding never chooses an invalid (option, goal, arm), the TorchScript
export reproduces the module, a feature-version mismatch is refused, the loss trains every head."""

import json

import numpy as np
import pytest
import torch

from tower_rl.student.spec import OPTIONS
from tower_rl.student.switcher import Normalizer, SwitcherDecider, SwitcherNet, export, loss_fn
from tower_rl.student.tokens import ARM_DIM, BLK_DIM, GLOB_DIM, GOAL_DIM, MAX_BLOCKS, MAX_GOALS

torch.set_num_threads(1)


def _batch(n: int, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    K = 6
    b = {"blk": torch.randn(n, MAX_BLOCKS, BLK_DIM, generator=g), "goal": torch.randn(n, MAX_GOALS, GOAL_DIM, generator=g),
         "arm": torch.randn(n, 2, ARM_DIM, generator=g), "glob": torch.randn(n, GLOB_DIM, generator=g)}
    b["blk_mask"] = torch.zeros(n, MAX_BLOCKS, dtype=torch.bool)
    b["blk_mask"][:, :12] = True
    b["goal_mask"] = torch.zeros(n, MAX_GOALS, dtype=torch.bool)
    b["goal_mask"][:, :K] = True
    vo = torch.zeros(n, 7, dtype=torch.bool)
    vo[:, [0, 1, 5, 6]] = True  # transfer family: acquire, deliver, park, idle
    vg = torch.zeros(n, 7, MAX_GOALS, dtype=torch.bool)
    vg[:, 0, :K] = True
    vg[:, 1, 2] = True  # deliver only goal 2 (the held block)
    va = torch.zeros(n, 7, MAX_GOALS, 2, dtype=torch.bool)
    va[:, 0, :K, :] = True
    va[:, 1, 2, 1] = True  # ... with the back arm only
    va[:, 5, 0, :] = True
    b.update(valid_opt=vo, valid_goal=vg, valid_arm=va)
    return b


def test_decode_respects_masks():
    net = SwitcherNet().eval()
    b = _batch(256)
    with torch.no_grad():
        out, prog = net.decide(*(b[k] for k in ("blk", "blk_mask", "goal", "goal_mask", "arm", "glob", "valid_opt",
                                                "valid_goal", "valid_arm")))
    o, g, a = out[:, 0], out[:, 1], out[:, 2]
    assert b["valid_opt"][torch.arange(256), o].all()
    for i in range(256):
        opt = OPTIONS[int(o[i])]
        if opt == "idle":
            assert g[i] == -1 and a[i] == -1
        elif opt == "park":
            assert g[i] == -1 and a[i] >= 0
        else:
            assert b["valid_goal"][i, o[i], g[i]] and b["valid_arm"][i, o[i], g[i], a[i]]
            if opt == "deliver":
                assert g[i] == 2 and a[i] == 1
    assert prog.shape == (256, MAX_GOALS) and ((prog >= 0) & (prog <= 1)).all()


def test_export_round_trip_and_feature_check(tmp_path):
    net = SwitcherNet().eval()
    p = export(net, tmp_path / "sw.pt", {"note": "test"})
    dec = SwitcherDecider.load(p)
    b = _batch(32, 1)
    keys = ("blk", "blk_mask", "goal", "goal_mask", "arm", "glob", "valid_opt", "valid_goal", "valid_arm")
    with torch.no_grad():
        a, pa = net.decide(*(b[k] for k in keys))
        c, pc = dec.mod(*(b[k] for k in keys))
    assert torch.equal(a, c) and torch.allclose(pa, pc, atol=1e-6)
    meta = json.loads((tmp_path / "sw.pt.json").read_text())
    meta["features"] = "v0"
    (tmp_path / "sw.pt.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        SwitcherDecider.load(p)


def test_loss_trains_every_head():
    torch.manual_seed(0)
    b = _batch(64, 2)
    n = 64
    b["kind"] = torch.zeros(n, dtype=torch.long)
    b["kind"][48:] = 1  # progress-only rows
    b["label_opt"] = torch.where(torch.arange(n) % 3 == 0, 5, 0)
    b["label_opt"][1::7] = 1
    b["label_goal"] = torch.where(b["label_opt"] == 0, torch.arange(n) % 6, torch.full((n,), -1))
    b["label_goal"][b["label_opt"] == 1] = 2
    b["label_arm"] = torch.where(b["label_opt"] == 1, 1, torch.arange(n) % 2)
    b["label_opt"][48:] = -1
    b["label_goal"][48:] = -1
    b["label_arm"][48:] = -1
    b["progress"] = torch.full((n, MAX_GOALS), -1, dtype=torch.long)
    b["progress"][:, :6] = (torch.arange(n)[:, None] + torch.arange(6)[None]) % 2
    b["nonnominal"] = (torch.arange(n) % 4 == 0).long()
    stats = Normalizer.fit({k: b[k].numpy() for k in ("blk", "blk_mask", "goal", "goal_mask", "arm", "glob")})
    net = SwitcherNet(stats=stats)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    first = None
    for _ in range(150):
        loss, st = loss_fn(net, b)
        first = first if first is not None else st
        opt.zero_grad()
        loss.backward()
        opt.step()
    for k in ("opt", "goal", "arm", "progress"):
        assert st[k] < 0.3 * first[k] + 1e-3, (k, first[k], st[k])
