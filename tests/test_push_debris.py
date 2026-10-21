"""Debris poses of pushed-off blocks (push skill env resets): on the table, clear of the foundation, flat or on a side."""
import math

import torch

from tower_rl import geometry as geo
from tower_rl import push as P
from tower_rl.config import load_yaml


def _db():
    db = dict(load_yaml("skills/push")["reset"]["debris"])
    db["frac"] = 1.0
    return db


def test_debris_clear_of_the_foundation_and_on_the_table():
    db = _db()
    n = 2000
    g = torch.Generator().manual_seed(0)
    # blocks of a layer along x (rows at y = -0.05, 0, 0.05 around a tower axis at the origin), pushed along +x
    p0 = torch.zeros(n, 3)
    p0[:, 1] = (torch.randint(3, (n,), generator=g) - 1).float() * 0.05
    p0[:, 2] = 0.0731
    q0 = geo.quat_from_yaw(torch.zeros(n))
    d = torch.tensor([[1.0, 0.0, 0.0]]).expand(n, 3)
    u = torch.rand(n, 4, generator=g)
    p, q = P.debris_poses(p0, q0, d, u, db)
    assert torch.allclose(q.norm(dim=-1), torch.ones(n), atol=1e-5)
    side = u[:, 3] < db["side_frac"]
    assert torch.allclose(p[side, 2], torch.full_like(p[side, 2], db["z_side_m"]))
    assert torch.allclose(p[~side, 2], torch.full_like(p[~side, 2], db["z_flat_m"]))
    # nearest point of the block's footprint to the tower (half extents 0.075 x 0.025, yawed) beyond the foundation face
    yaw = geo.yaw_from_quat(q)
    reach = 0.075 * torch.cos(yaw).abs() + 0.025 * torch.sin(yaw).abs()
    assert bool((p[:, 0] - reach > 0.075).all()), float((p[:, 0] - reach).min())
    # on its side: the long axis stays horizontal, the 50 mm width points up
    up = geo.quat_apply(q[side], torch.tensor([[0.0, 1.0, 0.0]]).expand(int(side.sum()), 3))
    assert bool((up[:, 2].abs() > math.cos(math.radians(1))).all())
    ax = geo.quat_apply(q, torch.tensor([[1.0, 0.0, 0.0]]).expand(n, 3))
    assert bool((ax[:, 2].abs() < 1e-5).all())
