"""The station light (one RectLight per env) under the robustness case "lighting": intensity scale, colour temperature,
rotation about the vertical through the table centre (azimuth) and tilt about the light's own centre. USD edits on a
running stage (needs Isaac); renders converge over more than one 4-subframe capture after a change."""

from __future__ import annotations

import math

import numpy as np
from pxr import Gf, UsdGeom

LIGHT = "/World/envs/env_{}/Station/Model/StationModel/StationLighting/TopWhitePlaneLight"
TABLE_CENTRE = (0.30, -0.40, 0.0)  # between the two pads (station frame, m)


class StationLights:
    def __init__(self, stage, origins: np.ndarray):
        self.prims = [stage.GetPrimAtPath(LIGHT.format(e)) for e in range(len(origins))]
        assert all(p.IsValid() for p in self.prims), "station light prim not found"
        self.origins = np.asarray(origins)
        self.base = [(UsdGeom.Xformable(p).GetLocalTransformation(), p.GetAttribute("inputs:intensity").Get(),
                      p.GetAttribute("inputs:colorTemperature").Get()) for p in self.prims]

    def set(self, e: int, scale: float = 1.0, kelvin: float | None = None, azimuth_deg: float = 0.0, tilt_deg: float = 0.0,
            tilt_dir_deg: float = 0.0) -> dict:
        p = self.prims[e]
        local0, inten0, kelvin0 = self.base[e]
        xf = UsdGeom.Xformable(p)
        parent = xf.ComputeParentToWorldTransform(0)
        world0 = local0 * parent
        c_light = world0.ExtractTranslation()
        c_table = Gf.Vec3d(*(np.asarray(TABLE_CENTRE) + self.origins[e]).tolist())
        tdir = math.radians(tilt_dir_deg)
        tilt = (Gf.Matrix4d().SetTranslate(-c_light) * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(math.cos(tdir), math.sin(tdir), 0), tilt_deg))
                * Gf.Matrix4d().SetTranslate(c_light))
        az = (Gf.Matrix4d().SetTranslate(-c_table) * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), azimuth_deg))
              * Gf.Matrix4d().SetTranslate(c_table))
        xf.MakeMatrixXform().Set(world0 * tilt * az * parent.GetInverse())
        p.GetAttribute("inputs:intensity").Set(float(inten0 * scale))
        p.GetAttribute("inputs:colorTemperature").Set(float(kelvin if kelvin is not None else kelvin0))
        return {"scale": round(scale, 3), "kelvin": round(float(kelvin if kelvin is not None else kelvin0)), "azimuth_deg": round(azimuth_deg, 1),
                "tilt_deg": round(tilt_deg, 1), "tilt_dir_deg": round(tilt_dir_deg, 1)}

    def apply(self, dom) -> list[dict]:
        """Every env's light from a ``tower_rl.domain.DomainSample`` (nan colour temperature = nominal)."""
        out = []
        for e in range(len(self.prims)):
            k = float(dom.color_temp_k[e])
            out.append(self.set(e, scale=float(dom.light_scale[e]), kelvin=None if math.isnan(k) else k,
                                azimuth_deg=math.degrees(float(dom.light_azimuth[e]))))
        return out
