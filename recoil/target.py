"""Polarized NH3 target and scattering-chamber geometry (material seen by the recoil).

Dimensions follow the UVA/JLab target as modelled for g2p (camsonne/g2psim
src/G2PTarget.cc): 2.83 cm long Kel-F (PCTFE) cup of radius 1.36 cm along the beam, inside a
vertical LHe-filled nose (r = 2.10 cm, 4 mil Al wall), 4 K shield (r = 3.81 cm, 1.5 mil Al),
LN2 shield (r = 41.9 cm, 1.5 mil Al) and the scattering-chamber exit window (r = 47.94 cm,
20 mil Al).  All of these are vertical cylinders (axis = y) except the cup (axis = z).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .materials import AL, AIR, LHE, PCTFE, target_material


@dataclass
class Shell:
    """Thin cylindrical shell. axis 'y' (vertical) or 'z' (beam)."""
    name: str
    radius: float
    thickness: float
    material: object
    axis: str = "y"
    half_length: float = 10.0  # along the axis
    is_detector: bool = False
    view: str = ""          # detector view (X, Y, U, V)
    station: int = -1
    phi_range: tuple = None  # (min, max) horizontal angle [rad] for detector coverage
    y_range: tuple = None
    hole: tuple = None       # (alpha half-width [rad], y half-height [m]) left open for the beam


@dataclass
class TargetConfig:
    packing_fraction: float = 0.60
    cup_radius: float = 0.0136144
    cup_length: float = 0.0282702
    cup_wall: float = 0.000889
    nose_radius: float = 0.0210058
    nose_wall: float = 0.0001016
    shield4k_radius: float = 0.0381
    shield4k_wall: float = 0.0000381
    ln2_radius: float = 0.4191
    ln2_wall: float = 0.0000381
    chamber_radius: float = 0.479425
    chamber_window: float = 0.000508
    raster_radius: float = 0.010   # beam raster radius on target


class Target:
    def __init__(self, cfg: TargetConfig = None):
        self.cfg = cfg or TargetConfig()
        c = self.cfg
        self.mix = target_material(c.packing_fraction)
        self.shells = [
            Shell("cup_wall", c.cup_radius, c.cup_wall, PCTFE, axis="z", half_length=c.cup_length / 2),
            Shell("nose_wall", c.nose_radius, c.nose_wall, AL),
            Shell("shield_4K", c.shield4k_radius, c.shield4k_wall, AL),
            Shell("shield_LN2", c.ln2_radius, c.ln2_wall, AL),
            Shell("chamber_window", c.chamber_radius, c.chamber_window, AL),
        ]

    def continuous_material(self, pos):
        """Material index per point: 0 vacuum, 1 NH3/LHe mix, 2 LHe, 3 air."""
        c = self.cfg
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        rxy = np.hypot(x, y)
        rxz = np.hypot(x, z)
        in_cup = (rxy < c.cup_radius) & (np.abs(z) < c.cup_length / 2)
        in_nose = (rxz < c.nose_radius) & (np.abs(y) < 0.30)
        air = rxz > c.chamber_radius + c.chamber_window
        mat = np.zeros(len(pos), int)
        mat[in_nose] = 2
        mat[in_cup] = 1
        mat[air] = 3
        return mat

    @property
    def continuous_materials(self):
        return {1: self.mix, 2: LHE, 3: AIR}

    def sample_vertex(self, rng, n):
        c = self.cfg
        r = c.raster_radius * np.sqrt(rng.random(n))
        a = 2 * np.pi * rng.random(n)
        z = (rng.random(n) - 0.5) * c.cup_length
        return np.column_stack([r * np.cos(a), r * np.sin(a), z])
