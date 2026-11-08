"""Polarized-target magnet: Oxford 5 T split-pair coils of the UVA/JLab NH3 target.

The coil package (14 rectangular-section windings, two inverse-wired correction coils)
is taken from camsonne/g2psim src/oxfordfield.f (B. Niczyporuk 1998, used for g2p/GEp/SANE).
The field is computed by Biot-Savart for coaxial circular current sheets (elliptic
integrals, Gauss-Legendre quadrature over each winding cross section), tabulated on an
(r, z) grid in the coil frame and scaled so that |B(0)| = B0.

The magnet body that stops recoil protons is modelled by the windings plus a margin
(default 5 mm, standing for formers, cryostat walls and heat shields).  The real body is
larger than the windings: the clear openings must be checked against the magnet drawings.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.special import ellipe, ellipk

MU0 = 4e-7 * np.pi

# Oxford coil geometry [cm] (oxfordfield.f)
_RI = [6.478, 6.478, 10.963, 10.963, 11.648, 11.648, 14.451, 14.451, 18.936, 18.936,
       19.559, 19.559, 22.424, 22.424]
_RA = [8.487, 8.487, 11.648, 11.648, 13.877, 13.877, 17.465, 17.465, 19.510, 19.510,
       21.908, 21.908, 25.430, 25.430]
_ZA = [6.042, -9.047, 6.967, -13.935, 6.967, -13.935, 8.709, -14.930, 10.949, -16.920,
       10.949, -16.920, 12.690, -16.323]
_ZB = [9.047, -6.042, 13.935, -6.967, 13.935, -6.967, 14.930, -8.709, 16.920, -10.949,
       16.920, -10.949, 16.323, -12.690]
_TURNS = [1118, 1118, 616, 616, 2003, 2003, 3515, 3515, 638, 638, 3534, 3534, 3853, 3853]
_SIGN = [-1, -1] + [1] * 12  # first pair inverse wired


@dataclass
class Coil:
    ri: float
    ra: float
    za: float
    zb: float
    amp_turns: float  # signed


def oxford_coils(current=106.0958):
    return [Coil(_RI[i] / 100, _RA[i] / 100, _ZA[i] / 100, _ZB[i] / 100,
                 _SIGN[i] * _TURNS[i] * current) for i in range(14)]


def loop_field(a, z0, I, r, z):
    """Field (Br, Bz) [T] of a circular loop radius a [m] at z0 carrying I [A]."""
    r = np.asarray(r, float)
    z = np.asarray(z, float)
    dz = z - z0
    rr = np.maximum(r, 1e-9)
    den = (a + rr) ** 2 + dz**2
    m = 4 * a * rr / den
    m = np.minimum(m, 1 - 1e-12)
    K = ellipk(m)
    E = ellipe(m)
    q = (a - rr) ** 2 + dz**2
    q = np.maximum(q, 1e-12)
    c = MU0 * I / (2 * np.pi * np.sqrt(den))
    Bz = c * (K + (a**2 - rr**2 - dz**2) / q * E)
    Br = c * dz / rr * (-K + (a**2 + rr**2 + dz**2) / q * E)
    Br = np.where(r < 1e-9, 0.0, Br)
    return Br, Bz


def coil_field(coils, r, z, nq=6):
    xg, wg = np.polynomial.legendre.leggauss(nq)
    Br = np.zeros_like(np.asarray(r, float))
    Bz = np.zeros_like(Br)
    for c in coils:
        rs = 0.5 * (c.ra + c.ri) + 0.5 * (c.ra - c.ri) * xg
        zs = 0.5 * (c.zb + c.za) + 0.5 * (c.zb - c.za) * xg
        for i in range(nq):
            for j in range(nq):
                I = c.amp_turns * wg[i] * wg[j] / 4.0
                br, bz = loop_field(rs[i], zs[j], I, r, z)
                Br += br
                Bz += bz
    return Br, Bz


class FieldMap:
    """Axisymmetric field map in the coil frame, placed in the lab with a given axis."""

    def __init__(self, B0=5.0, axis=(1.0, 0.0, 0.0), rmax=1.2, zmax=1.2, step=0.004,
                 cache_dir=None):
        self.B0 = B0
        self.axis = np.asarray(axis, float) / np.linalg.norm(axis)
        self.coils = oxford_coils()
        self.step = step
        self.rg = np.arange(0.0, rmax + step / 2, step)
        self.zg = np.arange(0.0, zmax + step / 2, step)
        cache = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache = os.path.join(cache_dir, f"oxford_map_{rmax}_{zmax}_{step}.npz")
        if cache and os.path.exists(cache):
            d = np.load(cache)
            self.Br, self.Bz = d["Br"], d["Bz"]
        else:
            R, Z = np.meshgrid(self.rg, self.zg, indexing="ij")
            self.Br, self.Bz = coil_field(self.coils, R, Z)
            if cache:
                np.savez_compressed(cache, Br=self.Br, Bz=self.Bz)
        b00 = self.Bz[0, 0]
        self.scale = B0 / b00 if B0 != 0 else 0.0
        self.Br = self.Br * self.scale
        self.Bz = self.Bz * self.scale

    # coil-frame bilinear interpolation (z mirrored: Br odd, Bz even)
    def field_rz(self, r, z):
        az = np.abs(z)
        s = self.step
        ir = np.clip((r / s).astype(int), 0, len(self.rg) - 2)
        iz = np.clip((az / s).astype(int), 0, len(self.zg) - 2)
        fr = np.clip(r / s - ir, 0, 1)
        fz = np.clip(az / s - iz, 0, 1)

        def bil(A):
            return ((1 - fr) * (1 - fz) * A[ir, iz] + fr * (1 - fz) * A[ir + 1, iz]
                    + (1 - fr) * fz * A[ir, iz + 1] + fr * fz * A[ir + 1, iz + 1])

        out = (r > self.rg[-1]) | (az > self.zg[-1])
        Br = np.where(out, 0.0, bil(self.Br) * np.sign(z))
        Bz = np.where(out, 0.0, bil(self.Bz))
        return Br, Bz

    def __call__(self, pos):
        """Lab field [T] at positions pos (N,3) [m]."""
        if self.scale == 0.0:
            return np.zeros_like(pos)
        zc = pos @ self.axis
        rvec = pos - zc[:, None] * self.axis
        r = np.linalg.norm(rvec, axis=1)
        Br, Bz = self.field_rz(r, zc)
        rhat = rvec / np.maximum(r, 1e-12)[:, None]
        return Bz[:, None] * self.axis + Br[:, None] * rhat


class MagnetBody:
    """Proton-stopping magnet body: windings grown by `margin` [m], with the radial gaps
    between the winding blocks of one half filled (formers/structure).

    In the coil frame (r, |z|) a point is inside the body if it lies between the gap face
    (lower envelope of the windings) and the back face, and between bore and outer radius.
    """

    def __init__(self, axis=(1.0, 0.0, 0.0), margin=0.005, enabled=True):
        self.axis = np.asarray(axis, float) / np.linalg.norm(axis)
        self.enabled = enabled
        coils = [c for c in oxford_coils() if c.za > 0]
        m = margin
        # piecewise description: for each block (r range) the |z| range
        self.blocks = [(c.ri - m, c.ra + m, c.za - m, c.zb + m) for c in coils]
        # fill the radial gaps between consecutive blocks with the overlapping z range
        blocks = sorted(self.blocks)
        fill = []
        for b1, b2 in zip(blocks[:-1], blocks[1:]):
            if b2[0] > b1[1]:
                fill.append((b1[1], b2[0], max(b1[2], b2[2]), min(b1[3], b2[3])))
        self.blocks += fill

    def inside(self, pos):
        if not self.enabled:
            return np.zeros(len(pos), bool)
        zc = np.abs(pos @ self.axis)
        r = np.linalg.norm(pos - np.outer(pos @ self.axis, self.axis), axis=1)
        ins = np.zeros(len(pos), bool)
        for r0, r1, z0, z1 in self.blocks:
            ins |= (r >= r0) & (r <= r1) & (zc >= z0) & (zc <= z1)
        return ins

    def straight_line_clear(self, origin, direction, rmax=0.30, step=0.002):
        """True for rays (neutral particles) that do not cross the magnet body."""
        d = direction / np.linalg.norm(direction, axis=1)[:, None]
        clear = np.ones(len(d), bool)
        for s in np.arange(step, rmax, step):
            clear &= ~self.inside(origin + s * d)
        return clear

    def opening_angles(self):
        """Half-angles [deg] seen from the centre: split gap (from the mid-plane) and axial
        cone (from the axis), from the block corners."""
        gap = min(np.degrees(np.arctan2(z0, r1)) for r0, r1, z0, z1 in self.blocks)
        cone = min(np.degrees(np.arctan2(r0, z1)) for r0, r1, z0, z1 in self.blocks)
        return gap, cone
