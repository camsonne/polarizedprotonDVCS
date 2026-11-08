"""Materials, proton stopping power, range tables, multiple scattering.

Stopping power is the Bethe formula without shell or density corrections.  For protons
between 5 MeV and 1 GeV kinetic energy this agrees with NIST PSTAR to a few percent
(checked in tests/test_materials.py), which is enough for a design study; below ~2 MeV
the protons are about to stop anyway.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .constants import M_P, M_E

K_BETHE = 0.307075  # MeV cm^2 / mol


@dataclass
class Material:
    name: str
    rho: float      # g/cm^3
    ZA: float       # <Z/A>
    I: float        # mean excitation energy [eV]
    X0: float       # radiation length [g/cm^2]
    _rt: tuple = field(default=None, repr=False)

    # ------------------------------------------------------------------
    def dedx(self, T):
        """Mass stopping power for protons [MeV cm^2/g]; T kinetic energy [GeV]."""
        T = np.maximum(np.asarray(T, dtype=float), 2e-4)
        g = 1.0 + T / M_P
        b2 = 1.0 - 1.0 / g**2
        bg2 = b2 * g**2
        me = M_E * 1e3  # MeV
        mp = M_P * 1e3
        tmax = 2 * me * bg2 / (1 + 2 * g * me / mp + (me / mp) ** 2)
        I = self.I * 1e-6
        arg = 2 * me * bg2 * tmax / I**2
        L = 0.5 * np.log(arg) - b2
        return K_BETHE * self.ZA / b2 * np.maximum(L, 0.05)

    def dedx_lin(self, T):
        """Linear stopping power [GeV/m]."""
        return self.dedx(T) * self.rho * 1e-3 * 100.0

    # ------------------------------------------------------------------
    def _range_table(self):
        if self._rt is None:
            Tg = np.geomspace(2e-4, 3.0, 1200)  # GeV
            S = self.dedx(Tg) * 1e-3  # GeV cm^2/g
            # cumulative trapezoid of dT/S -> g/cm^2
            inv = 1.0 / S
            R = np.concatenate([[Tg[0] * inv[0] * 0.5],
                                np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(Tg))])
            R[1:] += R[0]
            self._rt = (Tg, R)
        return self._rt

    def range_gcm2(self, T):
        Tg, R = self._range_table()
        return np.interp(T, Tg, R)

    def T_from_range(self, R):
        Tg, Rg = self._range_table()
        return np.interp(R, Rg, Tg, left=0.0)

    def T_after(self, T, path_m, density_scale=1.0):
        """Kinetic energy after a path [m] (mean continuous loss via range table)."""
        x = path_m * 100.0 * self.rho * density_scale  # g/cm^2
        Rleft = self.range_gcm2(T) - x
        Tn = self.T_from_range(Rleft)
        return np.where(Rleft > 0, Tn, 0.0)

    def x_over_X0(self, path_m, density_scale=1.0):
        return path_m * 100.0 * self.rho * density_scale / self.X0

    def straggling_sigma(self, T, path_m, density_scale=1.0):
        """Bohr straggling width [GeV] (Gaussian approximation)."""
        g = 1.0 + np.asarray(T) / M_P
        b2 = np.maximum(1.0 - 1.0 / g**2, 1e-6)
        x = path_m * 100.0 * self.rho * density_scale
        s2 = 0.1569 * self.ZA * x * (1 - b2 / 2) / (1 - b2)  # MeV^2
        return np.sqrt(s2) * 1e-3


def mixture(name, comps, rho):
    """Mass-fraction mixture.  comps = [(Material, mass_fraction), ...]."""
    wsum = sum(w for _, w in comps)
    ZA = sum(m.ZA * w for m, w in comps) / wsum
    lnI = sum(m.ZA * w * math.log(m.I) for m, w in comps) / (ZA * wsum)
    invX0 = sum(w / m.X0 for m, w in comps) / wsum
    return Material(name, rho, ZA, math.exp(lnI), 1.0 / invX0)


def highland_theta0(p, beta, xX0, log_ref=None):
    """Highland multiple-scattering width [rad] for unit charge.

    For a sequence of thin steps the per-step log term underestimates the total; pass
    log_ref (x/X0 of the whole layer stack the particle sees) to use a fixed correction.
    """
    xX0 = np.maximum(xX0, 0.0)
    ref = xX0 if log_ref is None else log_ref
    corr = 1.0 + 0.038 * np.log(np.maximum(ref / np.maximum(beta, 1e-3) ** 2, 1e-12))
    corr = np.clip(corr, 0.5, 2.0)
    return 0.0136 / (np.maximum(beta, 1e-3) * p) * np.sqrt(xX0) * corr


# ---------------------------------------------------------------------------
# Material table.  Numbers follow the PDG tables and, for the target, the g2p
# simulation (camsonne/g2psim src/G2PTarget.cc) which modelled the same UVA/JLab
# polarized NH3 target.
NH3 = Material("NH3", 0.817, 10.0 / 17.0305, 53.7, 40.87)
LHE = Material("LHe", 0.145, 2.0 / 4.0026, 41.8, 94.32)
AL = Material("Al", 2.699, 13.0 / 26.9815, 166.0, 24.01)
PCTFE = Material("PCTFE", 2.10, 0.4808, 120.7, 28.19)        # Kel-F target cup
POLYSTYRENE = Material("Polystyrene", 1.05, 0.5377, 68.7, 43.79)  # fibre core
AIR = Material("Air", 1.205e-3, 0.4992, 85.7, 36.62)
CFRP = Material("CFRP", 1.60, 0.50, 78.0, 42.7)               # carbon-fibre skins
ROHACELL = Material("Rohacell", 0.052, 0.54, 70.0, 41.0)
KAPTON = Material("Kapton", 1.42, 0.5126, 79.6, 40.58)
SILICON = Material("Si", 2.329, 0.4985, 173.0, 21.82)


def target_material(packing_fraction=0.60):
    """NH3 beads in LHe at the given volume packing fraction."""
    pf = packing_fraction
    m_nh3 = pf * NH3.rho
    m_he = (1 - pf) * LHE.rho
    rho = m_nh3 + m_he
    return mixture(f"NH3/LHe pf={pf:.2f}", [(NH3, m_nh3), (LHE, m_he)], rho)


MATERIALS = {m.name: m for m in [NH3, LHE, AL, PCTFE, POLYSTYRENE, AIR, CFRP,
                                 ROHACELL, KAPTON, SILICON]}
