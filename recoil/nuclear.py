"""Bound nucleons in 14N and 4He: Fermi motion and removal energy.

Momentum distribution: Fermi gas with k_F (Moniz et al. 1971: 221 MeV/c for C/N; 4He is
taken as a Gaussian-like gas with k_F = 180 MeV/c) plus a 1/k^4 high-momentum tail carrying
a fraction `tail` of the strength between k_F and 0.8 GeV/c (short-range correlations,
~20% for medium nuclei).  The struck nucleon is off-shell with energy
E = M - E_rem - p^2/(2 M_{A-1}).  Final-state interactions are not modelled.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import M_P


@dataclass
class Nucleus:
    name: str
    A: int
    Z: int
    kF: float
    E_rem: float
    tail: float

    def sample_momentum(self, rng, n):
        p = self.kF * rng.random(n) ** (1.0 / 3.0)
        is_tail = rng.random(n) < self.tail
        # 1/k^4 between kF and kmax  (dN/dk ~ k^2/k^4 = 1/k^2)
        kmax = 0.8
        u = rng.random(n)
        ptail = 1.0 / (1.0 / self.kF - u * (1.0 / self.kF - 1.0 / kmax))
        p = np.where(is_tail, ptail, p)
        c = 2 * rng.random(n) - 1
        a = 2 * np.pi * rng.random(n)
        s = np.sqrt(1 - c**2)
        return p[:, None] * np.column_stack([s * np.cos(a), s * np.sin(a), c])

    def initial_nucleon(self, rng, n):
        pv = self.sample_momentum(rng, n)
        p2 = np.sum(pv**2, axis=1)
        MA1 = (self.A - 1) * 0.9315
        E = M_P - self.E_rem - p2 / (2 * MA1)
        return np.column_stack([E, pv])


N14 = Nucleus("14N", 14, 7, 0.221, 0.020, 0.20)
HE4 = Nucleus("4He", 4, 2, 0.180, 0.020, 0.15)


def target_composition(packing_fraction=0.60, rho_nh3=0.817, rho_he=0.145):
    """Number of nucleons of each kind per unit volume (relative to free protons)."""
    n_nh3 = packing_fraction * rho_nh3 / 17.031
    n_he = (1 - packing_fraction) * rho_he / 4.0026
    free_p = 3 * n_nh3
    return {
        "free_p": 1.0,
        "N14_p": 7 * n_nh3 / free_p,
        "N14_n": 7 * n_nh3 / free_p,
        "He_p": 2 * n_he / free_p,
        "He_n": 2 * n_he / free_p,
    }
