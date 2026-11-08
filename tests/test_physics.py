"""Unit tests of the physics building blocks.  Run: python3 -m pytest tests"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recoil.constants import M_P  # noqa: E402
from recoil.kinematics import (SETTINGS, generate_free, mdot, predicted_recoil,  # noqa: E402
                               trento_phi)
from recoil.magnet import MU0, FieldMap, MagnetBody, coil_field, loop_field, oxford_coils  # noqa: E402
from recoil.materials import Material  # noqa: E402

WATER = Material("Water", 1.0, 0.5551, 75.0, 36.08)
CACHE = os.path.join(os.path.dirname(__file__), "..", "cache")


@pytest.mark.parametrize("T,R_pstar", [(0.050, 2.227), (0.100, 7.718), (0.200, 25.96)])
def test_proton_range_water(T, R_pstar):
    """CSDA range against NIST PSTAR (g/cm^2)."""
    assert WATER.range_gcm2(T) == pytest.approx(R_pstar, rel=0.04)


def test_stopping_power_water():
    assert WATER.dedx(0.100) == pytest.approx(7.289, rel=0.03)  # PSTAR, MeV cm2/g


def test_loop_on_axis():
    a, I, z = 0.1, 1000.0, 0.07
    Br, Bz = loop_field(a, 0.0, I, np.array([0.0]), np.array([z]))
    assert Bz[0] == pytest.approx(MU0 * I * a**2 / (2 * (a**2 + z**2) ** 1.5), rel=1e-6)
    assert Br[0] == 0.0


def test_oxford_central_field():
    """The coil table at the nominal current gives 5.0 T at the centre (oxfordfield.f)."""
    _, Bz = coil_field(oxford_coils(), np.array([0.0]), np.array([0.0]))
    assert Bz[0] == pytest.approx(5.0, abs=0.01)


def test_fieldmap_orientation_and_symmetry():
    fm = FieldMap(B0=5.0, axis=(1, 0, 0), cache_dir=CACHE)
    B = fm(np.array([[0.0, 0, 0], [0.03, 0.02, 0.05], [-0.03, 0.02, 0.05]]))
    assert B[0] == pytest.approx([5.0, 0, 0], abs=1e-6)
    # Bx even, transverse components odd under x -> -x
    assert B[1, 0] == pytest.approx(B[2, 0], rel=1e-6)
    assert B[1, 2] == pytest.approx(-B[2, 2], rel=1e-6, abs=1e-9)


def test_magnet_openings():
    gap, cone = MagnetBody(margin=0.0).opening_angles()
    assert 24 < gap < 30 and 30 < cone < 40


@pytest.mark.parametrize("s", SETTINGS)
def test_dvcs_kinematics(s):
    rng = np.random.default_rng(1)
    ev = generate_free(s, rng, 2000)
    v = ev["valid"]
    tot = ev["k"] + ev["P"] - ev["kp"] - ev["gam"] - ev["rec"]
    assert np.abs(tot[v]).max() < 1e-10
    assert np.allclose(mdot(ev["rec"], ev["rec"])[v], M_P**2, atol=1e-9)
    assert np.allclose(mdot(ev["gam"], ev["gam"])[v], 0, atol=1e-9)
    d = ev["P"] - ev["rec"]
    assert np.allclose(mdot(d, d)[v], ev["t"][v], atol=1e-9)
    dphi = np.angle(np.exp(1j * (trento_phi(ev["k"], ev["kp"], ev["gam"]) - ev["phi"])))
    assert np.abs(dphi[v]).max() < 1e-8
    pr, _ = predicted_recoil(ev["k"], ev["kp"], ev["gam"][:, 1:])
    assert np.abs(pr - ev["rec"])[v].max() < 1e-9
