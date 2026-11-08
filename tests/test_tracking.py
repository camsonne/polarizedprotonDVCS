"""Transport and track-fit tests."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recoil.constants import M_P  # noqa: E402
from recoil.detector import view_coordinate  # noqa: E402
from recoil.reco import RecoilFitter, p3_to_params  # noqa: E402
from recoil.study import Scenario, build  # noqa: E402
from recoil.transport import EXITED, STOPPED  # noqa: E402

CACHE = os.path.join(os.path.dirname(__file__), "..", "cache")


def _p3(p, alpha, beta):
    """momentum p toward horizontal angle alpha and vertical angle beta."""
    return p * np.array([[np.sin(alpha) * np.cos(beta), np.sin(beta), np.cos(alpha) * np.cos(beta)]])


def test_straight_line_without_field():
    scn = Scenario(B0=0.0)
    _, _, _, tr = build(scn, cache_dir=CACHE)
    res = tr.run(np.zeros((1, 3)), _p3(0.8, 0.2, 0.3), physics=False)
    assert res.status[0] == EXITED
    pos = res.det_pos[0, 0]
    R = scn.detector.stations[0].radius
    # straight line: y / rho = tan(beta) where rho = sqrt(x^2 + z^2) = R
    assert pos[1] / R == pytest.approx(np.tan(0.3), rel=1e-3)
    assert np.arctan2(pos[0], pos[2]) == pytest.approx(0.2, abs=1e-4)


def test_upward_bending_and_energy_loss():
    scn = Scenario()
    _, _, _, tr = build(scn, cache_dir=CACHE)
    res0 = tr.run(np.zeros((1, 3)), _p3(0.5, 0.0, 0.25), physics=False)
    assert res0.status[0] == EXITED
    y = res0.det_pos[0, 0, 1]
    R = scn.detector.stations[0].radius
    assert y > R * np.tan(0.25) + 0.02  # positive charge, B along +x: bends up
    T0 = np.hypot(0.5, M_P) - M_P
    assert 0.0 < T0 - res0.det_T[0, 0] < 0.02  # a few MeV lost in target and windows


def test_slow_proton_stops():
    scn = Scenario()
    _, _, _, tr = build(scn, cache_dir=CACHE)
    res = tr.run(np.zeros((1, 3)), _p3(0.12, 0.0, 0.3), physics=False)  # T = 7.6 MeV
    assert res.status[0] == STOPPED


def test_fit_recovers_noiseless_track():
    scn = Scenario()
    _, _, _, tr = build(scn, cache_dir=CACHE)
    rng = np.random.default_rng(5)
    f = RecoilFitter(tr, scn.detector)
    f.build_seed_table(rng, 20000)
    p3 = np.vstack([_p3(0.45, 0.1, 0.35), _p3(0.7, -0.15, 0.4), _p3(0.9, 0.3, 0.6)])
    vtx = np.zeros((3, 3))
    res = tr.run(vtx, p3, physics=False)
    wm = np.column_stack([view_coordinate(res.det_pos[:, j], v, scn.detector.stereo)
                          for j, (_, v, _) in enumerate(scn.detector.view_list())])
    tm = np.nanmean(res.det_time, axis=1)
    seed, _ = f.seed(wm, vtx[:, :2], vtx[:, 2], tm)
    par, chi2, ndf, _ = f.fit(wm, tm, np.full(3, 0.1), vtx[:, :2], vtx[:, 2], seed)
    truth = p3_to_params(p3, np.zeros(3))
    assert np.all(chi2 < 1.0)
    assert np.allclose(par[:, 0], truth[:, 0], rtol=0.02)
    assert np.allclose(par[:, 1:3], truth[:, 1:3], atol=0.01)
