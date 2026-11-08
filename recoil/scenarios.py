"""Detector / magnet scenarios compared in the study."""
from __future__ import annotations

from .constants import DEG
from .detector import SciFiConfig, StationConfig
from .study import Scenario


def A5():
    """Baseline: 3 SciFi stations outside the scattering chamber (R = 52, 57, 62 cm),
    1 mm fibres, X/Y/U views, 5 T horizontal transverse field."""
    return Scenario(name="A5")


def A5_2st():
    det = SciFiConfig(name="A 2 stations",
                      stations=[StationConfig(0.52), StationConfig(0.62)])
    return Scenario(name="A5_2st", detector=det)


def A5_d2():
    det = SciFiConfig(name="A 2mm fibres", fibre_d=0.002)
    return Scenario(name="A5_d2", detector=det)


def A25():
    return Scenario(name="A25", B0=2.5)


def B5():
    """In-vacuum stations between the magnet and the LN2 shield (R = 30, 34, 38 cm)."""
    det = SciFiConfig(name="B in vacuum",
                      stations=[StationConfig(0.30), StationConfig(0.34), StationConfig(0.38)],
                      y_range=(-0.40 * 0.30 / 0.52, 0.65 * 0.30 / 0.52))
    return Scenario(name="B5", detector=det)


def A5_m0():
    return Scenario(name="A5_m0", magnet_margin=0.0)


def A5_m10():
    return Scenario(name="A5_m10", magnet_margin=0.010)


def A5_vert():
    """Same coils with a vertical field axis (would need a different magnet/cryostat)."""
    return Scenario(name="A5_vert", axis=(0.0, 1.0, 0.0))


def A5_tall():
    """Baseline with a taller tracker (y from -0.60 to +1.00 m at R = 52 cm)."""
    det = SciFiConfig(name="A tall", y_range=(-0.60, 1.00))
    return Scenario(name="A5_tall", detector=det)


def C5():
    """Inner barrel inside the magnet split gap, in the target vacuum between the 4 K shield
    (r = 3.8 cm) and the coil faces (|x| > 6 cm): two stations at r = 4.3 and 5.1 cm,
    each X (vertical) + U/V (+-20 deg stereo) of 0.5 mm fibres.  All fibres run
    roughly vertically, so they can be read out above the magnet."""
    det = SciFiConfig(name="C inner barrel",
                      stations=[StationConfig(0.043, views=("X", "U", "V"), view_gap=0.0015),
                                StationConfig(0.051, views=("X", "U", "V"), view_gap=0.0015)],
                      fibre_d=0.0005, stereo=20 * DEG, alpha_range=(-75 * DEG, 75 * DEG),
                      y_range=(-0.10, 0.12), beam_hole=14 * DEG,
                      support_cfrp=0.0002, support_foam=0.0)
    return Scenario(name="C5", detector=det, fit=False)


def C5_z1():
    """C5 with a 1 mm vertex-z resolution from the electron arm (instead of 5 mm)."""
    scn = C5()
    scn.name = "C5_z1"
    scn.arms.z_vertex = 0.001
    return scn


SCENARIOS = {f.__name__: f for f in [A5, A5_2st, A5_d2, A25, B5, A5_m0, A5_m10, A5_vert,
                                     A5_tall, C5, C5_z1]}

DESCRIPTIONS = {
    "A5": "baseline: 3 stations at R=52/57/62 cm outside the chamber, 1 mm fibres, 5 T",
    "A5_2st": "2 stations (R=52, 62 cm)",
    "A5_d2": "2 mm fibres",
    "A25": "2.5 T field",
    "B5": "3 stations in vacuum at R=30/34/38 cm",
    "A5_m0": "magnet body = bare windings (optimistic)",
    "A5_m10": "magnet body = windings + 10 mm (pessimistic)",
    "A5_vert": "field axis vertical (different magnet)",
    "A5_tall": "taller tracker: y from -60 to +100 cm at R=52 cm",
    "C5": "inner barrel in the magnet gap, r=4.3/5.1 cm, 0.5 mm fibres X/U/V, read out above the magnet",
    "C5_z1": "C5 with 1 mm vertex-z resolution",
}

ALL_SETTINGS = ["KinA", "KinB", "KinC", "KinD", "KinE"]


def run_plan():
    """List of (scenario, [(setting, species), ...]) tasks."""
    plan = []
    for s in ALL_SETTINGS:
        plan.append(("A5", [(s, nuc) for nuc in ("free", "N14", "He")]))
    for scn in SCENARIOS:
        if scn == "A5":
            continue
        plan.append((scn, [(s, nuc) for s in ("KinA", "KinD") for nuc in ("free", "N14", "He")]))
    return plan
