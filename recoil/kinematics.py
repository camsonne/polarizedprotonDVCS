"""DVCS kinematics: e N -> e' gamma N'.

Events are generated with the scattered electron inside an HMS-like acceptance and the
gamma* N -> gamma N' two-body final state in its centre-of-mass frame, flat in t and in the
azimuth phi (Trento convention: angle between leptonic and hadronic planes).  The initial
nucleon can be off-shell and moving (bound nucleons, see nuclear.py).

Four-vectors are arrays (..., 4) ordered (E, px, py, pz).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import DEG, M_P


def mdot(a, b):
    return a[..., 0] * b[..., 0] - np.sum(a[..., 1:] * b[..., 1:], axis=-1)


def boost(v, beta):
    """Boost four-vectors v by velocity beta (N,3) (active: rest frame -> lab)."""
    b2 = np.sum(beta**2, axis=-1)
    g = 1.0 / np.sqrt(1.0 - b2)
    bp = np.sum(beta * v[..., 1:], axis=-1)
    g2 = np.where(b2 > 0, (g - 1.0) / np.maximum(b2, 1e-300), 0.0)
    E = g * (v[..., 0] + bp)
    p = v[..., 1:] + (g2 * bp + g * v[..., 0])[..., None] * beta
    return np.concatenate([E[..., None], p], axis=-1)


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


@dataclass
class Setting:
    name: str
    E: float           # beam energy [GeV]
    xB: float
    Q2: float
    nps_distance: float = 4.0     # m
    hms_dp: float = 0.09          # +- momentum bite
    hms_dth: float = 0.028        # +- in-plane angle [rad]
    hms_dph: float = 0.070        # +- out-of-plane angle [rad]

    @property
    def nu(self):
        return self.Q2 / (2 * M_P * self.xB)

    @property
    def Eprime(self):
        return self.E - self.nu

    @property
    def theta_e(self):
        s2 = self.Q2 / (4 * self.E * self.Eprime)
        return 2 * np.arcsin(np.sqrt(s2))

    @property
    def theta_q(self):
        q = np.sqrt(self.nu**2 + self.Q2)
        return np.arcsin(self.Eprime * np.sin(self.theta_e) / q)

    def describe(self):
        return (f"{self.name}: E={self.E} GeV xB={self.xB} Q2={self.Q2} GeV2  "
                f"E'={self.Eprime:.2f} GeV th_e={self.theta_e / DEG:.1f} deg "
                f"th_q={self.theta_q / DEG:.1f} deg")


# Representative kinematics inspired by the Hall C NPS DVCS programme (10.6 GeV).
SETTINGS = [
    Setting("KinA", 10.6, 0.36, 3.0),
    Setting("KinB", 10.6, 0.36, 4.0),
    Setting("KinC", 10.6, 0.50, 3.4),
    Setting("KinD", 10.6, 0.50, 4.8),
    Setting("KinE", 10.6, 0.60, 5.1),
]


def sample_electron(s: Setting, rng, n):
    """Scattered electron in the HMS acceptance; HMS on beam right (-x)."""
    P0 = s.Eprime
    p = P0 * (1 + s.hms_dp * (2 * rng.random(n) - 1))
    dth = s.hms_dth * (2 * rng.random(n) - 1)
    dph = s.hms_dph * (2 * rng.random(n) - 1)
    th0 = s.theta_e
    # direction: central ray in the x-z plane at -th0, small in-plane / out-of-plane tilts
    tx = np.tan(-th0 - dth)
    d = np.column_stack([np.sin(-th0 - dth), np.tan(dph) * np.cos(-th0 - dth),
                         np.cos(-th0 - dth)])
    d = unit(d)
    del tx
    return np.column_stack([p, p[:, None] * d])


def dvcs_two_body(k, kp, Pi, t_range=(None, 1.2), rng=None, t=None, phi=None):
    """gamma*(q) N(Pi) -> gamma N'.  Returns (gamma, recoil, t, phi, valid).

    t is sampled flat between t_min and t_min - t_range[1] unless given.
    """
    n = len(kp)
    q = k - kp
    Ptot = q + Pi
    s = mdot(Ptot, Ptot)
    W = np.sqrt(np.maximum(s, 1e-9))
    beta = Ptot[:, 1:] / Ptot[:, 0:1]
    qs = boost(q, -beta)
    Pis = boost(Pi, -beta)
    qabs = np.linalg.norm(qs[:, 1:], axis=1)
    Mi2 = mdot(Pi, Pi)
    # final real photon energy in CM (final nucleon on-shell M_P)
    kcm = (s - M_P**2) / (2 * W)
    Q2 = -mdot(q, q)
    # t = (q - k')^2 = -Q2 - 2 kcm (q0* - |q*| cos)
    tmin = -Q2 - 2 * kcm * (qs[:, 0] - qabs)
    tmax = -Q2 - 2 * kcm * (qs[:, 0] + qabs)
    if t is None:
        span = np.minimum(t_range[1], tmin - tmax)
        t = tmin - span * rng.random(n)
    cos = (t + Q2 + 2 * kcm * qs[:, 0]) / (2 * kcm * qabs)
    valid = (np.abs(cos) <= 1) & (s > (M_P + 0.0) ** 2)
    cos = np.clip(cos, -1, 1)
    sin = np.sqrt(1 - cos**2)
    if phi is None:
        phi = 2 * np.pi * rng.random(n)
    # CM frame axes: z* along q*, x* in the lepton plane (pointing away from k'), y* = z* x x*
    ez = unit(qs[:, 1:])
    ks = boost(k, -beta)
    xs = ks[:, 1:] - np.sum(ks[:, 1:] * ez, axis=1)[:, None] * ez
    ex = unit(xs)
    ey = np.cross(ez, ex)
    # Trento phi: photon azimuth; recoil goes opposite (phi + pi)
    dg = cos[:, None] * ez + sin[:, None] * (np.cos(phi)[:, None] * ex + np.sin(phi)[:, None] * ey)
    gam_s = np.column_stack([kcm, kcm[:, None] * dg])
    rec_s = np.column_stack([W - kcm, -kcm[:, None] * dg])
    gam = boost(gam_s, beta)
    rec = boost(rec_s, beta)
    del Mi2, Pis
    return gam, rec, t, phi, valid


def generate_free(s: Setting, rng, n, t_span=1.2):
    k = np.tile([s.E, 0, 0, s.E], (n, 1)).astype(float)
    kp = sample_electron(s, rng, n)
    P = np.tile([M_P, 0, 0, 0], (n, 1)).astype(float)
    gam, rec, t, phi, valid = dvcs_two_body(k, kp, P, (None, t_span), rng)
    return dict(k=k, kp=kp, P=P, gam=gam, rec=rec, t=t, phi=phi, valid=valid)


def invariants(k, kp, P=None):
    q = k - kp
    Q2 = -mdot(q, q)
    if P is None:
        P = np.tile([M_P, 0, 0, 0], (len(k), 1))
    xB = Q2 / (2 * mdot(P, q))
    return Q2, xB


def predicted_recoil(k, kp, gdir, P=None):
    """Recoil four-momentum predicted from e' and the photon *direction* only, assuming
    e p -> e' gamma p on a free proton at rest (photon energy from exclusivity)."""
    n = len(k)
    if P is None:
        P = np.tile([M_P, 0, 0, 0], (n, 1)).astype(float)
    q = k - kp
    tot = q + P
    s = mdot(tot, tot)
    gd = unit(gdir)
    Eg = (s - M_P**2) / (2 * (tot[:, 0] - np.sum(tot[:, 1:] * gd, axis=1)))
    gam = np.column_stack([Eg, Eg[:, None] * gd])
    return tot - gam, gam


def missing_mass2(k, kp, gam, P=None):
    n = len(k)
    if P is None:
        P = np.tile([M_P, 0, 0, 0], (n, 1)).astype(float)
    X = k + P - kp - gam
    return mdot(X, X)


def t_from_recoil(rec):
    """t from the recoil four-momentum alone (target at rest)."""
    P = np.zeros_like(rec)
    P[:, 0] = M_P
    d = P - rec
    return mdot(d, d)


def trento_phi(k, kp, gam):
    """Trento phi from lab four-vectors (angle between lepton and hadron planes)."""
    q = (k - kp)[:, 1:]
    kv = k[:, 1:]
    kpv = kp[:, 1:]
    g = gam[:, 1:]
    nl = unit(np.cross(kv, kpv))
    nh = unit(np.cross(q, g))
    qh = unit(q)
    c = np.sum(nl * nh, axis=1)
    sgn = np.sign(np.sum(np.cross(nl, nh) * qh, axis=1))
    sgn = np.where(sgn == 0, 1, sgn)
    ph = sgn * np.arccos(np.clip(c, -1, 1))
    return np.mod(ph, 2 * np.pi)


def polar_azimuth(v3):
    v = unit(v3)
    return np.arccos(np.clip(v[:, 2], -1, 1)), np.arctan2(v[:, 1], v[:, 0])


__all__ = ["Setting", "SETTINGS", "generate_free", "dvcs_two_body", "predicted_recoil",
           "missing_mass2", "t_from_recoil", "trento_phi", "polar_azimuth", "mdot", "boost",
           "unit", "invariants", "sample_electron", "DEG"]
