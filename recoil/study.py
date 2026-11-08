"""End-to-end fast simulation of the recoil detector for one kinematic setting.

  generate e N -> e' gamma N' (free proton, or bound p in 14N / 4He)
  -> electron in HMS box, photon on the NPS face, both clear of the magnet body
  -> transport the recoil (field, target, chamber, tracker), digitise the fibres
  -> fit the recoil track; predict the recoil from e' + photon direction
  -> exclusivity variables (missing mass, recoil angle and momentum differences)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import DEG, M_P
from .detector import SciFiConfig, digitize
from .kinematics import (Setting, dvcs_two_body, missing_mass2, predicted_recoil,
                         sample_electron, t_from_recoil, unit)
from .magnet import FieldMap, MagnetBody
from .materials import POLYSTYRENE
from .reco import RecoilFitter, p3_to_params, params_to_p3
from .target import Target, TargetConfig
from .transport import EXITED, Transporter


@dataclass
class ArmResolution:
    hms_dp: float = 1.0e-3        # dp/p
    hms_dth: float = 1.5e-3       # rad (target field included in the optics)
    nps_stoch: float = 0.025      # sigma_E/E = a/sqrt(E) (+) b
    nps_const: float = 0.010
    nps_pos: float = 0.003        # m on the face
    z_vertex: float = 0.005       # m, electron-arm vertex resolution
    raster_xy: float = 0.0005     # m, raster position resolution


@dataclass
class Scenario:
    name: str = "A_5T_horizontal"
    B0: float = 5.0
    axis: tuple = (1.0, 0.0, 0.0)  # field axis (horizontal, perpendicular to the beam)
    magnet_margin: float = 0.005
    detector: SciFiConfig = field(default_factory=SciFiConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    arms: ArmResolution = field(default_factory=ArmResolution)
    nps_cols: int = 30
    nps_rows: int = 36
    nps_block: float = 0.0205
    min_stations: int = 2          # stations with a space point (>= 2 views) required
    fit: bool = True               # run the stand-alone fit (needs X and Y views)


_FIELD_CACHE = {}


def build(scn: Scenario, cache_dir="cache"):
    key = (scn.B0, tuple(scn.axis))
    if key not in _FIELD_CACHE:
        _FIELD_CACHE[key] = FieldMap(B0=scn.B0, axis=scn.axis, cache_dir=cache_dir)
    fmap = _FIELD_CACHE[key]
    body = MagnetBody(axis=scn.axis, margin=scn.magnet_margin)
    tgt = Target(scn.target)
    tr = Transporter(fmap, tgt, body, scn.detector.shells())
    return fmap, body, tgt, tr


def nps_frame(s: Setting):
    th = s.theta_q
    c = np.array([np.sin(th), 0.0, np.cos(th)])
    ex = np.array([np.cos(th), 0.0, -np.sin(th)])
    ey = np.array([0.0, 1.0, 0.0])
    return c, ex, ey


def nps_hit(s: Setting, scn: Scenario, vtx, gdir):
    c, ex, ey = nps_frame(s)
    D = s.nps_distance
    den = gdir @ c
    lam = (D - vtx @ c) / np.maximum(den, 1e-9)
    hit = vtx + lam[:, None] * gdir
    u = (hit - D * c) @ ex
    v = (hit - D * c) @ ey
    hu = (scn.nps_cols / 2 - 1) * scn.nps_block
    hv = (scn.nps_rows / 2 - 1) * scn.nps_block
    acc = (den > 0) & (np.abs(u) < hu) & (np.abs(v) < hv)
    return acc, u, v


def generate(s: Setting, rng, n, nucleus=None, t_span=1.2):
    k = np.tile([s.E, 0, 0, s.E], (n, 1)).astype(float)
    kp = sample_electron(s, rng, n)
    if nucleus is None:
        P = np.tile([M_P, 0, 0, 0], (n, 1)).astype(float)
    else:
        P = nucleus.initial_nucleon(rng, n)
    gam, rec, t, phi, valid = dvcs_two_body(k, kp, P, (None, t_span), rng)
    return dict(k=k, kp=kp, P=P, gam=gam, rec=rec, t=t, phi=phi, valid=valid)


def smear_arms(s: Setting, scn: Scenario, rng, ev, vtx, nps_u, nps_v):
    a = scn.arms
    n = len(vtx)
    kp = ev["kp"]
    p = np.linalg.norm(kp[:, 1:], axis=1)
    d = unit(kp[:, 1:])
    pm = p * (1 + a.hms_dp * rng.standard_normal(n))
    e1 = unit(np.cross(d, [0, 1, 0]))
    e2 = np.cross(d, e1)
    dm = unit(d + a.hms_dth * (rng.standard_normal((n, 1)) * e1 + rng.standard_normal((n, 1)) * e2))
    kp_m = np.column_stack([pm, pm[:, None] * dm])
    # photon: energy and position on the NPS face; direction from the measured vertex
    Eg = ev["gam"][:, 0]
    sig = Eg * np.hypot(a.nps_stoch / np.sqrt(Eg), a.nps_const)
    Eg_m = Eg + sig * rng.standard_normal(n)
    c, ex, ey = nps_frame(s)
    D = s.nps_distance
    hit = D * c + (nps_u + a.nps_pos * rng.standard_normal(n))[:, None] * ex \
        + (nps_v + a.nps_pos * rng.standard_normal(n))[:, None] * ey
    vtx_m = vtx.copy()
    vtx_m[:, 2] += a.z_vertex * rng.standard_normal(n)
    vtx_m[:, :2] += a.raster_xy * rng.standard_normal((n, 2))
    gdir_m = unit(hit - vtx_m)
    gam_m = np.column_stack([Eg_m, Eg_m[:, None] * gdir_m])
    return kp_m, gam_m, vtx_m


def simulate(s: Setting, scn: Scenario, n, rng, nucleus=None, fitter=None, do_fit=True,
             built=None):
    fmap, body, tgt, tr = built or build(scn)
    ev = generate(s, rng, n, nucleus)
    vtx = tgt.sample_vertex(rng, n)
    gdir = unit(ev["gam"][:, 1:])
    edir = unit(ev["kp"][:, 1:])
    acc_g, u, v = nps_hit(s, scn, vtx, gdir)
    clear_g = body.straight_line_clear(vtx, gdir)
    clear_e = body.straight_line_clear(vtx, edir)
    arms = ev["valid"] & acc_g & clear_g & clear_e
    sel = np.where(arms)[0]
    wgen = np.where(ev["valid"], np.exp(2.5 * ev["t"]), 0.0)
    out = dict(n_gen=n, n_arms=len(sel), n_valid=int(ev["valid"].sum()),
               wsum_gen=float(wgen.sum()),
               n_nps=int((ev["valid"] & acc_g).sum()),
               n_clear=int((ev["valid"] & acc_g & clear_g).sum()))
    ev = {k_: (v_[sel] if isinstance(v_, np.ndarray) else v_) for k_, v_ in ev.items()}
    vtx, u, v = vtx[sel], u[sel], v[sel]
    out["ev"] = ev
    out["vtx"] = vtx
    # recoil transport + digitisation
    res = tr.run(vtx, ev["rec"][:, 1:], rng)
    meas, npe, tmeas = digitize(scn.detector, res, rng, POLYSTYRENE.dedx)
    out.update(status=res.status, det_T=res.det_T, det_pos=res.det_pos, meas=meas, npe=npe,
               tmeas=tmeas, T_after_target=res.T_after_target)
    views = scn.detector.view_list()
    nst = len(scn.detector.stations)
    st_ok = np.zeros((len(sel), nst), bool)
    for si in range(nst):
        js = [j for j, (s_, v_, _) in enumerate(views) if s_ == si]
        fired = np.isfinite(meas[:, js])
        # a station gives a space point with >= 2 fired views (X/Y, or X/U/V stereo)
        st_ok[:, si] = fired.sum(axis=1) >= 2
    trk = (st_ok.sum(axis=1) >= scn.min_stations) & st_ok[:, 0] & st_ok[:, -1]
    out["tracked"] = trk
    # measured electron arm
    kp_m, gam_m, vtx_m = smear_arms(s, scn, rng, ev, vtx, u, v)
    out.update(kp_m=kp_m, gam_m=gam_m, vtx_m=vtx_m)
    pred, _ = predicted_recoil(ev["k"], kp_m, gam_m[:, 1:])
    out["pred"] = pred
    out["mm2"] = missing_mass2(ev["k"], kp_m, gam_m)
    # kinematic matching: propagate the predicted recoil to the tracker and compare
    out["match_res"] = np.full(meas.shape, np.nan)
    out["match_dt"] = np.full(len(sel), np.nan)
    if np.any(trk):
        it = np.where(trk)[0]
        pr = pred[it, 1:]
        okp = np.isfinite(pr).all(axis=1) & (np.linalg.norm(pr, axis=1) > 0.15)
        it = it[okp]
        rp = tr.run(vtx_m[it], pred[it, 1:], physics=False)
        from .detector import view_coordinate
        wpred = np.column_stack([view_coordinate(rp.det_pos[:, j], v_, scn.detector.stereo)
                                 for j, (_, v_, _) in enumerate(views)])
        out["match_res"][it] = meas[it] - wpred
        with np.errstate(all="ignore"):
            out["match_dt"][it] = (np.nanmean(tmeas[it], axis=1)
                                   - np.nanmean(rp.det_time, axis=1))
    out["fit_ok"] = np.zeros(len(sel), bool)
    out["rec_fit"] = np.full((len(sel), 4), np.nan)
    if do_fit and scn.fit and np.any(trk):
        fitter = fitter or make_fitter(tr, scn, rng)
        it = np.where(trk)[0]
        wm = meas[it]
        sig_t = np.full(len(it), np.hypot(scn.detector.sigma_t_view /
                                          np.sqrt(np.isfinite(tmeas[it]).sum(axis=1).clip(1)),
                                          fitter.st0))
        with np.errstate(all="ignore"):
            tm = np.nanmean(tmeas[it], axis=1) + fitter.st0 * rng.standard_normal(len(it))
        xy = vtx_m[it, :2]
        zm = vtx_m[it, 2]
        seed, _ = fitter.seed(wm, xy, zm, tm)
        par, chi2, ndf, cov = fitter.fit(wm, tm, sig_t, xy, zm, seed)
        p3 = params_to_p3(par)
        E = np.sqrt(np.sum(p3**2, axis=1) + M_P**2)
        out["rec_fit"][it] = np.column_stack([E, p3])
        out["chi2"] = np.full(len(sel), np.nan)
        out["chi2"][it] = chi2
        out["ndf"] = np.zeros(len(sel))
        out["ndf"][it] = ndf
        out["zfit"] = np.full(len(sel), np.nan)
        out["zfit"][it] = par[:, 3]
        good = np.isfinite(chi2) & (chi2 / np.maximum(ndf, 1) < 10)
        out["fit_ok"][it] = good
    return out


def make_fitter(tr, scn, rng, n_table=60000):
    f = RecoilFitter(tr, scn.detector, sigma_z_prior=scn.arms.z_vertex)
    f.build_seed_table(rng, n_table)
    return f


def angles(v3):
    v = unit(v3)
    th = np.arccos(np.clip(v[:, 2], -1, 1))
    ph = np.arctan2(v[:, 1], v[:, 0])
    return th, ph


def exclusivity(out):
    """Differences between the fitted recoil and the recoil predicted from e' + gamma."""
    rf = out["rec_fit"]
    pr = out["pred"]
    pf = np.linalg.norm(rf[:, 1:], axis=1)
    pp = np.linalg.norm(pr[:, 1:], axis=1)
    thf, phf = angles(rf[:, 1:])
    thp, php = angles(pr[:, 1:])
    # angle between the two directions, decomposed: polar and azimuthal (sin-weighted)
    dth = thf - thp
    dph = np.angle(np.exp(1j * (phf - php))) * np.sin(thp)
    cosang = np.sum(unit(rf[:, 1:]) * unit(pr[:, 1:]), axis=1)
    return dict(dp=pf - pp, dp_rel=(pf - pp) / pp, dth=dth, dph=dph,
                dang=np.arccos(np.clip(cosang, -1, 1)))
