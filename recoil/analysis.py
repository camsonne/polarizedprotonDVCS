"""Summary metrics from `study.simulate` outputs."""
from __future__ import annotations

import numpy as np

from .constants import M_P, M_PI0
from .kinematics import unit
from .study import angles
from .transport import EXITED, LOST_MAGNET, STOPPED

MM2_WINDOW = (M_P**2 - 0.25, (M_P + M_PI0) ** 2)
T_SLOPE = 2.5  # GeV^-2, weight exp(B t) for integrated numbers


def weights(o, t_slope=T_SLOPE):
    t = o["ev"]["t"]
    return np.exp(t_slope * t)


def robust_sigma(x):
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan
    q = np.percentile(x, [16, 84])
    return 0.5 * (q[1] - q[0])


def breakdown(o, w=None):
    """Fractions of arm-accepted events by fate of the recoil."""
    st = o["status"]
    n = len(st)
    if w is None:
        w = np.ones(n)
    any_hit = np.isfinite(o["meas"]).any(axis=1)
    cats = {
        "stopped": st == STOPPED,
        "hit_magnet": st == LOST_MAGNET,
        "missed_tracker": (st == EXITED) & ~any_hit,
        "incomplete": (st == EXITED) & any_hit & ~o["tracked"],
        "tracked": o["tracked"],
        "fit_ok": o["fit_ok"],
    }
    W = w.sum()
    return {k: float((w * v).sum() / W) for k, v in cats.items()}


def view_groups(cfg):
    views = cfg.view_list()
    X = [j for j, (_, v, _) in enumerate(views) if v == "X"]
    Y = [j for j, (_, v, _) in enumerate(views) if v == "Y"]
    return X, Y


def match_variables(o, cfg):
    X, Y = view_groups(cfg)
    with np.errstate(all="ignore"):
        ds = np.nanmean(o["match_res"][:, X], axis=1)
        dy = np.nanmean(o["match_res"][:, Y], axis=1)
    return ds, dy, o["match_dt"]


def p_pred(o):
    return np.linalg.norm(o["pred"][:, 1:], axis=1)


class MatchCalibration:
    """sigma(p) = sqrt(a^2 + (b/(p beta))^2) for ds, dy, dt from the free-proton sample."""

    def __init__(self, o, cfg, bins=np.array([0.2, 0.35, 0.45, 0.55, 0.7, 0.9, 1.2, 2.0])):
        ds, dy, dt = match_variables(o, cfg)
        pp = p_pred(o)
        ok = o["tracked"] & np.isfinite(ds) & np.isfinite(dy)
        self.pars = {}
        for name, v in (("ds", ds), ("dy", dy), ("dt", dt)):
            xs, ys = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = ok & (pp >= lo) & (pp < hi)
                if m.sum() > 30:
                    pc = 0.5 * (lo + hi)
                    xs.append(1.0 / (pc * pc / np.sqrt(pc**2 + M_P**2)))
                    ys.append(robust_sigma(v[m]))
            xs, ys = np.array(xs), np.array(ys)
            if len(xs) >= 2:
                A = np.column_stack([np.ones_like(xs), xs**2])
                sol, *_ = np.linalg.lstsq(A, ys**2, rcond=None)
                a2, b2 = max(sol[0], 1e-10), max(sol[1], 1e-10)
            else:
                a2, b2 = robust_sigma(v[ok]) ** 2, 0.0
            self.pars[name] = (np.sqrt(a2), np.sqrt(b2))
            self.points = getattr(self, "points", {})
            self.points[name] = (xs, ys)

    def sigma(self, name, p):
        a, b = self.pars[name]
        beta = p / np.sqrt(p**2 + M_P**2)
        return np.sqrt(a**2 + (b / (p * beta)) ** 2)

    def pulls(self, o, cfg):
        ds, dy, dt = match_variables(o, cfg)
        pp = np.clip(p_pred(o), 0.2, 3.0)
        return ds / self.sigma("ds", pp), dy / self.sigma("dy", pp), dt / self.sigma("dt", pp)


def selection(o, cfg, cal, nsig=3.0, use_time=True, mm2=MM2_WINDOW):
    """Boolean masks: MM2 cut only, and MM2 + recoil matching."""
    mmc = (o["mm2"] > mm2[0]) & (o["mm2"] < mm2[1])
    ps, py, pt = cal.pulls(o, cfg)
    r2 = ps**2 + py**2
    rec = o["tracked"] & np.isfinite(r2) & (r2 < nsig**2)
    if use_time:
        rec &= np.isfinite(pt) & (np.abs(pt) < nsig)
    return mmc, mmc & rec


def fit_resolution(o, pbins=np.array([0.25, 0.4, 0.55, 0.7, 0.9, 1.3])):
    ok = o["fit_ok"]
    rt = o["ev"]["rec"][:, 1:]
    rf = o["rec_fit"][:, 1:]
    pt = np.linalg.norm(rt, axis=1)
    pf = np.linalg.norm(rf, axis=1)
    tht, pht = angles(rt)
    thf, phf = angles(rf)
    dp = (pf - pt) / pt
    dth = thf - tht
    dph = np.angle(np.exp(1j * (phf - pht))) * np.sin(tht)
    rows = []
    for lo, hi in zip(pbins[:-1], pbins[1:]):
        m = ok & (pt >= lo) & (pt < hi)
        rows.append(dict(p_lo=lo, p_hi=hi, n=int(m.sum()), dp_rel=robust_sigma(dp[m]),
                         dth_mrad=1e3 * robust_sigma(dth[m]), dph_mrad=1e3 * robust_sigma(dph[m])))
    tot = dict(dp_rel=robust_sigma(dp[ok]), dth_mrad=1e3 * robust_sigma(dth[ok]),
               dph_mrad=1e3 * robust_sigma(dph[ok]))
    # t from the recoil alone
    E = o["rec_fit"][:, 0]
    t_rec = -2 * M_P * (E - M_P)
    dt = (t_rec - o["ev"]["t"]) / np.abs(o["ev"]["t"])
    tot["dt_rel"] = robust_sigma(dt[ok])
    return rows, tot


def dilution(eff, comp, r_n=0.3):
    """Effective dilution factor from per-species efficiencies.

    eff: dict with keys free, N14, He -> (eps_mm2, eps_mm2_recoil)
    comp: target_composition();  r_n: neutron/proton DVCS+BH yield ratio.
    """
    fr, fr_r = eff["free"]
    n_, n_r = eff["N14"]
    h_, h_r = eff["He"]
    den_raw = fr + comp["N14_p"] * n_ + comp["N14_n"] * r_n * n_ + comp["He_p"] * h_ \
        + comp["He_n"] * r_n * h_
    den_rec = fr_r + comp["N14_p"] * n_r + comp["He_p"] * h_r
    f_raw = fr / den_raw
    f_rec = fr_r / den_rec if den_rec > 0 else np.nan
    # statistical figure of merit for a target asymmetry: f^2 N, N in free-proton units
    fom_raw = f_raw**2 * den_raw
    fom_rec = f_rec**2 * den_rec
    return dict(f_raw=f_raw, f_rec=f_rec, fom_raw=fom_raw, fom_rec=fom_rec,
                fom_ratio=fom_rec / fom_raw)
