#!/usr/bin/env python3
"""Make figures, tables (results/RESULTS.md) and results/summary.json from results/data/."""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recoil import analysis as A  # noqa: E402
from recoil.constants import DEG, M_P  # noqa: E402
from recoil.kinematics import SETTINGS  # noqa: E402
from recoil.magnet import FieldMap, MagnetBody, oxford_coils  # noqa: E402
from recoil.nuclear import target_composition  # noqa: E402
from recoil.scenarios import DESCRIPTIONS, SCENARIOS  # noqa: E402
from recoil.study import angles, build, generate, nps_hit, simulate  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "results", "data")
FIG = os.path.join(ROOT, "results", "figs")
os.makedirs(FIG, exist_ok=True)
COL = {"free": "#1f77b4", "N14": "#d62728", "He": "#2ca02c"}
LBL = {"free": "free p (H)", "N14": "p bound in $^{14}$N", "He": "p bound in $^4$He"}


def load(scn, s, nuc):
    fn = os.path.join(DATA, f"{scn}__{s}__{nuc}.npz")
    if not os.path.exists(fn):
        return None
    d = dict(np.load(fn, allow_pickle=True))
    ev = {k[3:]: v for k, v in d.items() if k.startswith("ev_")}
    o = {k: (v.item() if v.ndim == 0 else v) for k, v in d.items() if not k.startswith("ev_")}
    o["ev"] = ev
    return o


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_field():
    fm = FieldMap(B0=5.0, cache_dir=os.path.join(ROOT, "cache"))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    s = np.linspace(0, 0.8, 400)
    for lab, d in (("along beam (z)", [0, 0, 1]), ("along field axis (x)", [1, 0, 0]),
                   ("vertical (y)", [0, 1, 0]), ("recoil direction alpha=30deg, 30deg up",
                                                 [np.sin(30 * DEG) * np.cos(30 * DEG),
                                                  np.sin(30 * DEG),
                                                  np.cos(30 * DEG) * np.cos(30 * DEG)])):
        pts = np.outer(s, d)
        B = fm(pts)
        ax[0].plot(s * 100, np.linalg.norm(B, axis=1), label=lab)
    ax[0].axvline(47.9, color="k", ls=":", lw=1)
    ax[0].text(48.5, 3.5, "chamber\nwindow", fontsize=8)
    ax[0].set_xlabel("distance from target centre [cm]")
    ax[0].set_ylabel("|B| [T]")
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-3, 8)
    ax[0].legend(fontsize=8)
    ax[0].set_title("Oxford split pair, 5 T, axis horizontal, perpendicular to beam", fontsize=10)
    # map in the horizontal plane (x-z), coil blocks drawn
    xs = np.linspace(-0.5, 0.5, 251)
    zs = np.linspace(-0.5, 0.7, 301)
    X, Z = np.meshgrid(xs, zs)
    pts = np.column_stack([X.ravel(), np.zeros(X.size), Z.ravel()])
    B = np.linalg.norm(fm(pts), axis=1).reshape(X.shape)
    im = ax[1].pcolormesh(Z * 100, X * 100, np.log10(np.maximum(B, 1e-3)), shading="auto",
                          cmap="viridis", vmin=-2.5, vmax=0.8)
    plt.colorbar(im, ax=ax[1], label="log10 |B| [T]")
    draw_coils(ax[1], plane="top")
    ax[1].set_xlabel("z (beam) [cm]")
    ax[1].set_ylabel("x (beam left) [cm]")
    ax[1].set_title("|B|, horizontal plane;\nwhite: magnet body (windings + 5 mm)")
    ax[1].set_aspect("equal")
    save(fig, "fig_field.png")


def draw_coils(ax, plane="top", color="w", margin=0.005):
    """Coil blocks (axis = lab x).  top view: (z, x); in the side view (z, y) at x=0 the
    coils are not cut (they sit at |x| > 6 cm) and are drawn hatched as their projection."""
    body = MagnetBody(margin=margin)
    for r0, r1, z0, z1 in body.blocks:
        for sx in (1, -1):
            for sr in (1, -1):
                if plane == "top":
                    rect = plt.Rectangle((sr * r0 * 100 if sr > 0 else -r1 * 100, sx * z0 * 100 if sx > 0 else -z1 * 100),
                                         (r1 - r0) * 100, (z1 - z0) * 100, fc="none", ec=color, lw=0.8)
                    ax.add_patch(rect)


def fig_geometry():
    from recoil.scenarios import A5
    scn = A5()
    fm, body, tgt, tr = build(scn, cache_dir=os.path.join(ROOT, "cache"))
    rng = np.random.default_rng(7)
    s = SETTINGS[0]
    ev = generate(s, rng, 4000)
    vtx = tgt.sample_vertex(rng, 4000)
    gd = ev["gam"][:, 1:] / np.linalg.norm(ev["gam"][:, 1:], axis=1)[:, None]
    acc, _, _ = nps_hit(s, scn, vtx, gd)
    sel = np.where(acc & ev["valid"])[0][:120]
    res = tr.run(vtx[sel], ev["rec"][sel, 1:], rng, record=True)
    T = res.trajectory
    p = np.linalg.norm(ev["rec"][sel, 1:], axis=1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.5))
    cmap = plt.get_cmap("plasma")
    for i in range(len(sel)):
        c = cmap((p[i] - 0.25) / 1.0)
        ax[0].plot(T[:, i, 2] * 100, T[:, i, 0] * 100, color=c, lw=0.6, alpha=0.8)
        ax[1].plot(T[:, i, 2] * 100, T[:, i, 1] * 100, color=c, lw=0.6, alpha=0.8)
    draw_coils(ax[0], color="k")
    # chamber, shields, stations in the top view
    a = np.linspace(-np.pi, np.pi, 400)
    for R, ls, lab in ((0.4191, ":", "LN2 shield"), (0.4794, "-", "chamber")):
        ax[0].plot(R * 100 * np.cos(a), R * 100 * np.sin(a), "k", ls=ls, lw=0.8)
    a0, a1 = scn.detector.alpha_range
    aa = np.linspace(a0, a1, 200)
    for st in scn.detector.stations:
        ax[0].plot(st.radius * 100 * np.cos(aa), st.radius * 100 * np.sin(aa), "b", lw=2)
        y0, y1 = scn.detector.y_range
        f = st.radius / scn.detector.stations[0].radius
        ax[1].plot([st.radius * 100] * 2, [y0 * f * 100, y1 * f * 100], "b", lw=2)
    # spectrometer directions
    for th, lab, col in ((s.theta_q, "NPS (photon)", "g"), (-s.theta_e, "HMS (e')", "m")):
        ax[0].plot([0, 90 * np.cos(th)], [0, 90 * np.sin(th)], col + "--", lw=1.5)
        ax[0].text(90 * np.cos(th), 90 * np.sin(th), lab, color=col)
    ax[0].plot([-80, 90], [0, 0], "k-", lw=0.5)
    ax[0].set_xlim(-60, 95)
    ax[0].set_ylim(-70, 70)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("z (beam) [cm]")
    ax[0].set_ylabel("x (beam left) [cm]")
    ax[0].set_title("Top view: coil blocks (black), chamber, SciFi stations (blue)\n"
                    f"{s.name} recoil protons, 5 T along x; colour = p (0.25-1.25 GeV/c)")
    ax[1].set_xlim(-10, 95)
    ax[1].set_ylim(-60, 90)
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("z (beam) [cm]")
    ax[1].set_ylabel("y (up) [cm]")
    ax[1].set_title("Side view: protons bend upwards (B along +x)")
    save(fig, "fig_geometry.png")


def fig_kinematics():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for s in SETTINGS:
        o = load("A5", s.name, "free")
        if o is None:
            continue
        rec = o["ev"]["rec"][:, 1:]
        p = np.linalg.norm(rec, axis=1)
        th, _ = angles(rec)
        ax[0].scatter(th / DEG, p, s=1, alpha=0.3, label=s.name)
        tt = -o["ev"]["t"]
        h, e = np.histogram(tt, bins=np.linspace(0, 1.4, 29))
        ax[1].step(e[:-1], h / h.sum(), where="post", label=s.name)
    ax[0].set_xlabel("recoil polar angle [deg]")
    ax[0].set_ylabel("recoil momentum [GeV/c]")
    ax[0].set_title("Recoil protons of DVCS events with e' in HMS and photon in NPS")
    ax[0].legend(markerscale=8)
    ax[1].set_xlabel("-t [GeV$^2$] (flat generation, arm acceptance only)")
    ax[1].set_ylabel("fraction")
    ax[1].legend()
    save(fig, "fig_recoil_kinematics.png")


def fig_acceptance():
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    bins = np.linspace(0, 1.3, 14)
    c = 0.5 * (bins[1:] + bins[:-1])
    for s in SETTINGS:
        o = load("A5", s.name, "free")
        if o is None:
            continue
        tt = -o["ev"]["t"]
        h_all, _ = np.histogram(tt, bins)
        h_ok, _ = np.histogram(tt[o["fit_ok"]], bins)
        with np.errstate(all="ignore"):
            ax[0].plot(c, h_ok / h_all, "o-", ms=3, label=s.name)
    ax[0].set_xlabel("-t [GeV$^2$]")
    ax[0].set_ylabel("recoil efficiency (tracked and fitted)")
    ax[0].set_ylim(0, 1)
    ax[0].legend()
    ax[0].set_title("Baseline A5: recoil efficiency vs -t")
    # fate breakdown for KinA
    o = load("A5", "KinA", "free")
    if o is not None:
        tt = -o["ev"]["t"]
        st = o["status"]
        any_hit = np.isfinite(o["meas"]).any(axis=1)
        cats = [("stopped (target)", st == 2), ("hit magnet", st == 3),
                ("missed tracker", (st == 1) & ~any_hit),
                ("incomplete", (st == 1) & any_hit & ~o["tracked"]),
                ("tracked", o["tracked"])]
        h_all, _ = np.histogram(tt, bins)
        bottom = np.zeros(len(c))
        for lab, m in cats:
            h, _ = np.histogram(tt[m], bins)
            with np.errstate(all="ignore"):
                f = np.nan_to_num(h / h_all)
            ax[1].bar(c, f, width=bins[1] - bins[0], bottom=bottom, label=lab)
            bottom += f
        ax[1].set_xlabel("-t [GeV$^2$]")
        ax[1].set_title("KinA: fate of the recoil proton")
        ax[1].legend(fontsize=8)
        ph = o["ev"]["phi"] / DEG
        pb = np.linspace(0, 360, 19)
        pc = 0.5 * (pb[1:] + pb[:-1])
        for s_name in ("KinA", "KinD"):
            o2 = load("A5", s_name, "free")
            if o2 is None:
                continue
            ph = o2["ev"]["phi"] / DEG
            h_all, _ = np.histogram(ph, pb)
            h_ok, _ = np.histogram(ph[o2["fit_ok"]], pb)
            for lab, m in (("magnet", o2["status"] == 3),):
                h_m, _ = np.histogram(ph[m], pb)
                ax[2].plot(pc, h_m / h_all, "--", label=f"{s_name} lost in magnet")
            ax[2].plot(pc, h_ok / h_all, "o-", ms=3, label=f"{s_name} efficiency")
        ax[2].set_xlabel("Trento $\\phi$ [deg]")
        ax[2].set_ylim(0, 1)
        ax[2].legend(fontsize=8)
        ax[2].set_title("Efficiency vs $\\phi$: the coils shadow part of the azimuth")
    save(fig, "fig_acceptance.png")


def fig_resolution(cals):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    rows_all = []
    for s in SETTINGS:
        o = load("A5", s.name, "free")
        if o is None:
            continue
        rows, _ = A.fit_resolution(o)
        rows_all.append(rows)
    if rows_all:
        pc = np.array([0.5 * (r["p_lo"] + r["p_hi"]) for r in rows_all[0]])
        agg = {k: np.nanmedian([[r[k] for r in rows] for rows in rows_all], axis=0)
               for k in ("dp_rel", "dth_mrad", "dph_mrad")}
        ax[0].plot(pc, agg["dp_rel"] * 100, "o-", label="$\\sigma_p/p$ [%]")
        ax[0].plot(pc, agg["dth_mrad"], "s-", label="$\\sigma_\\theta$ [mrad]")
        ax[0].plot(pc, agg["dph_mrad"], "^-", label="$\\sin\\theta\\,\\sigma_\\phi$ [mrad]")
        ax[0].set_xlabel("recoil momentum [GeV/c]")
        ax[0].set_title("Stand-alone recoil fit (vertex + hits + time of flight)")
        ax[0].legend()
        ax[0].set_ylim(0, None)
    cal = cals.get(("A5", "KinA"))
    if cal is not None:
        p = np.linspace(0.25, 1.4, 100)
        for name, lab in (("ds", "$\\sigma(\\Delta s)$ [mm]"), ("dy", "$\\sigma(\\Delta y)$ [mm]"),
                          ("dt", "$\\sigma(\\Delta t)$ [100 ps]")):
            sc = 1e3 if name != "dt" else 10.0
            l, = ax[1].plot(p, cal.sigma(name, p) * sc, label=lab)
            xs, ys = cal.points[name]
            pts = []
            for x in xs:
                from scipy.optimize import brentq
                pts.append(brentq(lambda pp: 1 / (pp * pp / np.sqrt(pp**2 + M_P**2)) - x, 0.05, 5))
            ax[1].plot(pts, ys * sc, "o", color=l.get_color())
        ax[1].set_xlabel("predicted recoil momentum [GeV/c]")
        ax[1].set_title("Matching residuals: hits vs recoil predicted from e'+$\\gamma$ (KinA)")
        ax[1].legend()
        ax[1].set_ylim(0, None)
    save(fig, "fig_resolution.png")


def fig_matching(cals):
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    comp = target_composition()
    scale = {"free": 1.0, "N14": comp["N14_p"], "He": comp["He_p"]}
    cal = cals.get(("A5", "KinA"))
    cfg = SCENARIOS["A5"]().detector
    for nuc in ("free", "N14", "He"):
        o = load("A5", "KinA", nuc)
        if o is None or cal is None:
            continue
        w = A.weights(o) / o["wsum_gen"] * scale[nuc]
        ps, py, pt = cal.pulls(o, cfg)
        mmc = (o["mm2"] > A.MM2_WINDOW[0]) & (o["mm2"] < A.MM2_WINDOW[1])
        ax[0].hist(o["mm2"], bins=np.linspace(0, 2.5, 80), weights=w, histtype="step",
                   color=COL[nuc], label=LBL[nuc])
        for k, (v, rng_) in enumerate(((ps, 15), (py, 15), (pt, 15))):
            m = mmc & np.isfinite(v)
            ax[k + 1].hist(np.clip(v[m], -rng_, rng_), bins=np.linspace(-rng_, rng_, 90),
                           weights=w[m], histtype="step", color=COL[nuc], label=LBL[nuc])
    for a_ in A.MM2_WINDOW:
        ax[0].axvline(a_, color="k", ls=":")
    ax[0].set_xlabel("$M_X^2$(ep$\\to$e$\\gamma$X) [GeV$^2$]")
    ax[0].set_title("KinA, yields per free proton")
    ax[0].legend(fontsize=8)
    for k, lab in enumerate(("$\\Delta s/\\sigma$ (horizontal)", "$\\Delta y/\\sigma$ (bending)",
                             "$\\Delta t/\\sigma$ (time of flight)")):
        ax[k + 1].set_xlabel(lab + " after $M_X^2$ cut")
        ax[k + 1].set_yscale("log")
        ax[k + 1].axvline(-3, color="k", ls=":")
        ax[k + 1].axvline(3, color="k", ls=":")
    save(fig, "fig_matching.png")


def fig_rejection(cals):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    comp = target_composition()
    cfg = SCENARIOS["A5"]().detector
    for s_name, mk in (("KinA", "o-"), ("KinD", "s--")):
        cal = cals.get(("A5", s_name))
        os_ = {n: load("A5", s_name, n) for n in ("free", "N14", "He")}
        if cal is None or any(v is None for v in os_.values()):
            continue
        effs, survN, survHe, fs, foms = [], [], [], [], []
        nsigs = [1, 1.5, 2, 2.5, 3, 4, 5, 7, 10]
        for ns in nsigs:
            e = {}
            for n, o in os_.items():
                w = A.weights(o) / o["wsum_gen"]
                mmc, rec = A.selection(o, cfg, cal, nsig=ns)
                e[n] = ((w * mmc).sum(), (w * rec).sum())
            effs.append(e["free"][1] / e["free"][0])
            survN.append(e["N14"][1] / e["N14"][0])
            survHe.append(e["He"][1] / e["He"][0])
            d = A.dilution(e, comp)
            fs.append(d["f_rec"])
            foms.append(d["fom_ratio"])
        ax[0].plot(effs, survN, mk, color=COL["N14"], label=f"{s_name}: 14N")
        ax[0].plot(effs, survHe, mk, color=COL["He"], label=f"{s_name}: 4He")
        for x, y, ns in zip(effs, survN, nsigs):
            if ns in (2, 3, 5):
                ax[0].annotate(f"{ns}$\\sigma$", (x, y), fontsize=7)
        ax[1].plot(nsigs, fs, mk, color="k", label=f"{s_name}: dilution f with recoil")
        ax[1].plot(nsigs, foms, mk, color="tab:orange", label=f"{s_name}: FOM ratio")
        ax[1].axhline(d["f_raw"], color="gray", ls=":")
    ax[0].set_xlabel("free-proton efficiency of the recoil cut")
    ax[0].set_ylabel("bound-proton survival of the recoil cut")
    ax[0].set_yscale("log")
    ax[0].set_title("Nuclear background rejection (after $M_X^2$ cut)")
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel("matching cut [$\\sigma$]")
    ax[1].set_title("Dilution factor (dotted: $M_X^2$ cut only) and\n"
                    "figure of merit $f^2N$ relative to no recoil")
    ax[1].legend(fontsize=8)
    ax[1].set_xscale("log")
    save(fig, "fig_rejection.png")


# ---------------------------------------------------------------------------
def scenario_metrics(scn, s_name, cals):
    os_ = {n: load(scn, s_name, n) for n in ("free", "N14", "He")}
    if os_["free"] is None:
        return None
    cfg = SCENARIOS[scn]().detector
    o = os_["free"]
    cal = A.MatchCalibration(o, cfg)
    cals[(scn, s_name)] = cal
    m = {"scenario": scn, "setting": s_name, "n_arms": int(o["n_arms"])}
    m.update({"frac_" + k: v for k, v in A.breakdown(o, A.weights(o)).items()})
    _, tot = A.fit_resolution(o)
    m.update({"res_" + k: v for k, v in tot.items()})
    for name in ("ds", "dy", "dt"):
        m[f"match_sigma_{name}_p0.5"] = float(cal.sigma(name, 0.5))
    if all(v is not None for v in os_.values()):
        e = {}
        for n, oo in os_.items():
            w = A.weights(oo) / oo["wsum_gen"]
            mmc, rec = A.selection(oo, cfg, cal, nsig=3.0)
            e[n] = (float((w * mmc).sum()), float((w * rec).sum()))
        m["eff_free_recoil_cut"] = e["free"][1] / e["free"][0]
        m["surv_N14"] = e["N14"][1] / e["N14"][0]
        m["surv_He"] = e["He"][1] / e["He"][0]
        m.update(A.dilution(e, target_composition()))
    m["channels"] = cfg.n_channels()
    m["x_over_X0"] = cfg.material_budget()
    m["area_m2_first_view"] = cfg.area()
    return m


def fmt(v, f="{:.3f}"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f.format(v)


def main():
    cals = {}
    metrics = []
    for scn in SCENARIOS:
        for s in SETTINGS:
            m = scenario_metrics(scn, s.name, cals)
            if m:
                metrics.append(m)
    with open(os.path.join(ROOT, "results", "summary.json"), "w") as f:
        json.dump(metrics, f, indent=1, default=float)
    fig_field()
    fig_geometry()
    fig_kinematics()
    fig_acceptance()
    fig_resolution(cals)
    fig_matching(cals)
    fig_rejection(cals)
    write_markdown(metrics)


def write_markdown(metrics):
    L = []
    L.append("# Recoil SciFi detector: simulation results\n")
    L.append("Generated by `scripts/make_report.py` from the runs of `scripts/run_study.py`. "
             "Numbers are weighted with exp(2.5 t) and are per DVCS event with the electron in "
             "the HMS and the photon on the NPS face. See `docs/recoil_scifi_design.md` for the "
             "method and the assumptions.\n")
    L.append("## Kinematic settings\n")
    L.append("| setting | E [GeV] | x_B | Q^2 [GeV^2] | E' [GeV] | theta_e (HMS) | theta_q (NPS) |")
    L.append("|---|---|---|---|---|---|---|")
    for s in SETTINGS:
        L.append(f"| {s.name} | {s.E} | {s.xB} | {s.Q2} | {s.Eprime:.2f} | "
                 f"{s.theta_e / DEG:.1f} deg | {s.theta_q / DEG:.1f} deg |")
    L.append("\n## Baseline (A5): fate of the recoil proton and stand-alone resolution\n")
    L.append("| setting | stopped | hit magnet | missed tracker | incomplete | tracked | fitted | "
             "sigma_p/p | sigma_theta [mrad] | sin(theta) sigma_phi [mrad] | sigma_t/t (recoil only) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in metrics:
        if m["scenario"] != "A5":
            continue
        L.append(f"| {m['setting']} | {fmt(m['frac_stopped'])} | {fmt(m['frac_hit_magnet'])} | "
                 f"{fmt(m['frac_missed_tracker'])} | {fmt(m['frac_incomplete'])} | "
                 f"{fmt(m['frac_tracked'])} | {fmt(m['frac_fit_ok'])} | {fmt(m['res_dp_rel'])} | "
                 f"{fmt(m['res_dth_mrad'], '{:.1f}')} | {fmt(m['res_dph_mrad'], '{:.1f}')} | "
                 f"{fmt(m['res_dt_rel'])} |")
    L.append("\n## Baseline (A5): exclusivity with the recoil (3 sigma matching cut)\n")
    L.append("Efficiencies are relative to the M_X^2 cut "
             f"[{A.MM2_WINDOW[0]:.2f}, {A.MM2_WINDOW[1]:.2f}] GeV^2. "
             "f = dilution factor (free-proton share of the selected DVCS yield); "
             "FOM = f^2 N relative to the M_X^2 cut alone (> 1: the recoil improves the "
             "statistical precision of target asymmetries). Neutron/proton yield ratio 0.3.\n")
    L.append("| setting | matching sigma_s / sigma_y at 0.5 GeV/c [mm] | sigma_tof [ns] | free eff. | "
             "14N survival | 4He survival | f (M_X^2 only) | f (with recoil) | FOM ratio |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for m in metrics:
        if m["scenario"] != "A5" or "f_raw" not in m:
            continue
        L.append(f"| {m['setting']} | {1e3 * m['match_sigma_ds_p0.5']:.1f} / "
                 f"{1e3 * m['match_sigma_dy_p0.5']:.1f} | {m['match_sigma_dt_p0.5']:.2f} | "
                 f"{fmt(m['eff_free_recoil_cut'])} | {fmt(m['surv_N14'])} | {fmt(m['surv_He'])} | "
                 f"{fmt(m['f_raw'])} | {fmt(m['f_rec'])} | {fmt(m['fom_ratio'], '{:.2f}')} |")
    L.append("\n## Scenario comparison (KinA and KinD)\n")
    L.append("| scenario | description | setting | hit magnet | tracked | sigma_p/p | "
             "free eff. | 14N survival | f (with recoil) | FOM ratio | channels | x/X0 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in metrics:
        if m["setting"] not in ("KinA", "KinD"):
            continue
        L.append(f"| {m['scenario']} | {DESCRIPTIONS[m['scenario']]} | {m['setting']} | "
                 f"{fmt(m['frac_hit_magnet'])} | {fmt(m['frac_tracked'])} | "
                 f"{fmt(m['res_dp_rel'])} | {fmt(m.get('eff_free_recoil_cut'))} | "
                 f"{fmt(m.get('surv_N14'))} | {fmt(m.get('f_rec'))} | "
                 f"{fmt(m.get('fom_ratio'), '{:.2f}')} | {m['channels']} | "
                 f"{100 * m['x_over_X0']:.2f}% |")
    L.append("\n## Figures\n")
    for f, cap in (("fig_geometry.png", "Geometry and sample recoil tracks"),
                   ("fig_field.png", "Target magnet field"),
                   ("fig_recoil_kinematics.png", "Recoil kinematics"),
                   ("fig_acceptance.png", "Recoil efficiency vs -t and phi"),
                   ("fig_resolution.png", "Resolutions"),
                   ("fig_matching.png", "Exclusivity variables, free vs bound protons"),
                   ("fig_rejection.png", "Nuclear background rejection and dilution")):
        L.append(f"**{cap}**\n\n![{cap}](figs/{f})\n")
    with open(os.path.join(ROOT, "results", "RESULTS.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
