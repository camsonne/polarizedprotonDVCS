#!/usr/bin/env python3
"""Unrolled view of the first SciFi station (horizontal angle vs height) with the
recoil hit density (KinA, KinD), the beam hole, and the NPS / HMS footprints."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from recoil.constants import DEG
from recoil.kinematics import SETTINGS
from recoil.scenarios import A5

ROOT = os.path.join(os.path.dirname(__file__), "..")
det = A5().detector
R0 = det.stations[0].radius
fig, axs = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
for ax, sname in zip(axs, ("KinA", "KinD")):
    s = next(x for x in SETTINGS if x.name == sname)
    fn = os.path.join(ROOT, "results", "data", f"A5__{sname}__free.npz")
    d = np.load(fn)
    pos = d["det_pos"][:, 0]
    ok = np.isfinite(pos[:, 0])
    a = np.degrees(np.arctan2(pos[ok, 0], pos[ok, 2]))
    y = pos[ok, 1] * 100
    h = ax.hist2d(a, y, bins=[np.arange(-45, 55, 2.5), np.arange(-50, 75, 2.5)], cmin=1, cmap="viridis")
    plt.colorbar(h[3], ax=ax, label="recoil hits (free p, arm-accepted)")
    a0, a1 = np.degrees(det.alpha_range)
    y0, y1 = np.array(det.y_range) * 100
    ax.add_patch(plt.Rectangle((a0, y0), a1 - a0, y1 - y0, fc="none", ec="b", lw=2, label="station 1 coverage"))
    hb = np.degrees(det.beam_hole); hy = R0 * np.tan(det.beam_hole) * 100
    ax.add_patch(plt.Rectangle((-hb, -hy), 2 * hb, 2 * hy, fc="w", ec="k", lw=1.5, label="beam hole"))
    # NPS face (+-hu,+-hv at distance D) seen at R0
    D = s.nps_distance; hu = 14 * 0.0205; hv = 17 * 0.0205
    au = np.degrees(np.arctan(hu / D)); av = np.degrees(np.arctan(hv / D))
    ax.add_patch(plt.Rectangle((np.degrees(s.theta_q) - au, -R0 * np.tan(av * DEG) * 100), 2 * au,
                               2 * R0 * np.tan(av * DEG) * 100, fc="none", ec="g", lw=2, ls="--", label="NPS acceptance"))
    ax.add_patch(plt.Rectangle((-np.degrees(s.theta_e) - 1.6, -R0 * 0.07 * 100), 3.2, 2 * R0 * 0.07 * 100,
                               fc="none", ec="m", lw=2, ls="--", label="HMS acceptance"))
    ax.set_xlabel("horizontal angle alpha [deg]  (+ = beam left, NPS side)")
    ax.set_title(f"{sname}: station 1 (R = {R0 * 100:.0f} cm) unrolled")
    ax.set_xlim(-45, 55); ax.set_ylim(-50, 75)
axs[0].set_ylabel("height y at R = 52 cm [cm]  (protons bend up)")
axs[0].legend(loc="upper left", fontsize=8)
out = os.path.join(ROOT, "results", "figs", "fig_station_unrolled.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print(out)
