"""Scintillating-fibre recoil tracker: geometry and digitisation.

The tracker is a set of stations on vertical cylinders (axis = target insert axis = y)
centred on the target.  Each station has several views; each view is a double layer of
round fibres, the second layer staggered by half a pitch:

  X : vertical fibres, measure the arc length s = R * alpha  (alpha = atan2(x, z),
      the horizontal angle from the beam, positive on beam left)
  Y : horizontal (ring) fibres, measure the height y.  In a horizontal transverse field the
      recoil bends mostly vertically, so Y is the bending view.
  U : fibres inclined by +stereo from vertical, measure w = s cos(st) - y sin(st)
      (resolves ghost combinations when several tracks/background hits are present)

Light model: Birks-quenched energy deposit times a photo-electron yield per MeV at the
SiPM; a fibre fires if N_pe >= threshold.  Cluster position = mean of fired fibre centres.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import DEG
from .materials import CFRP, POLYSTYRENE, ROHACELL
from .target import Shell

BIRKS_KB = 0.0126  # cm/MeV, polystyrene


@dataclass
class StationConfig:
    radius: float
    views: tuple = ("X", "Y", "U")
    view_gap: float = 0.003          # radial distance between views [m]


@dataclass
class SciFiConfig:
    name: str = "A_outside_chamber"
    stations: list = field(default_factory=lambda: [StationConfig(0.52), StationConfig(0.57),
                                                    StationConfig(0.62)])
    fibre_d: float = 0.001           # fibre diameter [m]
    pitch_factor: float = 1.05       # pitch / diameter within a layer
    core_fraction: float = 0.94      # core diameter / outer diameter
    stereo: float = 10 * DEG
    alpha_range: tuple = (-40 * DEG, 50 * DEG)   # horizontal angle coverage (+ = beam left)
    y_range: tuple = (-0.40, 0.65)   # vertical extent [m] at the first station radius
    beam_hole: float = 5 * DEG       # half-size of the beam hole (alpha and vertical angle)
    pe_per_MeV: float = 120.0        # SiPM photo-electrons per MeV (quenched) deposited
    pe_threshold: float = 2.5
    sigma_t_view: float = 0.35       # time resolution per view [ns]
    support_cfrp: float = 0.0004     # per station [m]
    support_foam: float = 0.005

    @property
    def pitch(self):
        return self.fibre_d * self.pitch_factor

    def view_list(self):
        out = []
        for si, st in enumerate(self.stations):
            for vi, v in enumerate(st.views):
                out.append((si, v, st.radius + vi * st.view_gap))
        return out

    def shells(self):
        shells = []
        # equivalent thickness of a staggered double layer of round fibres (+10% glue)
        t_view = 2 * np.pi / 4 * self.fibre_d * 1.10 / self.pitch_factor
        R0 = self.stations[0].radius

        def common(R):
            return dict(phi_range=self.alpha_range,
                        y_range=(self.y_range[0] * R / R0, self.y_range[1] * R / R0),
                        hole=(self.beam_hole, R * np.tan(self.beam_hole)), half_length=2.0)

        for si, v, R in self.view_list():
            shells.append(Shell(f"st{si}_{v}", R, t_view, POLYSTYRENE, is_detector=True,
                                view=v, station=si, **common(R)))
        for si, st in enumerate(self.stations):
            R = st.radius - 0.004
            shells.append(Shell(f"st{si}_cfrp", R, self.support_cfrp, CFRP, **common(R)))
            shells.append(Shell(f"st{si}_foam", R - 0.003, self.support_foam, ROHACELL,
                                **common(R)))
        return shells

    def area(self):
        """Area of one view at the first station [m^2]."""
        R = self.stations[0].radius
        return R * (self.alpha_range[1] - self.alpha_range[0]) * (self.y_range[1] - self.y_range[0])

    def material_budget(self):
        tv = 2 * np.pi / 4 * self.fibre_d * 1.10 / self.pitch_factor
        nv = sum(len(s.views) for s in self.stations)
        x = nv * tv * 100 * POLYSTYRENE.rho / POLYSTYRENE.X0
        x += len(self.stations) * (self.support_cfrp * 100 * CFRP.rho / CFRP.X0
                                   + self.support_foam * 100 * ROHACELL.rho / ROHACELL.X0)
        return x

    def n_channels(self):
        n = 0
        a0, a1 = self.alpha_range
        for si, v, R in self.view_list():
            H = (self.y_range[1] - self.y_range[0]) * R / self.stations[0].radius
            arc = R * (a1 - a0)
            if v == "X":
                n += 2 * arc / self.pitch
            elif v == "Y":
                n += 2 * H / self.pitch
            else:
                n += 2 * (arc * np.cos(self.stereo) + H * np.sin(self.stereo)) / self.pitch
        return int(n)


def view_coordinate(pos, view, stereo):
    s = np.hypot(pos[..., 0], pos[..., 2]) * np.arctan2(pos[..., 0], pos[..., 2])
    y = pos[..., 1]
    if view == "X":
        return s
    if view == "Y":
        return y
    if view == "U":
        return s * np.cos(stereo) - y * np.sin(stereo)
    if view == "V":
        return s * np.cos(stereo) + y * np.sin(stereo)
    raise ValueError(view)


def digitize(cfg: SciFiConfig, res, rng, dedx_fn):
    """Return measured view coordinates (N, nview) (NaN = no cluster), photo-electrons and
    times, from the transport result `res` (detector crossings)."""
    views = cfg.view_list()
    n = res.det_T.shape[0]
    meas = np.full((n, len(views)), np.nan)
    npe_tot = np.zeros((n, len(views)))
    tmeas = np.full((n, len(views)), np.nan)
    p = cfg.pitch
    rc = 0.5 * cfg.fibre_d * cfg.core_fraction
    for j, (si, v, R) in enumerate(views):
        ok = np.isfinite(res.det_T[:, j])
        if not np.any(ok):
            continue
        pos = res.det_pos[ok, j]
        d = res.det_dir[ok, j]
        T = res.det_T[ok, j]
        w = view_coordinate(pos, v, cfg.stereo)
        nrm = pos.copy()
        nrm[:, 1] = 0
        nrm /= np.linalg.norm(nrm, axis=1)[:, None]
        cosn = np.maximum(np.abs(np.sum(d * nrm, axis=1)), 0.05)
        S = dedx_fn(T) * POLYSTYRENE.rho  # MeV/cm
        dLdx = S / (1 + BIRKS_KB * S)
        fired_sum = np.zeros(len(w))
        fired_n = np.zeros(len(w))
        npe_all = np.zeros(len(w))
        for off in (0.0, 0.5):
            for dk in (-1, 0, 1):
                k = np.floor(w / p - off) + dk
                c = (k + off + 0.5) * p
                b = np.abs(w - c)
                chord = 2 * np.sqrt(np.maximum(rc**2 - b**2, 0.0))
                # oblique tracks: also accept the width swept across the layer thickness
                path = chord / cosn
                npe = rng.poisson(dLdx * path * 100 * cfg.pe_per_MeV)
                f = npe >= cfg.pe_threshold
                fired_sum += np.where(f, c, 0.0)
                fired_n += f
                npe_all += npe
        m = np.where(fired_n > 0, fired_sum / np.maximum(fired_n, 1), np.nan)
        meas[ok, j] = m
        npe_tot[ok, j] = npe_all
        tt = res.det_time[ok, j] + cfg.sigma_t_view * rng.standard_normal(ok.sum())
        tmeas[ok, j] = np.where(fired_n > 0, tt, np.nan)
    return meas, npe_tot, tmeas


def view_resolution(cfg: SciFiConfig):
    """Nominal single-view resolution [m] used as measurement error in the fit."""
    return cfg.pitch / np.sqrt(12) / 1.6
