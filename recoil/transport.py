"""Vectorised transport of protons through field, target material and detector shells.

Runge-Kutta 4 in the field with operator-split continuous energy loss, Bohr straggling and
Gaussian (Highland) multiple scattering; thin shells are crossed exactly with the path length
at the crossing angle.  With `physics=False` the transport is deterministic (mean energy
loss only) and is used by the track fit.
"""
from __future__ import annotations

import numpy as np

from .constants import C_LIGHT, KAPPA, M_P
from .materials import highland_theta0

ALIVE, EXITED, STOPPED, LOST_MAGNET, TIMEOUT = 0, 1, 2, 3, 4
STATUS_NAMES = {ALIVE: "alive", EXITED: "exited", STOPPED: "stopped",
                LOST_MAGNET: "hit magnet", TIMEOUT: "timeout"}


def _perp_basis(u):
    a = np.where(np.abs(u[:, 0:1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    e1 = np.cross(u, a)
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(u, e1)
    return e1, e2


def _shell_rho(pos, axis):
    if axis == "y":
        return np.hypot(pos[:, 0], pos[:, 2])
    return np.hypot(pos[:, 0], pos[:, 1])


def _shell_normal(pos, axis):
    n = pos.copy()
    if axis == "y":
        n[:, 1] = 0.0
    else:
        n[:, 2] = 0.0
    return n / np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]


class TrackResult:
    def __init__(self, n, n_det):
        self.status = np.zeros(n, int)
        self.det_pos = np.full((n, n_det, 3), np.nan)
        self.det_dir = np.full((n, n_det, 3), np.nan)
        self.det_T = np.full((n, n_det), np.nan)
        self.det_time = np.full((n, n_det), np.nan)
        self.final_pos = np.full((n, 3), np.nan)
        self.final_T = np.full(n, np.nan)
        self.T_after_target = np.full(n, np.nan)

    def hit_mask(self):
        return np.isfinite(self.det_T)


class Transporter:
    def __init__(self, field, target, magnet_body, detector_shells=(), r_stop=None,
                 ms_log_ref=0.02):
        self.field = field
        self.target = target
        self.body = magnet_body
        self.shells = list(target.shells) + list(detector_shells)
        self.det_index = [i for i, s in enumerate(self.shells) if s.is_detector]
        if r_stop is None:
            rmax = max(s.radius for s in self.shells)
            r_stop = rmax + 0.02
        self.r_stop = r_stop
        self.ms_log_ref = ms_log_ref
        self.mats = target.continuous_materials

    # ------------------------------------------------------------------
    def _rk4(self, x, u, k, h):
        """k = KAPPA*q/p per particle; returns new x, u."""
        def f(xx, uu):
            B = self.field(xx)
            return uu, k[:, None] * np.cross(uu, B)

        k1x, k1u = f(x, u)
        k2x, k2u = f(x + 0.5 * h[:, None] * k1x, u + 0.5 * h[:, None] * k1u)
        k3x, k3u = f(x + 0.5 * h[:, None] * k2x, u + 0.5 * h[:, None] * k2u)
        k4x, k4u = f(x + h[:, None] * k3x, u + h[:, None] * k3u)
        xn = x + h[:, None] / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        un = u + h[:, None] / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
        un /= np.linalg.norm(un, axis=1)[:, None]
        return xn, un

    def _scatter(self, rng, u, p, beta, xX0):
        th0 = highland_theta0(np.maximum(p, 1e-3), beta, xX0, log_ref=self.ms_log_ref)
        e1, e2 = _perp_basis(u)
        g = rng.standard_normal((len(u), 2)) * th0[:, None]
        un = u + g[:, 0:1] * e1 + g[:, 1:2] * e2
        return un / np.linalg.norm(un, axis=1)[:, None]

    # ------------------------------------------------------------------
    def run(self, vertex, pvec, rng=None, physics=True, max_steps=2000, charge=1.0, record=False):
        n = len(vertex)
        res = TrackResult(n, len(self.det_index))
        x = np.array(vertex, float)
        p = np.linalg.norm(pvec, axis=1)
        u = pvec / p[:, None]
        T = np.sqrt(p**2 + M_P**2) - M_P
        tof = np.zeros(n)
        idx = np.arange(n)
        left_target = np.zeros(n, bool)
        detpos = {si: j for j, si in enumerate(self.det_index)}
        if physics and rng is None:
            rng = np.random.default_rng()

        traj = [x.copy()] if record else None
        for _ in range(max_steps):
            if len(idx) == 0:
                break
            xs, us, Ts = x[idx], u[idx], T[idx]
            ps = np.sqrt(np.maximum(Ts, 1e-6) * (np.maximum(Ts, 1e-6) + 2 * M_P))
            Es = Ts + M_P
            beta = ps / Es
            rad = np.linalg.norm(xs, axis=1)
            h = np.where(rad < 0.06, 0.002, np.where(rad < 0.35, 0.005, 0.010))
            k = KAPPA * charge / ps
            xn, un = self._rk4(xs, us, k, h)
            tof_s = tof[idx] + h / (beta * C_LIGHT)

            # continuous materials (evaluated at the mid-point)
            mid = 0.5 * (xs + xn)
            mat = self.target.continuous_material(mid)
            Tn = Ts.copy()
            for mi, m in self.mats.items():
                sel = mat == mi
                if not np.any(sel):
                    continue
                Tn[sel] = m.T_after(Ts[sel], h[sel])
                if physics:
                    xX0 = m.x_over_X0(h[sel])
                    un[sel] = self._scatter(rng, un[sel], ps[sel], beta[sel], xX0)
                    sig = m.straggling_sigma(Ts[sel], h[sel])
                    Tn[sel] = np.maximum(Tn[sel] + sig * rng.standard_normal(sel.sum()), 0.0)
            # thin shells
            for si, sh in enumerate(self.shells):
                r0 = _shell_rho(xs, sh.axis)
                r1 = _shell_rho(xn, sh.axis)
                cr = (r0 - sh.radius) * (r1 - sh.radius) < 0
                if not np.any(cr):
                    continue
                c = np.where(cr)[0]
                f = (sh.radius - r0[c]) / (r1[c] - r0[c])
                xc = xs[c] + f[:, None] * (xn[c] - xs[c])
                along = xc[:, 1] if sh.axis == "y" else xc[:, 2]
                ok = np.abs(along) < sh.half_length
                if sh.phi_range is not None:
                    a = np.arctan2(xc[:, 0], xc[:, 2])
                    ok &= (a >= sh.phi_range[0]) & (a <= sh.phi_range[1])
                if sh.y_range is not None:
                    ok &= (xc[:, 1] >= sh.y_range[0]) & (xc[:, 1] <= sh.y_range[1])
                if sh.hole is not None:
                    a = np.arctan2(xc[:, 0], xc[:, 2])
                    ok &= ~((np.abs(a) < sh.hole[0]) & (np.abs(xc[:, 1]) < sh.hole[1]))
                c, xc, f = c[ok], xc[ok], f[ok]
                if len(c) == 0:
                    continue
                nrm = _shell_normal(xc, sh.axis)
                cosps = np.maximum(np.abs(np.sum(un[c] * nrm, axis=1)), 0.05)
                path = sh.thickness / cosps
                if sh.is_detector:
                    j = detpos[si]
                    gi = idx[c]
                    first = ~np.isfinite(res.det_T[gi, j])
                    gi1 = gi[first]
                    res.det_pos[gi1, j] = xc[first]
                    res.det_dir[gi1, j] = un[c][first]
                    res.det_T[gi1, j] = Tn[c][first]
                    res.det_time[gi1, j] = (tof[idx][c] + f * (tof_s[c] - tof[idx][c]))[first]
                Tc = Tn[c]
                Tn[c] = sh.material.T_after(Tc, path)
                if physics:
                    pc = np.sqrt(Tc * (Tc + 2 * M_P))
                    bc = pc / (Tc + M_P)
                    un[c] = self._scatter(rng, un[c], pc, bc, sh.material.x_over_X0(path))
                    sig = sh.material.straggling_sigma(Tc, path)
                    Tn[c] = np.maximum(Tn[c] + sig * rng.standard_normal(len(c)), 0.0)

            x[idx], u[idx], T[idx], tof[idx] = xn, un, Tn, tof_s
            # bookkeeping of the energy after the target region (r > 4 cm)
            lt = (~left_target[idx]) & (np.linalg.norm(xn, axis=1) > 0.04)
            res.T_after_target[idx[lt]] = Tn[lt]
            left_target[idx[lt]] = True

            st = np.zeros(len(idx), int)
            st[Tn < 5e-4] = STOPPED
            hit = self.body.inside(xn)
            st[hit & (st == ALIVE)] = LOST_MAGNET
            rxz = np.hypot(xn[:, 0], xn[:, 2])
            st[((rxz > self.r_stop) | (np.abs(xn[:, 1]) > 1.5)) & (st == ALIVE)] = EXITED
            done = st != ALIVE
            res.status[idx[done]] = st[done]
            idx = idx[~done]
            if record:
                xr = x.copy()
                traj.append(xr)
        res.status[idx] = TIMEOUT
        if record:
            res.trajectory = np.array(traj)
        res.final_pos, res.final_T = x, T
        return res
