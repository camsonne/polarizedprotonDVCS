"""Recoil track reconstruction: vertex-constrained fit through the target field.

Parameters at the vertex: (1/p, a_x = u_x/u_z, a_y = u_y/u_z, z_v); x_v, y_v come from the
raster.  Measurements: one coordinate per fibre view (correlated multiple scattering in the
chamber window and tracker layers in the covariance), the mean time of flight, and a
z-vertex prior from the electron arm.  Predictions come from the deterministic transport
(same field map and mean energy loss).  The fit is a vectorised Levenberg-Marquardt with a
numerical Jacobian, seeded from a nearest-neighbour lookup in a pre-computed track table.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .constants import M_P
from .detector import view_coordinate, view_resolution
from .materials import highland_theta0


def params_to_p3(par):
    ip, ax, ay = par[:, 0], par[:, 1], par[:, 2]
    p = 1.0 / np.abs(ip)
    u = np.column_stack([ax, ay, np.ones_like(ax)])
    u /= np.linalg.norm(u, axis=1)[:, None]
    return p[:, None] * u


def p3_to_params(p3, zv):
    p = np.linalg.norm(p3, axis=1)
    return np.column_stack([1.0 / p, p3[:, 0] / p3[:, 2], p3[:, 1] / p3[:, 2], zv])


class RecoilFitter:
    def __init__(self, transporter, cfg, sigma_z_prior=0.005, sigma_t0=0.10, use_tof=True,
                 n_iter=12):
        self.tr = transporter
        self.cfg = cfg
        self.views = cfg.view_list()
        self.sig_w = view_resolution(cfg)
        self.sz = sigma_z_prior
        self.st0 = sigma_t0
        self.use_tof = use_tof
        self.n_iter = n_iter
        self.n_candidates = 8
        self.proj = []
        for _, v, _ in self.views:
            self.proj.append({"X": (1, 0), "Y": (0, 1),
                              "U": (np.cos(cfg.stereo), -np.sin(cfg.stereo)),
                              "V": (np.cos(cfg.stereo), np.sin(cfg.stereo))}[v])
        self.proj = np.array(self.proj, float)
        self._tree = None

    # ------------------------------------------------------------------
    def predict(self, par, xy):
        vtx = np.column_stack([xy, par[:, 3]])
        res = self.tr.run(vtx, params_to_p3(par), physics=False)
        w = np.full((len(par), len(self.views)), np.nan)
        for j, (_, v, _) in enumerate(self.views):
            w[:, j] = view_coordinate(res.det_pos[:, j], v, self.cfg.stereo)
        with np.errstate(all="ignore"):
            tof = np.nanmean(res.det_time, axis=1)
        return w, tof

    def build_seed_table(self, rng, n=60000):
        """Deterministic tracks from the target centre for the nearest-neighbour seed."""
        p = np.exp(rng.uniform(np.log(0.22), np.log(2.0), n))
        ax = np.tan(rng.uniform(-50, 60, n) * np.pi / 180)
        ay = np.tan(rng.uniform(-50, 70, n) * np.pi / 180) * np.sqrt(1 + ax**2)
        par = np.column_stack([1 / p, ax, ay, np.zeros(n)])
        w, tof = self.predict(par, np.zeros((n, 2)))
        feats, ok = self._features(w)
        ok &= np.isfinite(tof)
        if self.use_tof:
            feats = np.column_stack([feats, self.TOF_W * tof])
        self._tree = cKDTree(feats[ok])
        self._tab = par[ok]

    def _features(self, w):
        """(s, y) of the first and last station from the X/Y views."""
        st = [s for s, _, _ in self.views]
        first, last = min(st), max(st)
        cols = []
        for target_st in (first, last):
            for vv in ("X", "Y"):
                j = [k for k, (s, v, _) in enumerate(self.views) if s == target_st and v == vv]
                cols.append(w[:, j[0]] if j else np.zeros(len(w)))
        f = np.column_stack(cols)
        return f, np.all(np.isfinite(f), axis=1)

    TOF_W = 0.02  # m/ns: weight of the time of flight in the seed lookup

    def seed(self, wmeas, xy, zv, tof=None):
        f, ok = self._features(wmeas)
        if self.use_tof:
            f = np.column_stack([f, self.TOF_W * np.nan_to_num(tof)])
        par = np.full((len(wmeas), 4), np.nan)
        if self._tree is None:
            raise RuntimeError("call build_seed_table first")
        # translate measured features by the vertex offset (approximate)
        f2 = f.copy()
        f2[:, 1] -= xy[:, 1]
        f2[:, 3] -= xy[:, 1]
        k = self.n_candidates
        _, ii = self._tree.query(np.nan_to_num(f2), k=k)
        ii = ii.reshape(len(f2), k)
        if k == 1 or tof is None:
            par[:] = self._tab[ii[:, 0]]
            par[:, 3] = zv
            return par, ok
        # evaluate all candidates with one transport call and keep the best chi2
        cand = self._tab[ii.ravel()].copy()
        cand[:, 3] = np.repeat(zv, k)
        w, t = self.predict(cand, np.repeat(xy, k, axis=0))
        w = w.reshape(len(f2), k, -1)
        t = t.reshape(len(f2), k)
        r = np.where(np.isfinite(wmeas)[:, None, :], wmeas[:, None, :] - w, 0.0)
        r = np.where(np.isfinite(wmeas)[:, None, :] & ~np.isfinite(w), 1.0, r)
        c2 = np.sum((r / 0.003) ** 2, axis=2)
        if self.use_tof:
            c2 += np.nan_to_num((tof[:, None] - t) / 0.3, nan=100.0) ** 2
        best = np.argmin(c2, axis=1)
        par[:] = cand.reshape(len(f2), k, 4)[np.arange(len(f2)), best]
        # rescale the momentum so that the seed reproduces the measured time of flight
        if self.use_tof:
            ts = t[np.arange(len(f2)), best]
            p = 1.0 / par[:, 0]
            b = p / np.sqrt(p**2 + M_P**2) * ts / tof
            good = np.isfinite(b) & (b > 0.1) & (b < 0.95)
            pn = M_P * b / np.sqrt(1 - np.clip(b, 0, 0.99) ** 2)
            par[good, 0] = 1.0 / pn[good]
            # re-adjust the bending-plane slope for the new momentum (1-D scan)
            offs = np.linspace(-0.5, 0.3, 17)
            m = len(offs)
            cand = np.repeat(par, m, axis=0)
            cand[:, 2] += np.tile(offs, len(par))
            w, _ = self.predict(cand, np.repeat(xy, m, axis=0))
            w = w.reshape(len(par), m, -1)
            r = np.where(np.isfinite(wmeas)[:, None, :], wmeas[:, None, :] - w, 0.0)
            r = np.where(np.isfinite(wmeas)[:, None, :] & ~np.isfinite(w), 1.0, r)
            c2 = np.sum(r**2, axis=2)
            best = np.argmin(c2, axis=1)
            par[:, 2] += offs[best]
        return par, ok

    # ------------------------------------------------------------------
    def covariance(self, par, wmask):
        """Measurement covariance (N, nv, nv): intrinsic + correlated multiple scattering."""
        n = len(par)
        nv = len(self.views)
        p = 1.0 / np.abs(par[:, 0])
        beta = p / np.sqrt(p**2 + M_P**2)
        R = np.array([r for _, _, r in self.views])
        C = np.zeros((n, nv, nv))
        idx = np.arange(nv)
        C[:, idx, idx] = self.sig_w**2
        scat = [(s.radius, s.material.x_over_X0(s.thickness)) for s in self.tr.shells
                if s.radius > 0.3]
        PP = self.proj @ self.proj.T
        for Rs, xX0 in scat:
            L = np.clip(R - Rs, 0, None)
            if not np.any(L > 0):
                continue
            th = highland_theta0(p, beta, np.full(n, xX0), log_ref=0.02)
            C += (th**2)[:, None, None] * (np.outer(L, L) * PP)[None]
        big = ~wmask
        C[:, idx, idx] += np.where(big, 1e6, 0.0)
        return C

    @staticmethod
    def _residuals(w, tof, par, wm, tm, zm):
        r = np.where(np.isfinite(wm), wm - w, 0.0)
        bad = np.isfinite(wm) & ~np.isfinite(w)
        r = np.where(bad, 1.0, r)  # 1 m residual: track lost in prediction
        rt = np.where(np.isfinite(tm) & np.isfinite(tof), tm - tof, 0.0)
        rz = zm - par[:, 3]
        return r, rt, rz

    def _chi2_parts(self, par, xy, wm, tm, zm):
        w, tof = self.predict(par, xy)
        return self._residuals(w, tof, par, wm, tm, zm)

    def fit(self, wm, tm, sig_t, xy, zm, seed_par):
        n = len(wm)
        par = seed_par.copy()
        lam = np.full(n, 1e-4)
        wmask = np.isfinite(wm)
        Cinv = np.linalg.inv(self.covariance(par, wmask))
        r, rt, rz = self._chi2_parts(par, xy, wm, tm, zm)

        def chi2(r, rt, rz):
            c = np.einsum("ni,nij,nj->n", r, Cinv, r) + (rz / self.sz) ** 2
            if self.use_tof:
                c += rt**2 / sig_t**2
            return c

        c0 = chi2(r, rt, rz)
        steps = np.array([2e-3, 1e-3, 1e-3, 5e-4])
        for it in range(self.n_iter):
            J = np.zeros((n, len(self.views), 4))
            Jt = np.zeros((n, 4))
            for k in range(4):
                dp = par.copy()
                h = steps[k] * (np.abs(par[:, 0]) if k == 0 else 1.0)
                dp[:, k] += h
                w2, t2 = self.predict(dp, xy)
                rr, rtt, _ = self._residuals(w2, t2, dp, wm, tm, zm)
                J[:, :, k] = -(rr - r) / (h[:, None] if np.ndim(h) else h)
                Jt[:, k] = -(rtt - rt) / h
            J = np.nan_to_num(J)
            Jt = np.nan_to_num(Jt)
            A = np.einsum("nik,nij,njl->nkl", J, Cinv, J)
            b = np.einsum("nik,nij,nj->nk", J, Cinv, r)
            A[:, 3, 3] += 1 / self.sz**2
            b[:, 3] += rz / self.sz**2
            if self.use_tof:
                A += np.einsum("nk,nl->nkl", Jt, Jt) / sig_t[:, None, None] ** 2
                b += Jt * (rt / sig_t**2)[:, None]
            D = np.einsum("nkk->nk", A)
            Al = A + lam[:, None, None] * D[:, :, None] * np.eye(4)[None]
            Al = np.nan_to_num(Al) + 1e-12 * np.eye(4)[None]
            delta = np.einsum("nkl,nl->nk", np.linalg.pinv(Al), np.nan_to_num(b))
            new = par + delta
            new[:, 0] = np.clip(new[:, 0], 0.3, 6.0)
            rn, rtn, rzn = self._chi2_parts(new, xy, wm, tm, zm)
            cn = chi2(rn, rtn, rzn)
            better = cn < c0
            par[better] = new[better]
            r[better], rt[better], rz[better], c0[better] = rn[better], rtn[better], rzn[better], cn[better]
            lam = np.where(better, np.maximum(lam * 0.1, 1e-7), lam * 10)
        ndf = wmask.sum(axis=1) + (np.isfinite(tm) & self.use_tof) + 1 - 4
        cov = np.linalg.pinv(A)
        return par, c0, ndf, cov
