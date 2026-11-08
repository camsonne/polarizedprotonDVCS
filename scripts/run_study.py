#!/usr/bin/env python3
"""Run the recoil-detector fast simulation for all scenarios / settings / target species.

Outputs one compressed .npz per run in results/data/ (not committed; re-create with this
script).  Usage:
    python3 scripts/run_study.py [--n 30000] [--jobs 4] [--only A5] [--quick]
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recoil.scenarios import SCENARIOS, run_plan  # noqa: E402
from recoil.study import build, make_fitter, simulate  # noqa: E402
from recoil.kinematics import SETTINGS  # noqa: E402
from recoil.magnet import FieldMap  # noqa: E402
from recoil.nuclear import HE4, N14  # noqa: E402

NUCLEI = {"free": None, "N14": N14, "He": HE4}
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "data")


def flatten(o):
    d = {}
    for k, v in o.items():
        if k == "ev":
            for k2, v2 in v.items():
                d["ev_" + k2] = v2
        else:
            d[k] = v
    return d


def run_task(task):
    scn_name, runs, n, seed = task
    scn = SCENARIOS[scn_name]()
    rng = np.random.default_rng(seed)
    built = build(scn)
    t0 = time.time()
    fitter = make_fitter(built[3], scn, rng) if scn.fit else None
    msgs = [f"[{scn_name}] seed table {time.time() - t0:.0f}s"]
    for set_name, nuc in runs:
        s = next(x for x in SETTINGS if x.name == set_name)
        t0 = time.time()
        o = simulate(s, scn, n, rng, nucleus=NUCLEI[nuc], fitter=fitter, built=built)
        fn = os.path.join(OUT, f"{scn_name}__{set_name}__{nuc}.npz")
        np.savez_compressed(fn, **flatten(o))
        msgs.append(f"[{scn_name}] {set_name} {nuc}: arms {o['n_arms']} tracked "
                    f"{o['tracked'].mean():.3f} fit {o['fit_ok'].mean():.3f} "
                    f"({time.time() - t0:.0f}s)")
    return "\n".join(msgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000, help="generated events per run")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default=None, help="comma-separated scenario names")
    ap.add_argument("--quick", action="store_true", help="n=4000, for testing")
    a = ap.parse_args()
    n = 4000 if a.quick else a.n
    os.makedirs(OUT, exist_ok=True)
    # field maps are cached on disk; compute once before forking
    FieldMap(B0=5.0, cache_dir="cache")
    tasks = []
    only = a.only.split(",") if a.only else None
    for i, (scn_name, runs) in enumerate(run_plan()):
        if only and scn_name not in only:
            continue
        tasks.append((scn_name, runs, n, 1000 + i))
    with Pool(a.jobs) as pool:
        for msg in pool.imap_unordered(run_task, tasks):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
