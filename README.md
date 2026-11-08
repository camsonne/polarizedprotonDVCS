# DVCS on a transversely polarized proton (NH3) target: recoil detector studies

This repository collects design studies for a recoil-proton detector for DVCS,
e p↑ → e′ γ p′, on the transversely polarized UVA/JLab NH3 target. The working
configuration is Hall C, with the scattered electron in the HMS, the photon in the NPS
calorimeter, and the target magnet at 5 T with its field horizontal and perpendicular to
the beam.

**Option 1 (this study): scintillating-fibre (SciFi) tracker.** The design note,
[docs/recoil_scifi_design.md](docs/recoil_scifi_design.md), covers the requirements, the
concept, the method, the results and the open issues. The tables and figures produced by
the simulation are in [results/RESULTS.md](results/RESULTS.md).

## What is in here

```
recoil/                  fast-simulation package (numpy/scipy)
  kinematics.py          DVCS 2->3 kinematics, Trento phi, exclusivity predictions
  nuclear.py             Fermi motion / removal energy for bound p in 14N and 4He
  magnet.py              Oxford split-pair coils (from g2psim oxfordfield.f), Biot-Savart map,
                         magnet body used as an obstacle
  target.py              NH3 cup, LHe nose, shields, chamber window (from g2psim G2PTarget.cc)
  materials.py           Bethe stopping power, range tables, Highland scattering, straggling
  transport.py           vectorised RK4 transport with energy loss and multiple scattering
  detector.py            SciFi stations/views, fibre digitisation (Birks, photo-electrons)
  reco.py                vertex-constrained track fit through the field (+ time of flight)
  study.py               end-to-end simulation of one setting
  analysis.py            efficiencies, resolutions, matching cuts, dilution factor
  scenarios.py           detector / magnet variants compared in the study
scripts/run_study.py     run all scenarios (parallel), writes results/data/*.npz
scripts/make_report.py   figures (results/figs), results/RESULTS.md, results/summary.json
tests/                   unit tests (python3 -m pytest tests)
```

## Quick start

```bash
pip install -r requirements.txt
python3 -m pytest tests                      # ~10 s
python3 scripts/run_study.py --n 30000       # ~20 min on 4 cores
python3 scripts/make_report.py               # figures + results/RESULTS.md
```

`scripts/run_study.py --quick` runs a 4000-event version for testing. The field map is
cached in `cache/` on first use.

## Status

| Option | Status |
|---|---|
| 1. Scintillating fibres | fast simulation done: geometry, acceptance, resolution, nuclear background rejection, scenario scan |
| Background rates | parametric only; full Geant4 background simulation still to do |
| Other technologies (GEM/µRWELL, silicon, scintillator bars) | not started |
