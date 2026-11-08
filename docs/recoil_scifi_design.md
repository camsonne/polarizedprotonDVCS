# Recoil detector for DVCS on a transversely polarized NH3 target: option 1, scintillating fibres

Status: first design study (fast simulation). Every number below is reproduced by
`scripts/run_study.py` + `scripts/make_report.py`. The full tables are in
[../results/RESULTS.md](../results/RESULTS.md).

## 1. Why detect the recoil proton

The measurement is e p↑ → e′ γ p′ on the UVA/JLab dynamically polarized NH3 target, at
5 T and 1 K, with the polarization transverse to the beam. The working configuration is
Hall C: e′ in the HMS and γ in the NPS calorimeter. Without the recoil, exclusivity rests
on the missing mass M_X² of ep → eγX alone, and three problems remain:

* **Dilution.** Free protons are 3 of the 17 + ~2 nucleons in the cup (NH3 beads in LHe, packing
  fraction 0.6). Quasi-free DVCS/BH on the protons and neutrons of ¹⁴N and ⁴He passes the
  M_X² cut almost as well as the free-proton signal, because Fermi motion only broadens
  M_X² to σ ≈ 0.2 GeV² against 0.08 GeV² for the signal. In this simulation the free-proton
  share of the selected yield is **f ≈ 0.26**.
* The nuclear background carries its own (small, polarized-¹⁵N/¹⁴N) asymmetries and must
  be subtracted with separate carbon/empty-target runs, which adds systematic uncertainty.
* Associated production (eγ N π, eγ Δ) sits just above the M_X² cut. This channel is not
  simulated here; a recoil with momentum and direction measured suppresses it as well.

Requiring a recoil proton that matches the direction, bending and time of flight predicted
from e′ + γ removes most of the nuclear background, as shown below.

## 2. Environment

| Item | Model used | Source |
|---|---|---|
| Magnet | Oxford split pair, 14 windings, 5.0 T at the centre (checked: 4.9996 T at the nominal 106.1 A); axis horizontal and ⟂ beam | `camsonne/g2psim` `src/oxfordfield.f` |
| Magnet body (stops protons) | windings + 5 mm margin, radial gaps between windings filled → clear openings ±24° about the mid-plane (split) and a 32° cone about the axis. Scanned 0 / 5 / 10 mm. | to verify with cryostat drawings |
| Target | Kel-F cup r = 1.36 cm, L = 2.83 cm, 0.89 mm wall; NH3 (0.817 g/cm³) in LHe, pf = 0.6; LHe nose r = 2.1 cm, 4 mil Al; 4 K and LN2 shields; chamber r = 47.9 cm, 20 mil Al window; raster r = 1 cm | `camsonne/g2psim` `src/G2PTarget.cc` |
| Kinematics | 10.6 GeV; (x_B, Q²) = (0.36, 3.0), (0.36, 4.0), (0.50, 3.4), (0.50, 4.8), (0.60, 5.1) | representative NPS-like settings |
| HMS | beam right, ±9% × ±28 mrad × ±70 mrad; σ_p/p = 1e-3, σ_θ = 1.5 mrad, σ_z(vertex) = 5 mm | |
| NPS | beam left, centred on q, 4 m, 30×36 blocks of 2.05 cm (one-block fiducial); σ_E/E = 2.5%/√E ⊕ 1%, 3 mm | |

Consequences for the recoil (figures `fig_field.png`, `fig_geometry.png`):

* The field is 5 T within ~10 cm of the target and drops to 0.3–0.7 T at the chamber
  window. The kick therefore happens in the first ~15 cm. With B along +x (beam left), the
  recoil is deflected **upwards** (vertical bending plane).
* Recoils span roughly −30° to +40° horizontally and −40° to +65° vertically. They are
  forward, on both sides of the beam, and overlap the NPS and HMS directions at low |t|.
* Recoils leaving between ~24° and ~55° from the beam in the horizontal plane hit the
  coils. Because p_x is conserved in a field along x, this loss is fixed at the vertex
  and depends strongly on the Trento φ: recoils near **φ ≈ 0 are lost** (KinA/KinD
  efficiency < 10% for φ within ±30° of 0), and φ ≈ 180° is fully open
  (`fig_acceptance.png`, right panel).
* Range threshold: a proton needs T ≳ 38 MeV (p ≈ 0.27 GeV/c, −t ≈ 0.07 GeV²) to leave
  the cup radially. The radial target material is 2.5% X₀, which gives θ₀ = 22 / 8 / 4 mrad
  at p = 0.3 / 0.5 / 0.8 GeV/c. **Multiple scattering in the target dominates the angular
  resolution**, not the tracker.

## 3. Baseline concept (scenario A5)

* **Location:** outside the scattering chamber, in air, on vertical cylinders centred on
  the target. Three stations at R = 52, 57, 62 cm (10 cm lever arm). Coverage: horizontal
  angle −40° to +50°; vertical −40 cm to +65 cm at R = 52 cm (scaled with R), i.e. −38° to
  +51°. A ±5° hole is left for the beam. The tracker also covers the HMS and NPS
  directions. For 6 GeV electrons the multiple scattering is negligible. For photons the
  4.2% X₀ means ~3% conversions, most of which still shower in the NPS.
* **Each station:** three views (X vertical fibres → horizontal angle; Y horizontal ring
  fibres → height, the bending coordinate; U at 10° stereo for ghost rejection). Each view
  is a staggered double layer of Ø1 mm round fibres (pitch 1.05 mm) on CFRP/Rohacell
  support.
* **Readout:** SiPMs on the fibre ends. They work in the 0.2–0.7 T fringe field.
  Light model: 120 photo-electrons per quenched MeV, Birks kB = 0.0126 cm/MeV, threshold
  2.5 p.e. Recoil protons give tens to hundreds of p.e. per fibre, so the efficiency per
  view is ≈ 100% with no dead gaps. Time resolution 350 ps per view.
* **Size:** 1.4 m² per view at the first station; **≈ 18 k channels** (12 k with two
  stations, 9 k with 2 mm fibres).

## 4. Simulation and reconstruction

* **Generator.** e N → e′ γ N′, flat in t (t_min to t_min − 1.2 GeV²) and in Trento φ,
  with e′ in the HMS box. Integrated numbers are weighted by exp(2.5 t). Bound protons in
  ¹⁴N (k_F = 221 MeV/c) and ⁴He (180 MeV/c) get a 1/k⁴ tail (15–20%) and 20 MeV removal
  energy; FSI is not included.
* **Transport.** RK4 in the computed field map with Bethe energy loss (validated against
  PSTAR to <4% in the tests), Bohr straggling and Highland scattering in every volume.
  The magnet body is an absorber.
* **Two reconstruction modes:**
  1. *Stand-alone fit.* Vertex from the raster (x, y) and the HMS (z prior), fibre
     clusters with correlated multiple-scattering covariance, and the time of flight.
     Levenberg–Marquardt through the field map, seeded from a lookup table refined with
     the time of flight.
  2. *Kinematic matching (used for exclusivity).* The recoil predicted from e′ and the
     photon **direction** (the photon energy follows from exclusivity) is transported
     through the same field to the tracker and compared with the hits: Δs (horizontal),
     Δy (bending) and Δt. The widths σ(p) are calibrated on free protons, and the cut is
     (Δs/σ)² + (Δy/σ)² < 3² together with |Δt/σ| < 3.

### Important finding: bending gives almost no momentum information

With all stations outside the strong field, the measured exit line plus the beam-spot
vertex leaves the initial vertical angle, 1/p and z_vertex almost degenerate. The kick
happens near the vertex, and the derivatives of the hit positions with respect to these
three parameters are nearly proportional. **The momentum comes from the time of flight**
(0.55–0.65 m path, 350 ps per view, 100 ps start time). This gives σ_p/p ≈ 3–9% for
p = 0.3–1.1 GeV/c, and σ_θ ≈ 15–20 mrad, which is dominated by target scattering and by
the momentum–angle correlation. Moving the stations inside the vacuum at R = 30–38 cm
(scenario B5) only improves σ_p/p from 7.5% to 5.4% at KinD. The recoil therefore does
not compete with e′γ for measuring t (σ_t/t ≈ 10–15% from the recoil alone). Its job is
exclusivity.

## 5. Results (baseline A5, 30 k events per setting and species)

| | KinA | KinB | KinC | KinD | KinE |
|---|---|---|---|---|---|
| recoil stopped in target | 2.4% | 1.2% | 0 | 0 | 0 |
| recoil hits magnet | 32% | 32% | 26% | 23% | 20% |
| misses tracker (mostly above +65 cm) | 13% | 12% | 4% | 3% | 3% |
| **tracked and fitted** | **51%** | **54%** | **69%** | **74%** | **77%** |
| free-p efficiency of M_X² + 3σ recoil cut (relative to M_X² only) | 0.49 | 0.52 | 0.66 | 0.70 | 0.74 |
| ¹⁴N bound-p survival | 0.6% | 0.4% | 1.3% | 0.8% | 1.1% |
| ⁴He bound-p survival | 1.1% | 0.9% | 2.1% | 1.9% | 2.0% |
| dilution f, M_X² only → with recoil | 0.25 → 0.97 | 0.26 → 0.98 | 0.26 → 0.96 | 0.26 → 0.97 | 0.27 → 0.97 |
| FOM f²N, relative to M_X² only | 1.9 | 2.0 | 2.4 | 2.6 | 2.7 |

The neutron/proton yield ratio is set to 0.3. Neutrons without FSI never give a matching
proton.

**Overall: even with the coil and acceptance losses, requiring the recoil roughly doubles
the statistical figure of merit for target asymmetries, and it makes the sample almost
free of nuclear background (f ≈ 0.97).** This also removes most of the dilution-factor
systematic.

### Scenario scan (KinA / KinD)

| Variant | Effect |
|---|---|
| 2 stations instead of 3 | same performance (FOM 1.85/2.59 vs 1.89/2.60), 12 k channels, 2.8% X₀; the third station is mainly for robustness against background |
| 2 mm fibres | same performance, 9 k channels, but 7.7% X₀ in front of the NPS |
| 2.5 T | KinA tracked fraction 0.53 → 0.63, FOM 1.9 → 2.3 (less upward bending out of the coverage); target polarization is lower at 2.5 T, so this needs a polarization-weighted FOM |
| in vacuum, R = 30–38 cm | same acceptance, 40% fewer channels, slightly better σ_p; needs fibres and readout inside the chamber vacuum |
| magnet margin 0 / 10 mm | magnet losses 28/18% vs 38/29% (KinA/KinD); **the result depends on the real cryostat openings** |
| taller tracker (−60 to +100 cm) | KinA FOM 1.9 → 2.1 at +23% channels, if the chamber windows allow it |
| vertical field axis (another magnet) | KinD magnet loss 23% → 12%, FOM 3.0 |

## 6. Background and rates (parametric, still open)

No background simulation has been done yet. Occupancy per fibre for a uniform hit flux Φ
in a 20 ns window, for 1 mm fibres 0.8–1.05 m long:

| Φ [Hz/cm²] | rate per fibre | occupancy |
|---|---|---|
| 10⁴ | 0.1 MHz | 0.2% |
| 10⁵ | 1 MHz | 2% |
| 10⁶ | 10 MHz | 20% |

The tracker can work with Φ up to a few 10⁵ Hz/cm². Splitting the fibres at y = 0 or
reading both ends extends this. The transverse field helps: low-energy charged
background (Møller, δ-electrons) spirals along the field lines and leaves through the
axial cones at ~90° beam left and beam right, not through the forward split region where
the tracker sits. Photons and neutrons are not swept. The neutron dose on the SiPMs
(dark-count increase) needs to be estimated.

## 6b. Variant C: fibres inside the magnet, read out above it

The fibre barrel sits in the split gap of the magnet, in the target vacuum. It lies
between the 4 K shield (r = 3.8 cm) and the coil faces (|x| > 6 cm). There are two
stations at r = 4.3 and 5.1 cm. Each has X fibres (vertical) and U/V fibres (±20°
stereo) of 0.5 mm diameter. Coverage is ±75° horizontally and −67° to +70° vertically,
with a ±14° × ±1 cm hole for the rastered beam. All fibres run roughly vertically, so
they can leave the gap upwards along the target insert. Clear-fibre light guides then
take the light to SiPMs outside the field. The channel count is 4 k, and the material in
front of the NPS and HMS is 1.3% X₀.

The proton is measured **before** it reaches the coils, so the coil shadow disappears.
At r ≈ 5 cm the bending is negligible, so the barrel measures the direction near the
vertex. The beam-spot vertex is not precise enough here: a 5 mm z uncertainty is
~0.1 rad at 5 cm. Exclusivity therefore uses the **direction** residual between the two
stations, which does not depend on a common shift of the track. The resolution is
13–30 mrad, dominated by target multiple scattering, against ~300 mrad of Fermi smearing.

| | A5 (outer) KinA | C5 (inner) KinA | A5 KinD | C5 KinD |
|---|---|---|---|---|
| recoil reaches the tracker | 0.52 | **0.85** | 0.74 | 0.75 |
| free-p efficiency of the recoil cut | 0.49 | **0.80** | 0.70 | 0.70 |
| ¹⁴N bound-p survival | **0.6%** | 3.0% | **0.8%** | 2.5% |
| dilution f with recoil | **0.97** | 0.92 | **0.97** | 0.92 |
| FOM relative to M_X² only | 1.9 | **2.9** | **2.6** | 2.4 |
| channels | 18 k | **4 k** | 18 k | 4 k |

The "hit magnet" column of RESULTS.md is not meaningful for C5: the proton is measured
before it reaches the coils.

Tracking efficiency vs Trento φ (45° bins starting at φ = 0):

```
A5 KinA  0.04 0.34 0.64 0.77 0.74 0.91 0.67 0.04
C5 KinA  0.99 0.99 0.95 0.72 0.57 0.86 0.97 0.98
A5 KinD  0.09 0.69 1.00 0.99 0.90 0.91 0.72 0.07
C5 KinD  0.99 0.96 0.86 0.52 0.36 0.73 0.94 0.99
```

**The two are complementary.** The inner barrel is fully efficient near φ = 0, where the
coils block the outer tracker and where BH peaks. It loses efficiency near φ = 180°,
where the recoil goes close to the beam hole. The outer tracker does the opposite. An
**inner barrel + outer stations hybrid** would give nearly uniform φ acceptance. The inner
barrel would provide the vertex-near direction and acceptance; the outer stations would
provide the time of flight, the momentum and the best rejection. A combined matching for
the hybrid still has to be written.

Practical issues for the inner barrel:
* **Cryogenic vacuum.** It sits next to the 4 K shield, so the heat load on the
  refrigerator, the fibre behaviour at low temperature (brittleness, light yield) and
  outgassing all need checking.
* **Rates and radiation dose at 5 cm are ~100× those at 50 cm.** Low-energy charged
  particles spiral along the field lines (±x), i.e. through the barrel at α ≈ ±90°.
  Coverage is therefore limited to ±75°, but the photon flux remains. Fibre radiation
  damage over a run must be estimated with the Geant4 background.
* **Raster and beam hole.** A smaller raster or a smaller beam hole directly recovers
  efficiency near φ = 180°.
* **Momentum.** With no time-of-flight lever arm, dE/dx in the 12 fibre layers
  (truncated mean) could give β for slow protons; this has not been simulated yet.

## 7. Open issues and next steps

1. **Real openings:** use the magnet cryostat, heat-shield and chamber-window drawings.
   The magnet loss (20–38%) and the φ holes near φ = 0 are the leading uncertainties.
   The chamber needs large vertical exit windows (about −40° to +50°).
2. **Geant4 background** at 10³⁵ cm⁻² s⁻¹ (photons, hadrons, Møller in the field) to
   get Φ and to check hit finding with random hits (stereo views, timing window).
3. **Physics generator:** weight with real BH+DVCS cross sections and transverse target
   spin asymmetries (EpIC/PARTONS, HepMC3 input), including the effect of the φ-dependent
   acceptance on the extraction of sin(φ−φ_S) moments.
4. **Associated production** (eγNπ⁰, eγΔ⁺) and FSI in ¹⁴N.
5. **Integration:** NPS sweeping-magnet/shielding, HMS and beam-pipe clearance, the
   chicane-deflected beam, SiPM placement, cooling and radiation shielding.
6. **Alternative technologies** to compare with the same framework (only `detector.py`
   changes): µRWELL/GEM, silicon, or a thick-scintillator ΔE–E/ToF hodoscope. A hodoscope
   may be enough, since momentum comes from the time of flight anyway.
