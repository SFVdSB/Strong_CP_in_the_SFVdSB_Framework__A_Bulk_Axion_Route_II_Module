# SFV/dSB Strong-CP (Route II) — Bulk Axion + Overlap Phases + Bounce-Controlled PQ Quality

This repo contains a minimal **Route-II strong-CP module** for the SFV/dSB framework. The module:

1. Generates complex quark Yukawa matrices from **localization/overlap integrals** (with controllable SFV phase gradients).
2. Computes the induced **bare** strong-CP angle  
   \[
   \bar\theta = \theta_{\rm UV} + \arg\det(Y_u Y_d),
   \]
   and shows it is generically **O(1)** when overlap phases are enabled.
3. Introduces a **bulk axion-like mode** that relaxes \(\theta_{\rm eff}=\bar\theta+a/f_a\) to (nearly) zero.
4. Tests **axion quality** by adding explicit PQ breaking and verifying the residual \(\theta_{\rm eff}\).
5. In v7, estimates PQ-breaking suppression from the **corridor bounce** via a geometry-controlled proxy.

The full write-up is included as:
- `SFVdSB_StrongCP_RouteII_BulkAxion.tex`

## References (SFV/dSB gauge normalization & QCD pipeline)
- Phase A: https://doi.org/10.5281/zenodo.17328259  
- Phase B: https://doi.org/10.5281/zenodo.17328272  

(Foundational strong-CP / PQ / axion references are cited in the LaTeX article.)

---

## Repository Contents

**Scripts**
- `sfv_strongcp_route2_bulkaxion_v4.py`  
  Fixed-\(\Lambda_{\rm PQ}\) quality diagnostic (Newton minimizer + perturbative check).
- `sfv_strongcp_route2_bulkaxion_v7.py`  
  Bounce/geometry-driven PQ-breaking proxy with support models (`wall`, `patch`, `power`).

**Inputs**
- `sfv_dsb_calibration.json` (your consolidated SFV/dSB calibration pack)
- `quark_geometry_best.json` (quark localization geometry)
- `background_profile.csv` (bounce profile; used for wall FWHM and \(R/w\)) GENEREATED with goldenRunDetails_v4f_more3_1078.py

**Paper**
- `SFVdSB_StrongCP_RouteII_BulkAxion.tex`

---

## Requirements

- Python 3.10+ recommended
- Typical dependencies:
  - `numpy`
  - `scipy`

Install (example):
```bash
pip install numpy scipy

