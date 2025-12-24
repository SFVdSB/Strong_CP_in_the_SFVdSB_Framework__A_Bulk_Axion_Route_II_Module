#!/usr/bin/env python3
"""
sfv_strongcp_route2.py

Route-2 Strong-CP module for SFV/dSB:
- Builds Yukawa matrices from SFV overlap geometry (quark_geometry_best.json)
- Optionally injects geometric phases into overlaps to generate CKM CPV
- Computes bare strong-CP angle: theta_bar_bare = theta_uv + arg det(Yu Yd)
- Demonstrates axion relaxation: theta_eff -> 0 after minimizing V(a)
- Estimates f_a from SFV calibration scales (M_brane, ell_s) if provided

Usage examples:
  # baseline (real overlaps; will typically give argdet ~ 0)
  python sfv_strongcp_route2.py --geom quark_geometry_best.json

  # turn on phases (example k vectors in 1/(string-coordinate) units)
  python sfv_strongcp_route2.py --geom quark_geometry_best.json \
    --phases on --kL "0.0,0.4,0.9" --kRu "0.1,0.3,0.0" --kRd "-0.2,0.2,0.0" --kH 0.15

  # include SFV calibration pack to estimate f_a
  python sfv_strongcp_route2.py --geom quark_geometry_best.json \
    --calib "sfv_dsb_calibration_condensed_with_xi (1).json" --phases on
    
    python sfv_strongcp_route2.py --phases on ^
  --geom quark_geometry_best.json ^
  --calib sfv_dsb_calibration.json ^
  --kL 0.0,0.04,0.09 ^
  --kRu 0.01,0.03,0.0 ^
  --kRd=-0.02,0.02,0.0 ^
  --kH 0.015

"""

import argparse, json, math, csv
import numpy as np

# ----------------------------
# Helpers: parse CSV-like args
# ----------------------------
def parse_vec3(s: str) -> np.ndarray:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated values, got: {s}")
    return np.array([float(x) for x in parts], dtype=float)

def arg_det(z: complex) -> float:
    """Principal value arg in (-pi, pi]."""
    return float(np.angle(z))

def jarlskog(V: np.ndarray) -> float:
    """
    J = Im(V_ud V_cs V_us* V_cd*)
    using indices (0,0),(1,1),(0,1),(1,0).
    """
    return float(np.imag(V[0,0] * V[1,1] * np.conj(V[0,1]) * np.conj(V[1,0])))

# ----------------------------
# SFV quark-scan conventions
# ----------------------------
def scale_L(yL: np.ndarray, k: float, yH: float) -> np.ndarray:
    """Interpolate L centers toward yH with factor k in [0,1]."""
    return (1.0 - k) * yH + k * yL

def misalign_L_down(yL_d0: np.ndarray, th12: float, th23: float, th13: float) -> np.ndarray:
    """Apply small antisymmetric shifts to L_d centers to seed CKM structure (matches sfv_quark_scan.py)."""
    y = yL_d0.copy()
    y[0] += -th12 + th13
    y[1] +=  th12 - th23
    y[2] +=  th23 - th13
    return y

# ----------------------------
# Complex triple-overlap with phases
# ----------------------------
def triple_overlap_complex(aL, sL, aH, sH, aR, sR, k_tot: float) -> complex:
    """
    Integral of three normalized 1D Gaussians * exp(i k_tot y).

    This is the real triple_overlap multiplied by:
      exp(i k_tot * mu) * exp(-k_tot^2 / (2 a))
    where a = sum(1/s_i^2), mu is the precision-weighted mean.

    Matches the normalization used in sfv_quark_scan.py (normalized Gaussians).
    """
    aL2, aH2, aR2 = 1.0/(sL*sL), 1.0/(sH*sH), 1.0/(sR*sR)
    a = aL2 + aH2 + aR2
    mu = (aL2*aL + aH2*aH + aR2*aR) / a
    E = 0.5 * (aL2*(aL-mu)**2 + aH2*(aH-mu)**2 + aR2*(aR-mu)**2)
    pref = (1.0/(np.pi**(3/4.0) * math.sqrt(sL*sH*sR))) * math.sqrt(2.0*np.pi/a)

    real_part = pref * math.exp(-E)
    phase_factor = np.exp(1j * k_tot * mu) * np.exp(-(k_tot*k_tot) / (2.0*a))
    return complex(real_part) * phase_factor

def Ymat_complex(yL, sL, yR, sR, yH, sH, kL, kR, kH) -> np.ndarray:
    """
    Build 3x3 Yukawa matrix from overlaps:
      Y_ij = ∫ dy (L_i)(H)(R_j) exp(i (kL_i + kH + kR_j) y)
    """
    Y = np.zeros((3,3), dtype=complex)
    for i in range(3):
        for j in range(3):
            k_tot = float(kL[i] + kH + kR[j])
            Y[i,j] = triple_overlap_complex(float(yL[i]), float(sL[i]), float(yH), float(sH),
                                            float(yR[j]), float(sR[j]), k_tot)
    return Y

def svd_sorted_complex(Y: np.ndarray):
    """SVD with singular values sorted descending."""
    U, S, Vh = np.linalg.svd(Y, full_matrices=True)
    idx = np.argsort(-S)
    return U[:,idx], S[idx], Vh[idx,:]

# ----------------------------
# f_a estimates from calibration
# ----------------------------
def estimate_fa_from_calib(calib: dict, kappa_a: float = 1.0, L_a_model: str = "ell_s", L_a_custom_GeV_inv: float = None, bounce_csv_path: str = None):
    """
    Estimate axion decay constant candidates from the SFV condensed calibration JSON.

    We assume the axion is a BULK phase mode (Route II). The 4D effective decay constant depends on:
      (i) a microscopic bulk stiffness scale (we proxy with M_brane),
      (ii) the zero-mode support length L_a in the normal/bulk direction.

    Models returned:
      fa1:  ~ kappa_a / ell_s                       (string-scale estimate)
      fa2:  ~ kappa_a * M_brane * sqrt(M_brane*L_a) (bulk zero-mode normalization with M5~M_brane)
      fa_phi: ~ kappa_a * v_bulk * Mpl_reduced      (if axion is phase of the bulk bounce field Φ)

    L_a choices:
      - ell_s: use ell_s_GeV_inv from electroweak.SFV_scales
      - xi_heal: use hydrodynamics.sfv_bulk_params.xi_heal_m converted to GeV^-1
      - custom: use L_a_custom_GeV_inv

    Requires:
      electroweak.SFV_scales.{ell_s_GeV_inv, M_brane_GeV}
    Optional:
      hydrodynamics.sfv_bulk_params.xi_heal_m
      bounce.two_field_O4.vevs_over_Mpl.v_bulk and units.Mpl_reduced_GeV
    """
    ew = calib.get("electroweak", {}).get("SFV_scales", {})
    M = float(ew.get("M_brane_GeV", np.nan))
    ell = float(ew.get("ell_s_GeV_inv", np.nan))
    if not np.isfinite(M) or not np.isfinite(ell) or ell <= 0 or M <= 0:
        return None

    # 1) string-scale-like estimate
    fa1 = kappa_a / ell

    # 2) bulk zero-mode support length
    L_a = ell

    if L_a_model == "bounce_FWHM":
        # Prefer the calibrated FWHM used in transport, else read from bounce CSV.
        FWHM = None
        R_peak = None
        try:
            FWHM = float(calib.get("transport", {}).get("distributed_LZ", {}).get("inputs", {}).get("FWHM", np.nan))
            if not np.isfinite(FWHM) or FWHM <= 0:
                FWHM = None
        except Exception:
            FWHM = None
        if FWHM is None:
            if bounce_csv_path is None:
                # Try calib.bounce.profile_csv (if present)
                bounce_csv_path = calib.get("bounce", {}).get("profile_csv", None)
            if bounce_csv_path is None:
                raise ValueError("L_a_model=bounce_FWHM requires --bounce_csv or calib.transport.distributed_LZ.inputs.FWHM")
            FWHM, R_peak = read_bounce_fwhm_from_csv(bounce_csv_path)
        # Convert dimensionless FWHM to physical length using ell_s
        L_a = float(FWHM) * ell
    elif L_a_model == "xi_heal":
        # SI -> GeV^-1 conversion (1 m = 5.067730716e15 GeV^-1)
        m_to_GeV_inv = 5.067730716e15
        xi_m = calib.get("hydrodynamics", {}).get("sfv_bulk_params", {}).get("xi_heal_m", None)
        if xi_m is not None and float(xi_m) > 0:
            L_a = float(xi_m) * m_to_GeV_inv
    elif L_a_model == "custom":
        if L_a_custom_GeV_inv is None or L_a_custom_GeV_inv <= 0:
            raise ValueError("L_a_model=custom requires --L_a_GeV_inv > 0")
        L_a = float(L_a_custom_GeV_inv)
    elif L_a_model != "ell_s":
        raise ValueError(f"Unknown L_a_model: {L_a_model}")

    fa2 = kappa_a * M * math.sqrt(M * L_a)

    # 3) bulk bounce VEV estimate (Φ phase mode)
    fa_phi = None
    try:
        v_bulk = float(calib.get("bounce", {}).get("two_field_O4", {}).get("vevs_over_Mpl", {}).get("v_bulk", np.nan))
        Mpl_red = float(calib.get("units", {}).get("Mpl_reduced_GeV", np.nan))
        if np.isfinite(v_bulk) and np.isfinite(Mpl_red) and v_bulk > 0 and Mpl_red > 0:
            fa_phi = kappa_a * v_bulk * Mpl_red
    except Exception:
        fa_phi = None

    return dict(
        M_brane_GeV=M,
        ell_s_GeV_inv=ell,
        L_a_model=L_a_model,
        L_a_GeV_inv=L_a,
        fa1_GeV=fa1,
        fa2_GeV=fa2,
        fa_phi_GeV=fa_phi,
    )
 

def estimate_pq_break_from_bounce(calib: dict, model: str = "wall", power: float = 1.0,
                                 Mstar_choice: str = "M_brane", bounce_csv_path: str = None,
                                 S_E_override: float = None,
                                 micro_ell_model: str = "ell_s", micro_n_tan: int = 3) -> dict:
    """
    Estimate an explicit PQ-breaking scale Lambda_PQ from SFV/dSB bounce + geometry as a *proxy*.

    We parameterize a semiclassical suppression:
        Lambda_PQ ≈ M_* exp(-S_break/4)

    and build S_break as a scaled version of the already-derived O(4) bounce action S_E,
    multiplied by a geometric factor involving the wall thickness.

    This is NOT yet the final first-principles derivation of PQ violation (that requires modeling
    the specific PQ-violating defect/instanton). It is a controlled way to turn your existing
    SFV/dSB data into a falsifiable corridor for axion quality.

    Models:
      - "bounce":  S_break = S_E
      - "wall":    S_break = S_E * (w/R)          (defect supported over wall 4-volume ~ R^3 w)
      - "patch":   S_break = S_E * (w/R)^2        (more localized, generally *worse* quality)
      - "power":   S_break = S_E * (w/R)^power

    M_* choices:
      - "M_brane": use electroweak.SFV_scales.M_brane_GeV (default)
      - "M_string": use 1/ell_s_GeV_inv
      - "fa2": use the fa2 estimate if present (requires you already called estimate_fa_from_calib)
    """
    out = {"ok": False}
    b = calib.get("bounce", {}).get("two_field_O4", {})
    geom = b.get("geometry", {}) if isinstance(b, dict) else {}

    # Action: prefer explicit CLI override if provided (override wins).
    # This avoids brittle dependence on calibration JSON key paths across repo versions.
    if (S_E_override is not None):
        try:
            S_E = float(S_E_override)
        except Exception:
            S_E = float('nan')
    else:
        S_E = float('nan')

    # Otherwise, try to read from calibration pack (multiple possible key paths).
    if not np.isfinite(S_E):
        # Newer condensed packs: bounce.two_field_O4.S_E
        try:
            S_E = float(b.get('S_E', np.nan))
        except Exception:
            S_E = float('nan')

    if not np.isfinite(S_E):
        # Older packs: bounce.corridor_solution.S_E
        try:
            S_E = float(calib.get('bounce', {}).get('corridor_solution', {}).get('S_E', np.nan))
        except Exception:
            S_E = float('nan')

    if (not np.isfinite(S_E)) or (S_E <= 0):
        return out



    # Try to read absolute bounce scales (dimensionless) from bounce CSV if provided.
    # This is needed for "micro_size" model where the operator coherence length is compared to the bubble radius R_peak.
    w_dimless = float("nan")
    R_dimless = float("nan")
    if bounce_csv_path:
        try:
            w_csv, R_csv = read_bounce_fwhm_from_csv(bounce_csv_path)
            w_dimless = float(w_csv)
            if R_csv is not None:
                R_dimless = float(R_csv)
        except Exception:
            pass

    R_over_w = float(geom.get("R_over_w", np.nan))
    if not np.isfinite(R_over_w) or R_over_w <= 0:
        # Some packs store w_over_R_sq; invert safely
        w_over_R_sq = float(geom.get("w_over_R_sq", np.nan))
        if np.isfinite(w_over_R_sq) and w_over_R_sq > 0:
            w_over_R = math.sqrt(w_over_R_sq)
            R_over_w = 1.0 / w_over_R
    if not np.isfinite(R_over_w) or R_over_w <= 0:
        # Fallback: infer w/R from the bounce profile CSV (FWHM/R_peak)
        if bounce_csv_path:
            try:
                FWHM, R_peak = read_bounce_fwhm_from_csv(bounce_csv_path)
                if (FWHM is not None) and (R_peak is not None) and (FWHM > 0) and (R_peak > 0):
                    R_over_w = float(R_peak / FWHM)
            except Exception:
                pass
    if not np.isfinite(R_over_w) or R_over_w <= 0:
        return out

    w_over_R = 1.0 / R_over_w

    # For micro_size diagnostic outputs
    ell_tilde = float("nan")

    # Geometric participation factor f_geom (fraction of the S^3 wall involved in the PQ-violating event).
    if model == "bounce":
        f_geom = 1.0
    elif model == "wall":
        f_geom = w_over_R
    elif model == "patch":
        f_geom = (w_over_R**2)
    elif model == "power":
        f_geom = (w_over_R**float(power))
    elif model == "micro_size":
        # Phase II: derive the effective patch fraction from a microphysical coherence length on the wall.
        # Default: use the string length ell_s from the SFV→EW matching, converted to bounce-dimensionless units.
        # Conversion: ell_tilde = ell_micro[GeV^-1] * v_brane[GeV], where v_brane[GeV] = v_brane_over_Mpl * Mpl_reduced.
        Mpl = float(calib.get("units", {}).get("Mpl_reduced_GeV", np.nan))
        v_brane = float(calib.get("bounce", {}).get("two_field_O4", {}).get("vevs_over_Mpl", {}).get("v_brane", np.nan))
        v_brane_GeV = v_brane * Mpl if (np.isfinite(Mpl) and np.isfinite(v_brane)) else float("nan")

        # Need bubble radius in bounce-dimensionless units to compare to ell_tilde.
        if not np.isfinite(R_dimless) or R_dimless <= 0:
            # If we have w and the ratio R/w, infer R = w * (R/w).
            if np.isfinite(w_dimless) and w_dimless > 0 and np.isfinite(R_over_w) and R_over_w > 0:
                R_dimless = float(w_dimless * R_over_w)

        if (not np.isfinite(v_brane_GeV)) or v_brane_GeV <= 0 or (not np.isfinite(R_dimless)) or R_dimless <= 0:
            raise ValueError("pq_break_S_model=micro_size requires calib.units.Mpl_reduced_GeV, calib.bounce.two_field_O4.vevs_over_Mpl.v_brane, and a usable --bounce_csv with R_peak and w_FWHM.")

        ew_scales = calib.get("electroweak", {}).get("SFV_scales", {})
        ell_s = float(ew_scales.get("ell_s_GeV_inv", np.nan))

        if micro_ell_model == "ell_s":
            ell_micro = ell_s
            ell_micro_note = "ell_s"
        else:
            raise ValueError(f"Unknown --pq_break_micro_ell: {micro_ell_model}")

        if (not np.isfinite(ell_micro)) or ell_micro <= 0:
            raise ValueError(f"Invalid micro length for {ell_micro_note}: {ell_micro}")

        ell_tilde = float(ell_micro * v_brane_GeV)     # convert to bounce-dimensionless length
        ell_tilde = float(min(ell_tilde, R_dimless))    # cannot exceed bubble radius
        n_tan = int(micro_n_tan)
        if n_tan < 1:
            raise ValueError("--pq_break_micro_n_tan must be >= 1")
        f_geom = float((ell_tilde / R_dimless)**n_tan)
    else:
        raise ValueError(f"Unknown pq_break_S_model: {model}")

    S_break = float(S_E * f_geom)

    # Derived effective exponent (diagnostic): f_geom ≈ (w/R)^p_eff
    if np.isfinite(w_over_R) and w_over_R > 0 and abs(w_over_R - 1.0) > 1e-12 and np.isfinite(f_geom) and f_geom > 0:
        p_eff = float(math.log(f_geom) / math.log(w_over_R))
    else:
        p_eff = float("nan")

    # Choose M_*
    ew = calib.get("electroweak", {}).get("SFV_scales", {})
    M_brane = float(ew.get("M_brane_GeV", np.nan))
    ell_s = float(ew.get("ell_s_GeV_inv", np.nan))
    if Mstar_choice == "M_brane":
        Mstar = M_brane
    elif Mstar_choice == "M_string":
        Mstar = 1.0/ell_s if np.isfinite(ell_s) and ell_s > 0 else np.nan
    else:
        # fallback
        Mstar = M_brane

    if (not np.isfinite(Mstar)) or Mstar <= 0:
        return out

    Lambda = float(Mstar * math.exp(-S_break/4.0))

    out.update(dict(
        ok=True,
        S_E=float(S_E),
        R_over_w=float(R_over_w),
        w_over_R=float(w_over_R),
        f_geom=float(f_geom),
        p_eff=float(p_eff),
        w_dimless=float(w_dimless) if np.isfinite(w_dimless) else float("nan"),
        R_dimless=float(R_dimless) if np.isfinite(R_dimless) else float("nan"),
        ell_tilde_dimless=float(ell_tilde) if np.isfinite(ell_tilde) else float("nan"),
        micro_ell_model=str(micro_ell_model),
        micro_n_tan=int(micro_n_tan),
        S_break=float(S_break),
        Mstar=float(Mstar),
        Mstar_choice=str(Mstar_choice),
        model=str(model),
        power=float(power),
        Lambda_PQ_GeV=float(Lambda),
    ))
    return out


def wrap_angle(theta: float) -> float:
    """Map angle to (-pi, pi]."""
    return float((theta + np.pi) % (2.0*np.pi) - np.pi)


def ma_eV_from_fa(fa_GeV: float) -> float:
    """Approx QCD axion mass scaling (QCD piece), in eV."""
    return 5.7e-6 * (1.0e12 / fa_GeV)

def chi_qcd_GeV4_from_fa(fa_GeV: float) -> float:
    """Topological susceptibility estimate via chi ≈ (m_a f_a)^2 in GeV^4."""
    ma_GeV = ma_eV_from_fa(fa_GeV) * 1.0e-9  # eV -> GeV
    return float((ma_GeV * fa_GeV)**2)


def read_bounce_fwhm_from_csv(path: str):
    """Read w_FWHM (dimensionless) and R_peak from a background_profile.csv.

    The file produced by your bounce pipeline typically includes constant columns
    w_FWHM and R_peak repeated on every row. We just read the first data row.
    Returns (w_FWHM, R_peak_or_None).
    """
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            if "w_FWHM" not in row:
                raise ValueError(f"CSV missing w_FWHM column: {path}")
            w = float(row["w_FWHM"])
            rp = float(row["R_peak"]) if ("R_peak" in row and row["R_peak"] not in (None, "")) else None
            if w <= 0:
                raise ValueError(f"Non-positive w_FWHM={w} in {path}")
            return w, rp
    raise ValueError(f"No data rows found in {path}")

def minimize_theta_eff_grid(theta_bar: float, chi_qcd: float, Lambda_pq_GeV: float, delta: float,
                            n_grid: int = 20001):
    """Coarse-but-robust grid minimizer for V(x) over x=a/f_a in (-pi,pi]."""
    Lam4 = float(Lambda_pq_GeV)**4
    xs = np.linspace(-np.pi, np.pi, int(n_grid))
    V = chi_qcd*(1.0 - np.cos(theta_bar + xs)) + Lam4*(1.0 - np.cos(xs - delta))
    j = int(np.argmin(V))
    x_star = float(xs[j])
    theta_eff = wrap_angle(theta_bar + x_star)
    return x_star, theta_eff, float(V[j]), float(xs[1]-xs[0])


def minimize_theta_eff_newton(theta_bar: float, chi_qcd: float, Lambda_pq_GeV: float, delta: float,
                              x0: float = None, maxiter: int = 80, tol: float = 1e-15):
    """High-precision minimizer via Newton on dV/dx = 0.

    V(x) = chi*(1-cos(theta_bar+x)) + Lam4*(1-cos(x-delta)).
    dV/dx = chi*sin(theta_bar+x) + Lam4*sin(x-delta).

    Returns (x_star, theta_eff, Vmin, n_iter, converged).
    """
    Lam4 = float(Lambda_pq_GeV)**4
    x = wrap_angle(-theta_bar) if x0 is None else float(x0)
    converged = False
    for it in range(1, maxiter+1):
        f = chi_qcd*np.sin(theta_bar + x) + Lam4*np.sin(x - delta)
        fp = chi_qcd*np.cos(theta_bar + x) + Lam4*np.cos(x - delta)
        if abs(fp) < 1e-30:
            break
        dx = -f/fp
        x = wrap_angle(x + dx)
        if abs(dx) < tol:
            converged = True
            break
    theta_eff = wrap_angle(theta_bar + x)
    Vmin = chi_qcd*(1.0 - np.cos(theta_bar + x)) + Lam4*(1.0 - np.cos(x - delta))
    return float(x), float(theta_eff), float(Vmin), int(it), converged


def theta_eff_perturbative(theta_bar: float, chi_qcd: float, Lambda_pq_GeV: float, delta: float) -> float:
    """Small-breaking estimate: if Lam4 << chi, residual theta_eff ≈ -(Lam4/chi)*sin(-theta_bar - delta)."""
    Lam4 = float(Lambda_pq_GeV)**4
    if chi_qcd <= 0:
        return float('nan')
    return float(-(Lam4/chi_qcd) * np.sin(-theta_bar - delta))


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", required=True, help="Path to quark_geometry_best.json")
    ap.add_argument("--calib", default=None, help="Path to SFV condensed calibration JSON (optional)")
    ap.add_argument("--theta_uv", type=float, default=0.0, help="UV theta angle at matching scale (radians)")
    ap.add_argument("--phases", choices=["off","on"], default="off", help="Enable geometric phases in overlaps")
    ap.add_argument("--kL", default="0,0,0", help="kL for left-handed profiles (comma list, 3 vals)")
    ap.add_argument("--kRu", default="0,0,0", help="kR for up-sector right-handed profiles (3 vals)")
    ap.add_argument("--kRd", default="0,0,0", help="kR for down-sector right-handed profiles (3 vals)")
    ap.add_argument("--kH", type=float, default=0.0, help="k for Higgs profile")
    ap.add_argument("--kappa_a", type=float, default=1.0, help="O(1) coefficient in fa estimates from SFV scales")
    ap.add_argument("--L_a_model", default="ell_s", choices=["ell_s","bounce_FWHM","xi_heal","custom"],
                    help="Bulk axion zero-mode support length model for fa2: ell_s, xi_heal, or custom")
    ap.add_argument("--L_a_GeV_inv", type=float, default=None,
                    help="Custom L_a in GeV^-1 (used if --L_a_model custom)")
    ap.add_argument("--bounce_csv", default=None,
                    help="Bounce profile CSV (e.g. background_profile.csv) used if --L_a_model bounce_FWHM")
    ap.add_argument("--fa_choice", default="fa2", choices=["fa1","fa2","fa_phi"],
                    help="Which fa estimate to use for a* (fa_phi requires bounce v_bulk in calib)")
    ap.add_argument("--pq_break_Lambda_GeV", type=float, default=0.0,
                    help="Explicit PQ-breaking scale Lambda_PQ in GeV (0 disables quality term)")
    ap.add_argument("--pq_break_delta", type=float, default=0.0,
                    help="Phase offset delta (radians) for PQ-breaking cosine term")
    ap.add_argument("--pq_break_method", default="newton", choices=["newton","grid"],
                    help="How to minimize the axion potential when PQ breaking is enabled")
    ap.add_argument("--pq_break_model", default="fixed", choices=["none","fixed","from_bounce"],
                    help="PQ-breaking source: none disables; fixed uses --pq_break_Lambda_GeV; from_bounce derives Lambda_PQ from bounce action/geometry proxy.")
    ap.add_argument("--bounce_S_E", type=float, default=None,
                    help="Override bounce Euclidean action S_E for pq_break_model=from_bounce if calib JSON lacks it (e.g. 1078).")
    ap.add_argument("--pq_break_S_model", default="wall", choices=["bounce","wall","patch","power","micro_size"],
                    help="If --pq_break_model=from_bounce: how to scale S_break from the O(4) bounce action S_E.")
    ap.add_argument("--pq_break_power", type=float, default=1.0,
                    help="If pq_break_S_model=power: exponent p in S_break = S_E*(w/R)^p.")
    ap.add_argument("--pq_break_micro_ell", default="ell_s", choices=["ell_s"],
                    help="If pq_break_S_model=micro_size: microphysical coherence length model for the PQ-violating operator (currently: ell_s).")
    ap.add_argument("--pq_break_micro_n_tan", type=int, default=3,
                    help="If pq_break_S_model=micro_size: tangential dimensionality n on the S^3 wall (default 3, so f_geom=(ell/R)^3).")
    ap.add_argument("--pq_break_Mstar", default="M_brane", choices=["M_brane","M_string"],
                    help="If --pq_break_model=from_bounce: choose UV scale M_* for Lambda_PQ ≈ M_* exp(-S_break/4).")
    ap.add_argument("--theta_tol", type=float, default=1e-10,
                    help="Target bound for |theta_eff| when estimating Lambda_PQ(max)")
    ap.add_argument("--chi_qcd_GeV4", type=float, default=None,
                    help="Override QCD topological susceptibility chi in GeV^4 (default from m_a(f_a))")
    args = ap.parse_args()

    with open(args.geom, "r") as f:
        G = json.load(f)

    # Geometry
    yH = float(G["H"]["yH"]); sH = float(G["H"]["sH"])

    # Up: left (baseline) and right
    yL_base = np.array(G["up"]["L"]["yL_base"], dtype=float)
    sL_u    = np.array(G["up"]["L"]["sL"], dtype=float)
    ku      = float(G["up"]["ku"])
    yL_u    = scale_L(yL_base, ku, yH)

    yR_u = np.array(G["up"]["R"]["yR"], dtype=float)
    sR_u = np.array(G["up"]["R"]["sR"], dtype=float)

    # Down: left constructed from same baseline + kd + epsL + misalign angles, right from JSON
    kd    = float(G["down"]["kd"])
    epsL  = float(G["down"].get("epsL", 0.0))
    th12  = float(G["down"]["L_misalign"]["theta12"])
    th23  = float(G["down"]["L_misalign"]["theta23"])
    th13  = float(G["down"]["L_misalign"]["theta13"])

    yL_d0 = scale_L(yL_base, kd, yH)
    yL_d0 = yL_d0 + epsL
    yL_d  = misalign_L_down(yL_d0, th12, th23, th13)
    sL_d  = sL_u.copy()  # matches scan behavior (uses sL_base for both)

    yR_d = np.array(G["down"]["R"]["yR"], dtype=float)
    sR_d = np.array(G["down"]["R"]["sR"], dtype=float)

    # Phase knobs
    if args.phases == "on":
        kL  = parse_vec3(args.kL)
        kRu = parse_vec3(args.kRu)
        kRd = parse_vec3(args.kRd)
        kH  = float(args.kH)
    else:
        kL  = np.zeros(3); kRu = np.zeros(3); kRd = np.zeros(3); kH = 0.0

    if args.phases == "on":
        if np.allclose(kL, 0) and np.allclose(kRu, 0) and np.allclose(kRd, 0) and abs(kH) < 1e-15:
            print("WARNING: --phases on but all k's are zero. Overlaps remain real; CKM CPV will be absent.")
        print(f"kL={kL}, kRu={kRu}, kRd={kRd}, kH={kH}")

    # Yukawas
    Yu = Ymat_complex(yL_u, sL_u, yR_u, sR_u, yH, sH, kL, kRu, kH)
    Yd = Ymat_complex(yL_d, sL_d, yR_d, sR_d, yH, sH, kL, kRd, kH)

    # CKM from left-unitaries
    Uu, Su, _ = svd_sorted_complex(Yu)
    Ud, Sd, _ = svd_sorted_complex(Yd)
    V = Uu.conj().T @ Ud

    # Bare strong-CP angle from determinants (masses ∝ Yukawas so overall v cancels in arg)
    det_prod = np.linalg.det(Yu) * np.linalg.det(Yd)
    theta_mass = arg_det(det_prod)
    theta_bar_bare = float(args.theta_uv + theta_mass)
    theta_bar_wrapped = wrap_angle(theta_bar_bare)
    print(f"theta_bar wrapped     = {theta_bar_wrapped:+.6e} rad   (mapped to (-pi,pi])")

    

    # Pure QCD axion relaxation (no explicit PQ breaking):
    # choose x = a/f_a = wrap(-theta_bar_bare) so theta_eff = wrap(theta_bar_bare + x) -> 0 (mod 2π).
    a_over_fa_pure = wrap_angle(-theta_bar_bare)
    theta_eff = wrap_angle(theta_bar_bare + a_over_fa_pure)

    print("\n=== SFV/dSB Route-2 Strong-CP Diagnostic ===")
    print(f"phases: {args.phases}")
    print(f"theta_uv            = {args.theta_uv:+.6e} rad")
    print(f"arg det(Yu*Yd)       = {theta_mass:+.6e} rad (principal branch)")
    print(f"theta_bar (bare)     = {theta_bar_bare:+.6e} rad")
    print(f"theta_eff after axion= {theta_eff:+.6e} rad   (by minimization)")

    print("\n--- CKM diagnostics (from complex overlaps) ---")
    print(f"|Vus|,|Vcb|,|Vub|    = {abs(V[0,1]):.6g}, {abs(V[1,2]):.6g}, {abs(V[0,2]):.6g}")
    print(f"Jarlskog J           = {jarlskog(V):+.6e}")
    print(f"arg(det(V))          = {arg_det(np.linalg.det(V)):+.6e} rad")
    



    # Optional: estimate f_a from calibration pack
    if args.calib:
        with open(args.calib, "r") as f:
            C = json.load(f)
        fa = estimate_fa_from_calib(C, kappa_a=args.kappa_a, L_a_model=args.L_a_model, L_a_custom_GeV_inv=args.L_a_GeV_inv, bounce_csv_path=args.bounce_csv)
        if fa:
            print("\n--- f_a estimates from SFV scales (bulk axion candidates) ---")
            print(f"M_brane              = {fa['M_brane_GeV']:.6e} GeV")
            print(f"ell_s                = {fa['ell_s_GeV_inv']:.6e} GeV^-1")
            print(f"L_a model            = {fa['L_a_model']}  (L_a = {fa['L_a_GeV_inv']:.6e} GeV^-1)")
            print(f"fa1 ~ kappa/ell_s       = {fa['fa1_GeV']:.6e} GeV")
            print(f"fa2 ~ kappa*M*sqrt(M L_a)= {fa['fa2_GeV']:.6e} GeV")
            if fa.get("fa_phi_GeV") is not None:
                print(f"fa_phi ~ kappa*v_bulk*Mpl= {fa['fa_phi_GeV']:.6e} GeV")
            else:
                print("fa_phi                = (not available in calib)")

            # Select which f_a to use for the axion VEV printout
            if args.fa_choice == "fa1":
                fa_sel = fa["fa1_GeV"]
            elif args.fa_choice == "fa2":
                fa_sel = fa["fa2_GeV"]
            elif args.fa_choice == "fa_phi":
                if fa.get("fa_phi_GeV") is None:
                    raise ValueError("fa_choice=fa_phi selected but fa_phi_GeV is not available. Check calib bounce/units.")
                fa_sel = fa["fa_phi_GeV"]
            else:
                raise ValueError(f"Unknown fa_choice: {args.fa_choice}")

            # Required axion VEV to relax theta_eff -> 0 (mod 2π)
            a_over_fa = -theta_bar_wrapped
            a_star = a_over_fa * fa_sel

            print("\n--- Axion relaxation (required VEV; using fa_choice) ---")
            print(f"fa_choice             = {args.fa_choice}  (fa = {fa_sel:.6e} GeV)")
            print(f"(a*/f_a)              = {a_over_fa:+.6e}   (dimensionless)")
            print(f"a*                    = {a_star:+.6e} GeV")
            print(f"(2π f_a)              = {2*np.pi*fa_sel:.6e} GeV")

            # --- PQ quality diagnostic (explicit PQ breaking can shift the minimum away from theta_eff=0) ---
            chi_qcd = float(args.chi_qcd_GeV4) if (args.chi_qcd_GeV4 is not None) else chi_qcd_GeV4_from_fa(fa_sel)
            Lambda_max = float((max(args.theta_tol, 0.0) * chi_qcd)**0.25) if args.theta_tol > 0 else float("nan")

            # Decide which PQ-breaking Lambda_PQ to use (fixed vs derived proxy).
            pq_lambda_eff = float(args.pq_break_Lambda_GeV)
            pq_src_note = "fixed (--pq_break_Lambda_GeV)"
            pq_proxy = None
            if args.pq_break_model == "none":
                pq_lambda_eff = 0.0
                pq_src_note = "none"
            elif args.pq_break_model == "from_bounce":
                pq_proxy = estimate_pq_break_from_bounce(
                    C,
                    model=args.pq_break_S_model,
                    power=args.pq_break_power,
                    Mstar_choice=args.pq_break_Mstar,
                    bounce_csv_path=args.bounce_csv,
                    S_E_override=args.bounce_S_E,
                    micro_ell_model=args.pq_break_micro_ell,
                    micro_n_tan=args.pq_break_micro_n_tan,
                )
                if pq_proxy.get("ok"):
                    pq_lambda_eff = float(pq_proxy["Lambda_PQ_GeV"])
                    pq_src_note = f"from_bounce:{pq_proxy['model']}"
                else:
                    raise ValueError("pq_break_model=from_bounce selected but could not infer bounce inputs. Provide --bounce_S_E (e.g. 1078) and ensure --bounce_csv points to a profile CSV with columns w_FWHM and R_peak (or add bounce.two_field_O4.{S_E,geometry.R_over_w} into calib).")
            print("\n--- PQ quality diagnostic ---")
            print(f"chi_QCD               = {chi_qcd:.6e} GeV^4")
            print(f"theta_tol             = {args.theta_tol:.3e}")
            print(f"Lambda_PQ(max) ~       = {Lambda_max:.6e} GeV  (rough, O(1) phase)")

            if pq_proxy is not None and pq_proxy.get("ok"):
                print("\n--- PQ-breaking proxy from bounce (heuristic / micro_size in v8) ---")
                print(f"S_E (O4 bounce)        = {pq_proxy['S_E']:.3f}")
                print(f"R_over_w               = {pq_proxy['R_over_w']:.6g}  (w/R={pq_proxy['w_over_R']:.6g})")
                print(f"S_break(model={pq_proxy['model']}, p={pq_proxy['power']:.3g}) = {pq_proxy['S_break']:.3f}")
                print(f"f_geom                = {pq_proxy.get('f_geom', float('nan')):.6g}  (p_eff={pq_proxy.get('p_eff', float('nan')):.3g})")
                if str(pq_proxy.get('model')) == 'micro_size':
                    print(f"micro ell model        = {pq_proxy.get('micro_ell_model')}  n_tan={pq_proxy.get('micro_n_tan')}")
                    print(f"R_dimless              = {pq_proxy.get('R_dimless', float('nan')):.6g}  ell_tilde={pq_proxy.get('ell_tilde_dimless', float('nan')):.6g}")
                print(f"M_* choice             = {pq_proxy['Mstar_choice']}  (M*={pq_proxy['Mstar']:.3e} GeV)")
                print(f"Lambda_PQ(proxy)        = {pq_proxy['Lambda_PQ_GeV']:.6e} GeV")
            # Translate a Lambda scale to a semiclassical suppression S_break via Lambda ~ M_* exp(-S/4)
            Mstar = float(fa.get("M_brane_GeV", np.nan))
            if np.isfinite(Mstar) and (args.theta_tol > 0) and (Lambda_max > 0):
                Sreq_max = 4.0*np.log(Mstar / Lambda_max)
                print(f"S_break needed (to reach Lambda_PQ(max) from M*) ~ {Sreq_max:.3f}  (M*={Mstar:.3e} GeV)")
            if pq_lambda_eff > 0:
                # High-precision solve for the shifted minimum.
                if args.pq_break_method == "newton":
                    x_star, theta_eff_res, _Vmin, n_it, ok = minimize_theta_eff_newton(
                        theta_bar_wrapped, chi_qcd, pq_lambda_eff, args.pq_break_delta
                    )
                    grid_dx = None
                else:
                    x_star, theta_eff_res, _Vmin, grid_dx = minimize_theta_eff_grid(
                        theta_bar_wrapped, chi_qcd, pq_lambda_eff, args.pq_break_delta
                    )
                    n_it, ok = None, None
                print("\n--- With explicit PQ breaking term ---")
                print(f"Lambda_PQ             = {pq_lambda_eff:.6e} GeV  (source: {pq_src_note})")
                print(f"delta                 = {args.pq_break_delta:+.6e} rad")
                print(f"(a*/f_a) minimizing V  = {x_star:+.6e}")
                print(f"theta_eff (residual)  = {theta_eff_res:+.6e} rad")
                # Print expected scaling in the small-breaking regime.
                Lam4 = float(pq_lambda_eff)**4
                ratio = Lam4/chi_qcd if chi_qcd > 0 else float('nan')
                theta_pert = theta_eff_perturbative(theta_bar_wrapped, chi_qcd, pq_lambda_eff, args.pq_break_delta)
                print(f"Lam^4/chi_QCD          = {ratio:.3e}")
                print(f"theta_eff (perturb.)  = {theta_pert:+.6e} rad")
                if args.pq_break_method == "newton":
                    print(f"minimizer              = newton  (iters={n_it}, converged={ok})")
                else:
                    print(f"minimizer              = grid    (dx~{grid_dx:.3e} rad)")
                if np.isfinite(Mstar) and (pq_lambda_eff > 0):
                    Sreq = 4.0*np.log(Mstar / pq_lambda_eff)
                    print(f"S_break implied by Lambda_PQ ~ M* exp(-S/4): {Sreq:.3f}")

            print("\n--- Axion relaxation (comparison) ---")
            print(f"a* (using fa1)        = {a_over_fa * fa['fa1_GeV']:+.6e} GeV")
            print(f"a* (using fa2)        = {a_over_fa * fa['fa2_GeV']:+.6e} GeV")
            if fa.get("fa_phi_GeV") is not None:
                print(f"a* (using fa_phi)     = {a_over_fa * fa['fa_phi_GeV']:+.6e} GeV")
        else:
            print("\n(calib provided, but electroweak.SFV_scales.{M_brane_GeV,ell_s_GeV_inv} not found)")

if __name__ == "__main__":
    main()