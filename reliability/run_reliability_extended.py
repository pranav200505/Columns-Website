# -*- coding: utf-8 -*-
"""
Extended Monte Carlo reliability analysis for RC columns.

Supports uncertainty in:
  - fck  (concrete compressive strength)
  - fy   (steel yield strength)
  - p    (reinforcement percentage) — geometric uncertainty
  - D    (column depth / diameter)  — geometric uncertainty
  - b    (column width, rectangular only)

Outputs (written to static/plots/):
  - ReliabilityIndex_combined.png   (Beta heatmap over Mu-Pu space)
  - Fragility_combined.xlsx         (Pf grid)
  - histogram_<param>.png for each uncertain parameter
"""

import os
import time
import uuid
import sys
import traceback
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Load the True Capacity generator (nondimensional IS456 approach)
# ---------------------------------------------------------------------------
def _get_gen_func():
    here = Path(__file__).resolve().parent
    gen_path = here / "generate_PM_True_Capacity_literal.py"
    if not gen_path.exists():
        raise FileNotFoundError(f"Generator not found: {gen_path}")
    import importlib.util
    spec = importlib.util.spec_from_file_location("gen_true_cap", str(gen_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.generate_PM_True_Capacity


# ---------------------------------------------------------------------------
# Single-sample PM curve (nondimensional IS456 approach)
# Returns (MuR_arr, PuR_arr) or (None, None) on failure
# ---------------------------------------------------------------------------
def _compute_one_curve(gen_func, fck, fy, Es, p, n, nbars_total):
    try:
        p = max(0.1, float(p))
        n = max(2, int(round(n)))
        nbars_total = max(4, int(round(nbars_total)))
        MuR, PuR, _, _ = gen_func(float(fck), float(fy), float(Es), p, n, nbars_total)
        MuR = np.asarray(MuR, dtype=float)
        PuR = np.asarray(PuR, dtype=float)
        return MuR, PuR
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_extended(
    N=200,
    # Material stats
    fck_nom=30.0, fck_cov=0.18, fck_vary=True,
    fy_nom=415.0, fy_cov=0.06, fy_vary=True,
    # Geometry stats (used for nondim scaling only; affect p if Ast uncertain)
    p_nom=2.0, p_cov=0.10, p_vary=False,
    # Reinforcement rows / layout (kept fixed for now)
    n_rows=3, nbars_total=8,
    # ES (always fixed)
    Es=200000.0,
    # RNG
    seed=None,
    # Axis grid resolution
    nmu=50, npu=50,
    # Output folder (defaults to project static/plots)
    out_dir=None,
    verbose=True,
):
    """
    Run extended Monte Carlo reliability analysis.

    Parameters
    ----------
    N : int
        Monte Carlo samples.
    fck_nom, fck_cov : float
        Nominal (mean) fck and coefficient of variation.
    fck_vary : bool
        Whether fck is treated as random.
    fy_nom, fy_cov : float
        Nominal fy and CoV.
    fy_vary : bool
        Whether fy is treated as random.
    p_nom, p_cov : float
        Nominal reinforcement % and CoV.
    p_vary : bool
        Whether reinforcement % is treated as random.
    n_rows : int
        Number of bar rows (fixed).
    nbars_total : int
        Total number of bars (fixed).
    Es : float
        Steel modulus (always fixed).
    seed : int or None
        RNG seed.
    nmu, npu : int
        Grid resolution for heatmap.
    out_dir : path-like or None
        Output directory. Defaults to <project>/static/plots.
    verbose : bool

    Returns
    -------
    dict with file paths and summary statistics.
    """
    t0 = time.time()

    # --- Setup output directory ---
    if out_dir is None:
        _here = Path(__file__).resolve().parent
        project_root = _here.parent
        out_dir = project_root / "static" / "plots"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[reliability] OUT: {out_dir.resolve()}")

    # --- Load generator ---
    gen_func = _get_gen_func()

    # --- RNG ---
    rng = np.random.default_rng(seed)

    # --- Sample random variables ---
    fck_std = fck_nom * fck_cov if fck_vary else 0.0
    fy_std  = fy_nom  * fy_cov  if fy_vary  else 0.0
    p_std   = p_nom   * p_cov   if p_vary   else 0.0

    fck_samples = np.clip(fck_nom + fck_std * rng.standard_normal(N), 5.0, None)
    fy_samples  = np.clip(fy_nom  + fy_std  * rng.standard_normal(N), 50.0, None)
    p_samples   = np.clip(p_nom   + p_std   * rng.standard_normal(N), 0.1, 8.0)

    if verbose:
        active = []
        if fck_vary: active.append(f"fck(CoV={fck_cov:.2f})")
        if fy_vary:  active.append(f"fy(CoV={fy_cov:.2f})")
        if p_vary:   active.append(f"p(CoV={p_cov:.2f})")
        print(f"[reliability] N={N}, random vars: {', '.join(active) if active else 'NONE (deterministic)'}")

    # --- Generate PM curves for each sample ---
    all_curves = []
    failed = 0
    for i in range(N):
        MuR, PuR = _compute_one_curve(
            gen_func,
            fck=fck_samples[i],
            fy=fy_samples[i],
            Es=Es,
            p=p_samples[i],
            n=n_rows,
            nbars_total=nbars_total
        )
        if MuR is None or len(MuR) < 2:
            failed += 1
            all_curves.append(None)
        else:
            all_curves.append((MuR, PuR))

    valid_curves = [(mu, pu) for c in all_curves if c is not None for mu, pu in [c]]
    if verbose:
        print(f"[reliability] Valid curves: {len(valid_curves)}/{N} (failed={failed})")

    if len(valid_curves) == 0:
        raise RuntimeError("All Monte Carlo samples failed — cannot generate reliability results.")

    # --- Build grid from curve ranges ---
    all_mu = np.concatenate([c[0] for c in valid_curves])
    all_pu = np.concatenate([c[1] for c in valid_curves])
    mu_max = float(np.nanpercentile(np.abs(all_mu), 99)) * 1.05
    pu_max = float(np.nanpercentile(np.abs(all_pu), 99)) * 1.05
    if mu_max == 0: mu_max = 0.5
    if pu_max == 0: pu_max = 1.5

    MU_vals = np.linspace(0.0, mu_max, nmu)
    PU_vals = np.linspace(0.0, pu_max, npu)
    MU, PU = np.meshgrid(MU_vals, PU_vals)

    # --- Count failures at each grid point ---
    failure_counts = np.zeros_like(MU, dtype=int)
    sample_count   = np.zeros_like(MU, dtype=int)

    PU_flat = PU.ravel()
    MU_flat = MU.ravel()

    for curve in all_curves:
        if curve is None:
            continue
        MuR, PuR = curve
        valid = np.isfinite(MuR) & np.isfinite(PuR)
        if np.sum(valid) < 2:
            continue
        Pu_v = PuR[valid]
        Mu_v = MuR[valid]
        order = np.argsort(Pu_v)
        try:
            f_interp = interp1d(
                Pu_v[order], Mu_v[order],
                bounds_error=False,
                fill_value=(Mu_v[order][0], Mu_v[order][-1])
            )
            Mu_cap = f_interp(PU_flat)
            fail_mask = MU_flat > Mu_cap
            failure_counts.flat += fail_mask.astype(int)
            sample_count.flat   += 1
        except Exception:
            continue

    # --- Pf and Beta ---
    denom = np.maximum(sample_count, 1)
    Pf = failure_counts.astype(float) / denom
    eps = 1e-12
    Pf_clip = np.clip(Pf, eps, 1 - eps)
    Beta = norm.ppf(1 - Pf_clip)

    # --- Save fragility Excel ---
    df = pd.DataFrame(Pf, index=np.round(PU[:, 0], 4), columns=np.round(MU[0, :], 4))
    df.index.name = "Pu (nondim)"
    df.columns.name = "Mu (nondim)"
    frag_path = out_dir / "Fragility_combined.xlsx"
    df.to_excel(frag_path)
    if verbose:
        print(f"[reliability] Saved fragility: {frag_path.name}")

    # --- Save Beta heatmap ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ctf = ax.contourf(MU, PU, Beta, levels=20, cmap='viridis')
    cbar = plt.colorbar(ctf, ax=ax, label='Reliability Index β')
    # overlay mean PM curve as reference
    mean_fck = fck_nom; mean_fy = fy_nom; mean_p = p_nom
    try:
        MuR_m, PuR_m = _compute_one_curve(gen_func, mean_fck, mean_fy, Es, mean_p, n_rows, nbars_total)
        if MuR_m is not None:
            ax.plot(MuR_m, PuR_m, 'w--', linewidth=1.5, label='Mean PM curve')
            ax.legend(fontsize=8)
    except Exception:
        pass
    ax.set_xlabel('Moment (non-dimensional Mu/fck·b·D²)', fontsize=10)
    ax.set_ylabel('Axial load (non-dimensional Pu/fck·b·D)', fontsize=10)
    ax.set_title('Reliability Index β  —  Monte Carlo Analysis', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add annotation box with simulation info
    uncertain_str = []
    if fck_vary: uncertain_str.append(f"fck CoV={fck_cov:.0%}")
    if fy_vary:  uncertain_str.append(f"fy CoV={fy_cov:.0%}")
    if p_vary:   uncertain_str.append(f"p CoV={p_cov:.0%}")
    info = f"N={N} samples\n" + ("\n".join(uncertain_str) if uncertain_str else "Deterministic")
    ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=8,
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    beta_path = out_dir / "ReliabilityIndex_combined.png"
    fig.savefig(beta_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    if verbose:
        print(f"[reliability] Saved beta map: {beta_path.name}")

    # --- Save per-variable histograms ---
    hist_files = []
    params_to_plot = []
    if fck_vary:
        params_to_plot.append(("fck (MPa)", fck_samples, fck_nom, fck_std))
    if fy_vary:
        params_to_plot.append(("fy (MPa)", fy_samples, fy_nom, fy_std))
    if p_vary:
        params_to_plot.append(("p (%)", p_samples, p_nom, p_std))

    for label, samples, mean, std in params_to_plot:
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        ax2.hist(samples, bins=30, density=True, color='steelblue', alpha=0.7, edgecolor='white')
        # overlay normal pdf
        xs = np.linspace(samples.min(), samples.max(), 200)
        ax2.plot(xs, norm.pdf(xs, mean, max(std, 1e-6)), 'r-', linewidth=2, label='Normal PDF')
        ax2.axvline(mean, color='k', linestyle='--', linewidth=1.5, label=f'Mean={mean:.1f}')
        ax2.set_xlabel(label)
        ax2.set_ylabel('Density')
        ax2.set_title(f'Sampled distribution: {label}')
        ax2.legend(fontsize=8)
        hname = f"hist_{label.split()[0].lower()}_{uuid.uuid4().hex[:8]}.png"
        hpath = out_dir / hname
        fig2.savefig(hpath, dpi=120, bbox_inches='tight')
        plt.close(fig2)
        hist_files.append(hname)

    # --- Summary statistics ---
    beta_vals = Beta[np.isfinite(Beta)]
    elapsed = time.time() - t0
    summary = {
        'N': N,
        'valid_curves': len(valid_curves),
        'failed': failed,
        'beta_min': float(np.nanmin(beta_vals)) if len(beta_vals) else None,
        'beta_max': float(np.nanmax(beta_vals)) if len(beta_vals) else None,
        'beta_mean': float(np.nanmean(beta_vals)) if len(beta_vals) else None,
        'elapsed_s': elapsed,
        'beta_image': beta_path.name,
        'fragility_file': frag_path.name,
        'hist_files': hist_files,
    }
    if verbose:
        print(f"[reliability] Done in {elapsed:.2f}s. β range: "
              f"{summary['beta_min']:.2f} – {summary['beta_max']:.2f}")
    return summary
