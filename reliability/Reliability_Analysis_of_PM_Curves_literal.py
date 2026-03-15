# Reliability_Analysis_of_PM_Curves_literal.py
"""
Monte Carlo reliability driver for RC column P-M interaction.

Outputs written to <project_root>/static/plots/:
  ReliabilityIndex_combined.png   -- Beta (reliability index) heatmap
  FailureProbability_combined.png -- Pf heatmap
  Hist_fck.png / Hist_fy.png / Hist_p.png -- distribution histograms
  Fragility_combined.xlsx         -- Pf grid table

Call main(...) from app.py or run directly as a script.
"""

import os
import time
import uuid
import sys
import traceback
from pathlib import Path
import importlib.util

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load a sibling module by file path (avoids import-path issues)
# ─────────────────────────────────────────────────────────────────────────────
def _load_module_from_path(module_name, filepath):
    filepath = Path(filepath).resolve()
    if not filepath.exists():
        raise FileNotFoundError(f"Module file not found: {filepath}")
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, str(filepath))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ─────────────────────────────────────────────────────────────────────────────
# Worker (must be top-level for multiprocessing pickling)
# ─────────────────────────────────────────────────────────────────────────────
def _worker_compute_curve(task):
    """
    task = (gen_filepath, fck, fy, Es, p, n, nbars_total, key)
    Returns (key, MuR, PuR) or (key, None, None, traceback_str)
    """
    gen_filepath, fck, fy, Es, p, n, nbars_total, key = task
    try:
        mod_name = f"genmod_{uuid.uuid4().hex}"
        genmod   = _load_module_from_path(mod_name, gen_filepath)
        genfunc  = getattr(genmod, "generate_PM_True_Capacity", None)
        if genfunc is None:
            raise AttributeError("generate_PM_True_Capacity not found in generator module.")
        MuR, PuR, xu_by_D, KeyPoints = genfunc(float(fck), float(fy), float(Es),
                                                float(p), int(n), int(nbars_total))
        return (key, np.asarray(MuR, dtype=float), np.asarray(PuR, dtype=float))
    except Exception:
        return (key, None, None, traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# Histogram helper
# ─────────────────────────────────────────────────────────────────────────────
def _save_histogram(samples, label, unit, out_path, color='steelblue'):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(samples, bins=30, color=color, edgecolor='white', alpha=0.85)
    ax.axvline(np.mean(samples), color='crimson', linewidth=1.5,
               linestyle='--', label=f'mean = {np.mean(samples):.2f}')
    ax.set_xlabel(f'{label} ({unit})')
    ax.set_ylabel('Count')
    ax.set_title(f'Sampled distribution — {label}')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def main(N=300,
         fck_nom=30.0,  fck_std=5.0,
         fy_nom=415.0,  fy_std=25.0,
         p_nom=2.0,     p_std=0.0,
         n=3,           nbars_total=8,
         Es=200000.0,
         seed=None,
         gen_dir=None,
         mu_range=None, pu_range=None,
         use_parallel=False,
         verbose=True):
    """
    Run Monte Carlo reliability analysis and write output plots + Excel.

    Parameters
    ----------
    N            : int   – number of Monte Carlo samples
    fck_nom/std  : float – mean and std of concrete fck (MPa); std=0 → fixed
    fy_nom/std   : float – mean and std of steel fy (MPa);    std=0 → fixed
    p_nom/std    : float – mean and std of rebar ratio p (%); std=0 → fixed
    n            : int   – number of bar rows in cross-section
    nbars_total  : int   – total number of bars
    Es           : float – elastic modulus of steel (MPa)
    seed         : int|None – RNG seed for reproducibility
    gen_dir      : path  – folder containing generator scripts (default: same dir as this file)
    mu_range     : array – custom Mu grid (nondim); None → auto
    pu_range     : array – custom Pu grid (nondim); None → auto
    use_parallel : bool  – use multiprocessing (False = safer inside Flask)
    verbose      : bool  – print progress

    Returns
    -------
    dict with keys: elapsed_s, fragility_file, beta_image, pf_image,
                    hist_images, Pf_grid, MU_grid, PU_grid
    """

    t0 = time.time()

    # ── Output directory ─────────────────────────────────────────────────────
    _THIS_FILE   = Path(__file__).resolve()
    PROJECT_ROOT = _THIS_FILE.parents[1] if len(_THIS_FILE.parents) >= 2 else _THIS_FILE.parent
    OUT = PROJECT_ROOT / "static" / "plots"
    OUT.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[reliability] OUT = {OUT.resolve()}")

    # ── Generator path ───────────────────────────────────────────────────────
    if gen_dir is None:
        gen_dir = _THIS_FILE.parent
    gen_true_path = Path(gen_dir) / "generate_PM_True_Capacity_literal.py"
    if not gen_true_path.exists():
        raise FileNotFoundError(f"Cannot find generator: {gen_true_path}")

    # ── Sample random variables ───────────────────────────────────────────────
    rng = np.random.default_rng(seed)

    fck_samples = fck_nom + (fck_std * rng.standard_normal(N) if fck_std > 0 else np.zeros(N))
    fy_samples  = fy_nom  + (fy_std  * rng.standard_normal(N) if fy_std  > 0 else np.zeros(N))
    p_samples   = p_nom   + (p_std   * rng.standard_normal(N) if p_std   > 0 else np.zeros(N))

    # Clamp to physically meaningful bounds
    fck_samples = np.clip(fck_samples, 5.0, None)
    fy_samples  = np.clip(fy_samples,  50.0, None)
    p_samples   = np.clip(p_samples,   0.1,  8.0)

    # ── Save histograms ───────────────────────────────────────────────────────
    hist_images = []
    if fck_std > 0:
        p_fck = OUT / "Hist_fck.png"
        _save_histogram(fck_samples, 'f_ck', 'MPa', p_fck, color='steelblue')
        hist_images.append(str(p_fck))
    if fy_std > 0:
        p_fy = OUT / "Hist_fy.png"
        _save_histogram(fy_samples, 'f_y', 'MPa', p_fy, color='darkorange')
        hist_images.append(str(p_fy))
    if p_std > 0:
        p_p = OUT / "Hist_p.png"
        _save_histogram(p_samples, 'p', '%', p_p, color='seagreen')
        hist_images.append(str(p_p))

    # ── Build cache keys (quantize to reduce redundant curve computations) ────
    fck_bin, fy_bin, p_bin = 0.1, 1.0, 0.05

    def _q(v, b):
        return float(round(v / b) * b)

    keys       = []
    unique_map = {}
    for i in range(N):
        key = (_q(fck_samples[i], fck_bin),
               _q(fy_samples[i],  fy_bin),
               _q(p_samples[i],   p_bin))
        keys.append(key)
        if key not in unique_map:
            unique_map[key] = (float(fck_samples[i]),
                               float(fy_samples[i]),
                               float(p_samples[i]))

    unique_keys = list(unique_map.keys())
    if verbose:
        print(f"[reliability] N={N}, unique keys={len(unique_keys)}")

    # ── Build task list ───────────────────────────────────────────────────────
    tasks = []
    for key in unique_keys:
        rep_fck, rep_fy, rep_p = unique_map[key]
        tasks.append((str(gen_true_path), rep_fck, rep_fy, Es, rep_p,
                      int(n), int(nbars_total), key))

    # ── Compute unique P-M curves (serial; parallel optional) ─────────────────
    results_cache = {}
    if use_parallel:
        import multiprocessing as mp
        nprocs = max(1, mp.cpu_count() - 1)
        if verbose:
            print(f"[reliability] Parallel: {len(tasks)} curves on {nprocs} procs ...")
        with mp.Pool(processes=nprocs) as pool:
            for res in pool.imap_unordered(_worker_compute_curve, tasks):
                _store_result(res, results_cache, verbose)
    else:
        if verbose:
            print(f"[reliability] Serial: computing {len(tasks)} unique curves ...")
        for task in tasks:
            res = _worker_compute_curve(task)
            _store_result(res, results_cache, verbose)

    # ── Auto-detect grid dimensions from generator output ─────────────────────
    if mu_range is None:
        mu_range = np.linspace(0.0, 10.0, 50)
    if pu_range is None:
        pu_range = np.linspace(0.0, 50.0, 50)

    all_Mu_max = all_Pu_max = 0.0
    any_curve  = False
    for v in results_cache.values():
        if v is None:
            continue
        MuR, PuR = v
        cur_mu = np.nanmax(np.abs(MuR)) if len(MuR) else 0
        cur_pu = np.nanmax(np.abs(PuR)) if len(PuR) else 0
        if np.isfinite(cur_mu) and cur_mu > all_Mu_max:
            all_Mu_max = cur_mu
        if np.isfinite(cur_pu) and cur_pu > all_Pu_max:
            all_Pu_max = cur_pu
        any_curve = True

    if any_curve and (all_Mu_max > 10 * max(1, np.nanmax(mu_range))
                      or all_Pu_max > 10 * max(1, np.nanmax(pu_range))):
        MU_vals = np.linspace(0, all_Mu_max * 1.05, len(mu_range))
        PU_vals = np.linspace(0, all_Pu_max * 1.05, len(pu_range))
        if verbose:
            print(f"[reliability] Dimensional grid: Mu 0..{MU_vals[-1]:.3g}, "
                  f"Pu 0..{PU_vals[-1]:.3g}")
    else:
        MU_vals = mu_range
        PU_vals = pu_range
        if verbose:
            print("[reliability] Non-dimensional grid.")

    MU, PU = np.meshgrid(MU_vals, PU_vals)
    failure_counts = np.zeros_like(MU, dtype=int)

    # ── Count failures across samples ─────────────────────────────────────────
    PU_flat = PU.ravel()
    MU_flat = MU.ravel()

    for i, key in enumerate(keys):
        cv = results_cache.get(key)
        if cv is None:
            continue
        MuR, PuR = cv
        valid = np.isfinite(MuR) & np.isfinite(PuR)
        if valid.sum() < 2:
            continue
        Pu_v = PuR[valid]; Mu_v = MuR[valid]
        order    = np.argsort(Pu_v)
        try:
            f_interp = interp1d(Pu_v[order], Mu_v[order],
                                bounds_error=False,
                                fill_value=(Mu_v[order][0], Mu_v[order][-1]))
            failure_counts.flat += (MU_flat > f_interp(PU_flat)).astype(int)
        except Exception as exc:
            if verbose:
                print(f"[reliability] interpolation failed for sample {i}: {exc}")

    # ── Failure probability grid ──────────────────────────────────────────────
    Pf = failure_counts.astype(float) / max(1, N)

    # ── Excel export ──────────────────────────────────────────────────────────
    df = pd.DataFrame(Pf, index=np.round(PU[:, 0], 4), columns=np.round(MU[0, :], 4))
    frag_path = OUT / "Fragility_combined.xlsx"
    df.to_excel(frag_path, index_label="Pu \\ Mu")

    # ── Beta heatmap ──────────────────────────────────────────────────────────
    eps    = 1e-12
    Beta   = norm.ppf(1 - np.clip(Pf, eps, 1 - eps))

    fig, ax = plt.subplots(figsize=(8, 6))
    ctf = ax.contourf(MU, PU, Beta, levels=20, cmap='viridis')
    plt.colorbar(ctf, ax=ax, label='β (Reliability Index)')
    ax.set_xlabel('Moment (nondim M/f_ck·b·D²)')
    ax.set_ylabel('Axial load (nondim P/f_ck·b·D)')
    ax.set_title(f'Reliability Index β — N={N} samples')
    ax.grid(True, alpha=0.3)
    beta_path = OUT / "ReliabilityIndex_combined.png"
    fig.savefig(beta_path, dpi=200)
    plt.close(fig)

    # ── Pf heatmap ────────────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ctf2 = ax2.contourf(MU, PU, Pf, levels=20, cmap='Reds')
    plt.colorbar(ctf2, ax=ax2, label='Probability of Failure Pf')
    ax2.set_xlabel('Moment (nondim)')
    ax2.set_ylabel('Axial load (nondim)')
    ax2.set_title(f'Failure Probability Pf — N={N} samples')
    ax2.grid(True, alpha=0.3)
    pf_path = OUT / "FailureProbability_combined.png"
    fig2.savefig(pf_path, dpi=200)
    plt.close(fig2)

    elapsed = time.time() - t0
    if verbose:
        print(f"[reliability] Done in {elapsed:.2f}s")
        for p in [beta_path, pf_path, frag_path] + [Path(h) for h in hist_images]:
            print(f"  → {p.name}")

    return {
        'elapsed_s'     : elapsed,
        'fragility_file': str(frag_path),
        'beta_image'    : str(beta_path),
        'pf_image'      : str(pf_path),
        'hist_images'   : hist_images,
        'Pf_grid'       : Pf,
        'MU_grid'       : MU,
        'PU_grid'       : PU,
    }


def _store_result(res, cache, verbose):
    """Helper to unpack a worker result tuple into the cache dict."""
    if len(res) == 3:
        key, MuR, PuR = res
        cache[key] = (np.asarray(MuR, dtype=float), np.asarray(PuR, dtype=float)) \
                     if MuR is not None else None
    else:
        key, _, _, tb = res
        cache[key] = None
        if verbose:
            print(f"[reliability] Worker failed for key {key}:\n{tb}")


if __name__ == "__main__":
    main(N=200, seed=42)
