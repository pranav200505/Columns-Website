# -*- coding: utf-8 -*-
"""
Literal line-by-line translation of MATLAB function generate_PM_True_Capacity.m.
NOTE: The original MATLAB file contained "..." placeholders, meaning sections are incomplete.
These are marked with comments in this translation. No assumptions/modifications added.
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.interpolate import interp1d

# Optional Numba integration (safe)
try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False
import matplotlib
matplotlib.use("Agg")  # force headless, non-GUI backend

# -----------------------------------------------------------------------------
# Numba-accelerated inner loop: compute Cs, Ms and es_bottom for a given xu
# (this mirrors the inner `for j in range(n):` loop in the original translation)
# -----------------------------------------------------------------------------
def _compute_CsMs_py(xu, y_by_D, k, p, n, fc_mean, Es, fy, ey):
    """
    Pure-Python fallback implementation.
    Returns (Cs, Ms, es_bottom)
    """
    Cs = 0.0
    Ms = 0.0
    es_bottom = 0.0
    for j in range(n):
        if xu <= 1:
            es = 0.0035 * ((xu - 0.5 + y_by_D[j]) / xu)
        else:
            es = 0.002 * (1 + (y_by_D[j] - 1.0/14.0) / (xu - 3.0/7.0))
        if j == 0:
            es_bottom = es
        fc = (0.67 * fc_mean) if (es > 0) else 0.0
        if abs(es) < ey:
            fs = Es * es
        else:
            fs = np.sign(es) * fy
        Cs = Cs + (fs - fc) * k[j] * p / 100.0
        Ms = Ms + (fs - fc) * k[j] * p * y_by_D[j] / 100.0
    return Cs, Ms, es_bottom

if _NUMBA_AVAILABLE:
    # Numba-compatible implementation (typed arithmetic, no NumPy methods inside)
    @njit
    def _compute_CsMs_numba(xu, y_by_D, k, p, n, fc_mean, Es, fy, ey):
        Cs = 0.0
        Ms = 0.0
        es_bottom = 0.0
        for j in range(n):
            if xu <= 1.0:
                es = 0.0035 * ((xu - 0.5 + y_by_D[j]) / xu)
            else:
                es = 0.002 * (1.0 + (y_by_D[j] - 1.0/14.0) / (xu - 3.0/7.0))
            if j == 0:
                es_bottom = es
            if es > 0.0:
                fc = 0.67 * fc_mean
            else:
                fc = 0.0
            if abs(es) < ey:
                fs = Es * es
            else:
                if es >= 0.0:
                    fs = fy
                else:
                    fs = -fy
            Cs = Cs + (fs - fc) * k[j] * p / 100.0
            Ms = Ms + (fs - fc) * k[j] * p * y_by_D[j] / 100.0
        return Cs, Ms, es_bottom

    # alias used by the rest of the code
    _compute_CsMs = _compute_CsMs_numba
else:
    _compute_CsMs = _compute_CsMs_py

# -----------------------------------------------------------------------------
# Main generator function (preserves original signature and logic)
# -----------------------------------------------------------------------------
def generate_PM_True_Capacity(fc_mean, fy, Es, p, n, nbars_total):
    """
    Literal line-by-line translation of MATLAB function generate_PM_True_Capacity.m.
    NOTE: The original MATLAB file contained "..." placeholders, meaning sections are incomplete.
    These are marked with comments in this translation. No assumptions/modifications added.
    """

    # Constants
    EffCover_by_D = 0.1
    xu_by_D_max = 10
    step = 0.01
    ey = fy / Es  # steel yield strain

    # --- Generate y_by_D values based on n ---
    y_by_D = np.linspace(-0.5 + EffCover_by_D, 0.5 - EffCover_by_D, n)

    # --- Calculate k values ---
    k = np.zeros(n)
    middle_indices = list(range(1, n-1))  # MATLAB 2:n-1 -> Python 1..n-2
    bars_used = 0

    if n > 2:
        for i in middle_indices:
            k[i] = 2
            bars_used = bars_used + 2
    remaining_bars = nbars_total - bars_used
    k[0] = remaining_bars / 2
    k[-1] = remaining_bars / 2
    k = k / nbars_total

    # Initialize output arrays
    MuR = []
    PuR = []
    xu_by_D = []

    # Step 2: Find xu_min_by_D (pure compression equilibrium)
    xu_min_by_D = 0.001
    increment = 0.0001
    tolerance = 0.001
    Pu_trial = -10

    # The following while loop uses the same algorithm as the original.
    while abs(Pu_trial) > tolerance:
        Cc = 0.54259 * xu_min_by_D * fc_mean
        Mc = Cc * (0.5 - 0.416 * xu_min_by_D)
        Cs = 0
        Ms = 0

        # Replace inner loop with the compiled/fallback helper
        Cs_tmp, Ms_tmp, _ = _compute_CsMs(xu_min_by_D, y_by_D, k, p, n, fc_mean, Es, fy, ey)
        Cs = Cs + Cs_tmp
        Ms = Ms + Ms_tmp

        Pu_trial = Cc + Cs

        if Pu_trial < 0:
            xu_min_by_D = xu_min_by_D + increment
        else:
            xu_min_by_D = xu_min_by_D - increment / 2
            increment = increment / 2

    # Step 3: Generate P-M curve and capture Balanced Failure
    first_point = True
    KeyPoints = {}

    xu = xu_min_by_D
    while xu <= xu_by_D_max + 1e-12:
        g = 16 / ((7 * xu) - 3)**2
        if xu <= 1:
            Cc = 0.54259 * xu * fc_mean
            Mc = Cc * (0.5 - 0.416 * xu)
        else:
            Cc = 0.67 * (1 - 4 * g / 21) * fc_mean
            Mc = Cc * (0.5 - (0.5 - 8 * g / 49) / (1 - 4 * g / 21))

        Cs = 0
        Ms = 0
        es_bottom = 0

        # Use accelerated helper for the per-xu accumulation
        Cs_tmp, Ms_tmp, es_tmp = _compute_CsMs(xu, y_by_D, k, p, n, fc_mean, Es, fy, ey)
        Cs = Cs + Cs_tmp
        Ms = Ms + Ms_tmp
        es_bottom = es_tmp

        Pu_val = Cc + Cs
        Mu_val = Mc + Ms

        if first_point:
            Pu_val = 0
            first_point = False

        if abs(es_bottom + ey) <= 1e-4 and es_bottom < 0:
            KeyPoints['BalancedFailure'] = {'Mu': Mu_val, 'Pu': Pu_val}

        PuR.append(Pu_val)
        MuR.append(Mu_val)
        xu_by_D.append(xu)

        xu += step

    # Step 4: Assign salient points
    MuR = np.array(MuR)
    PuR = np.array(PuR)
    xu_by_D = np.array(xu_by_D)

    finite = np.isfinite(xu_by_D) & np.isfinite(MuR) & np.isfinite(PuR)
    if np.any(finite):
        f_interp_Mu = interp1d(xu_by_D[finite], MuR[finite], kind='linear', fill_value='extrapolate')
        f_interp_Pu = interp1d(xu_by_D[finite], PuR[finite], kind='linear', fill_value='extrapolate')
        KeyPoints['FullCompression'] = {'Mu': float(f_interp_Mu(1.0)), 'Pu': float(f_interp_Pu(1.0))}

    idx5 = np.argmin(np.abs(PuR))
    KeyPoints['PureBending'] = {'Mu': float(MuR[idx5]), 'Pu': float(PuR[idx5])}

    # Step 5: Smoothly close curve to Mu=0 if not already included
    finite_idx = np.where(~np.isnan(MuR) & ~np.isnan(PuR))[0]
    if finite_idx.size >= 2:
        idx1 = finite_idx[-2]
        idx2 = finite_idx[-1]
        Mu1 = MuR[idx1]; Mu2 = MuR[idx2]
        Pu1 = PuR[idx1]; Pu2 = PuR[idx2]

        if Mu2 != Mu1:
            slope = (Pu2 - Pu1) / (Mu2 - Mu1)
            Pu_at_Mu0 = Pu2 + slope * (0 - Mu2)
            if Pu_at_Mu0 < 0:
                Pu_at_Mu0 = 0
            MuR = np.append(MuR, 0)
            PuR = np.append(PuR, Pu_at_Mu0)
            xu_by_D = np.append(xu_by_D, np.nan)
        else:
            MuR = np.append(MuR, 0)
            PuR = np.append(PuR, PuR[idx2])
            xu_by_D = np.append(xu_by_D, np.nan)

    Pu1 = np.max(PuR)
    idx1 = np.argmax(PuR)
    KeyPoints['PureAxial'] = {'Mu': float(MuR[idx1]), 'Pu': float(Pu1)}

    Pu_design = 0.9 * Pu1
    idx2 = np.argmin(np.abs(PuR - Pu_design))
    KeyPoints['AxialLimit'] = {'Mu': float(MuR[idx2]), 'Pu': float(PuR[idx2])}

    # --- Original MATLAB file had "..." (missing content). Preserved here ---
    # ... MISSING MATLAB LINES ...

    return MuR, PuR, xu_by_D, KeyPoints

if __name__ == "__main__":
    MuR, PuR, xu_by_D, KeyPoints = generate_PM_True_Capacity(30, 415, 200000, 2, 3, 8)
    print("First 5 MuR:", MuR[:5])
    print("First 5 PuR:", PuR[:5])
    print("KeyPoints keys:", list(KeyPoints.keys()))
