import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.interpolate import interp1d


def generate_PM_IS456_nd_uncertain(fck, fy, Es, p, n, nbars_total):
    """
    Literal translation of MATLAB function generate_PM_IS456_nd_uncertain.m
    Preserves algorithm, variables, and structure exactly as in MATLAB code.
    """

    # Constants
    EffCover_by_D = 0.1
    xu_by_D_max = 10
    step = 0.01
    ey = 0.87 * fy / Es  # steel yield strain (design)

    # --- Generate y_by_D values based on n ---
    y_by_D = np.linspace(-0.5 + EffCover_by_D, 0.5 - EffCover_by_D, n)

    # --- Calculate k values ---
    k = np.zeros(n)
    middle_indices = list(range(1, n-1))  # MATLAB 2:n-1
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

    while abs(Pu_trial) > tolerance:
        Cc = 0.362 * xu_min_by_D * fck
        Mc = Cc * (0.5 - 0.416 * xu_min_by_D)
        Cs = 0
        Ms = 0

        for j in range(n):
            es = 0.0035 * ((xu_min_by_D - 0.5 + y_by_D[j]) / xu_min_by_D)
            fc = (0.447 * fck) if (es > 0) else 0
            if abs(es) < ey:
                fs = Es * es
            else:
                fs = np.sign(es) * 0.87 * fy
            Cs = Cs + (fs - fc) * k[j] * p / 100
            Ms = Ms + (fs - fc) * k[j] * p * y_by_D[j] / 100

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
            Cc = 0.362 * xu * fck
            Mc = Cc * (0.5 - 0.416 * xu)
        else:
            Cc = 0.447 * (1 - 4 * g / 21) * fck
            Mc = Cc * (0.5 - (0.5 - 8 * g / 49) / (1 - 4 * g / 21))

        Cs = 0
        Ms = 0
        es_bottom = 0

        for j in range(n):
            if xu <= 1:
                es = 0.0035 * ((xu - 0.5 + y_by_D[j]) / xu)
            else:
                es = 0.002 * (1 + (y_by_D[j] - 1/14) / (xu - 3/7))

            if j == 0:
                es_bottom = es

            fc = (0.447 * fck) if (es > 0) else 0
            if abs(es) < ey:
                fs = Es * es
            else:
                fs = np.sign(es) * 0.87 * fy

            Cs = Cs + (fs - fc) * k[j] * p / 100
            Ms = Ms + (fs - fc) * k[j] * p * y_by_D[j] / 100

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

    f_interp_Mu = interp1d(xu_by_D, MuR, kind='linear', fill_value='extrapolate')
    f_interp_Pu = interp1d(xu_by_D, PuR, kind='linear', fill_value='extrapolate')
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

    return MuR, PuR, xu_by_D, KeyPoints

if __name__ == "__main__":
    MuR, PuR, xu_by_D, KeyPoints = generate_PM_IS456_nd_uncertain(25, 415, 200000, 2, 3, 8)
    print("First 5 MuR:", MuR[:5])
    print("First 5 PuR:", PuR[:5])
    print("KeyPoints keys:", list(KeyPoints.keys()))
