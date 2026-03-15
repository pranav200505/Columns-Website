import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.interpolate import interp1d

def generate_PM_ACI318_nd_uncertain(fck, fy, Es, p, n, nbars_total):
    """
    Literal translation of MATLAB function generate_PM_ACI318_nd_uncertain.m
    Preserves algorithm, variables, and structure.
    """

    # Parameters
    EffCover_by_D = 0.1
    step = 0.001
    c_max_by_D = 3
    eps_bal = 1e-4

    # convert fck to fc' (fc_prime)
    fc_prime = 0.8 * fck

    # beta (compression block depth factor)
    if fc_prime >= 17.23 and fc_prime <= 28:
        beta = 0.85
    elif fc_prime > 28 and fc_prime < 56:
        beta = 0.85 - 0.05 * (fc_prime - 28) / 7
    else:
        beta = 0.65

    # geometry: steel layer positions
    y_by_D = np.linspace(1 - EffCover_by_D, EffCover_by_D, n)

    # k distribution
    k = np.zeros(n)
    bars_used = 0
    if n > 2:
        for ii in range(1, n-1):  # MATLAB 2:(n-1)
            k[ii] = 2
            bars_used = bars_used + 2
    remaining_bars = nbars_total - bars_used
    k[0] = remaining_bars / 2
    k[-1] = remaining_bars / 2
    k = k / nbars_total

    # outputs
    c_by_D_values = []
    PuR_values = []
    MuR_values = []
    BalancedPoint = None

    ey = fy / Es

    # Step 1: find c_min_by_D
    c_min_by_D = 0.05
    increment = 0.0001
    tolerance = 1e-4
    Pu_trial = -0.1

    while abs(Pu_trial) > tolerance:
        a_by_D = beta * c_min_by_D
        Cc = 0.85 * a_by_D * fc_prime
        Mc = Cc * (0.5 - 0.5 * a_by_D)
        Cs = 0
        Ms = 0
        for j in range(n):
            es_j = 0.003 * (c_min_by_D - y_by_D[j]) / c_min_by_D
            if abs(es_j) < ey:
                fs_j = Es * es_j
            else:
                fs_j = np.sign(es_j) * fy
            fc_j = (0.85 * fc_prime) if (es_j > 0) else 0
            Cs += (fs_j - fc_j) * k[j] * p / 100
            Ms += (fs_j - fc_j) * k[j] * p * (0.5 - y_by_D[j]) / 100
        Pu_trial = Cc + Cs
        if Pu_trial < 0:
            c_min_by_D = c_min_by_D + increment
        else:
            c_min_by_D = c_min_by_D - increment / 2
            increment = increment / 2
        if c_min_by_D > c_max_by_D:
            break

    # Step 2: Generate P-M curve
    c_by_D = c_min_by_D
    while c_by_D <= c_max_by_D + 1e-12:
        a_by_D = beta * c_by_D
        if a_by_D < 1:
            Cc = 0.85 * a_by_D * fc_prime
            Mc = Cc * (0.5 - 0.5 * a_by_D)
        else:
            Cc = 0.85 * fc_prime
            Mc = 0

        Cs = 0
        Ms = 0
        es = np.zeros(n)
        for j in range(n):
            es[j] = 0.003 * (c_by_D - y_by_D[j]) / c_by_D
            if abs(es[j]) < ey:
                fs_j = Es * es[j]
            else:
                fs_j = np.sign(es[j]) * fy
            fc_j = (0.85 * fc_prime) if (es[j] > 0) else 0
            Cs += (fs_j - fc_j) * k[j] * p / 100
            Ms += (fs_j - fc_j) * k[j] * p * (0.5 - y_by_D[j]) / 100

        if abs(es[0]) <= ey:
            phi = 0.65
        elif abs(es[0]) < ey + 0.003:
            phi = 0.65 + 0.25 * (abs(es[0]) - ey) / 0.003
        else:
            phi = 0.9

        PuR_val = phi * (Cc + Cs)
        MuR_val = phi * (Mc + Ms)

        c_by_D_values.append(c_by_D)
        PuR_values.append(PuR_val)
        MuR_values.append(MuR_val)

        es_bottom = es[0]
        if BalancedPoint is None and abs(es_bottom + ey) < eps_bal and es_bottom < 0:
            BalancedPoint = (MuR_val, PuR_val)

        c_by_D += step

    # Clean & outputs
    c_by_D = np.array(c_by_D_values)
    PuR = np.array(PuR_values)
    MuR = np.array(MuR_values)
    xu_by_D = c_by_D

    KeyPoints = {}

    if MuR.size == 0 or PuR.size == 0:
        KeyPoints['PureAxial'] = {'Mu': np.nan, 'Pu': np.nan}
        KeyPoints['AxialLimit'] = {'Mu': np.nan, 'Pu': np.nan}
        KeyPoints['FullCompression'] = {'Mu': np.nan, 'Pu': np.nan}
        KeyPoints['BalancedFailure'] = {'Mu': np.nan, 'Pu': np.nan}
        KeyPoints['PureBending'] = {'Mu': np.nan, 'Pu': np.nan}
        return MuR, PuR, xu_by_D, KeyPoints

    idxP1 = np.argmax(PuR)
    Pu1 = PuR[idxP1]
    KeyPoints['PureAxial'] = {'Mu': float(MuR[idxP1]), 'Pu': float(Pu1)}

    Pu_design = 0.9 * Pu1
    idxP2 = np.argmin(np.abs(PuR - Pu_design))
    KeyPoints['AxialLimit'] = {'Mu': float(MuR[idxP2]), 'Pu': float(PuR[idxP2])}

    try:
        f_interp_Mu = interp1d(xu_by_D, MuR, kind='linear', fill_value='extrapolate')
        f_interp_Pu = interp1d(xu_by_D, PuR, kind='linear', fill_value='extrapolate')
        KeyPoints['FullCompression'] = {'Mu': float(f_interp_Mu(1.0)), 'Pu': float(f_interp_Pu(1.0))}
    except Exception:
        KeyPoints['FullCompression'] = {'Mu': np.nan, 'Pu': np.nan}

    if BalancedPoint is not None:
        KeyPoints['BalancedFailure'] = {'Mu': BalancedPoint[0], 'Pu': BalancedPoint[1]}
    else:
        KeyPoints['BalancedFailure'] = {'Mu': np.nan, 'Pu': np.nan}

    idx5 = np.argmin(np.abs(PuR))
    KeyPoints['PureBending'] = {'Mu': float(MuR[idx5]), 'Pu': float(PuR[idx5])}

    return MuR, PuR, xu_by_D, KeyPoints

if __name__ == "__main__":
    MuR, PuR, xu_by_D, KeyPoints = generate_PM_ACI318_nd_uncertain(30, 415, 200000, 2, 3, 8)
    print("First 5 MuR:", MuR[:5])
    print("First 5 PuR:", PuR[:5])
    print("KeyPoints keys:", list(KeyPoints.keys()))
