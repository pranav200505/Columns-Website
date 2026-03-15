# -*- coding: utf-8 -*-
"""
Wrapped jacketed-column PM curve code (literal translation of your script).
Provides:
 - generate_jacketed_pm(...) which follows your script inputs exactly
 - generate_jacketed_pm_wrapper(...) for backward compatibility with the previous app call signature
"""

import numpy as np
import matplotlib.pyplot as plt
import pathlib

# ——— Steel stress–strain tables ———
steel_tables = {
    415: {
        'strain': np.array([0.00000, 0.00144, 0.00163, 0.00192,
                            0.00241, 0.00276, 0.00380]),
        'stress': np.array([0.0,    288.7, 306.7, 324.8,
                            342.8, 351.8, 360.0])
    },
    500: {
        'strain': np.array([0.00000, 0.00174, 0.00195, 0.00226,
                            0.00277, 0.00312, 0.00417]),
        'stress': np.array([0.0,    347.8, 369.6, 391.3,
                            413.0, 423.9, 434.8])
    }
}

def f_si(eps, f_y):
    tbl = steel_tables[f_y]
    eps_c = np.clip(abs(eps), tbl['strain'][0], tbl['strain'][-1])
    s = np.interp(eps_c, tbl['strain'], tbl['stress'])
    return np.sign(eps) * s

def f_ci(eps, f_ck):
    if eps <= 0:
        return 0.0
    if eps >= 0.002:
        return 0.447 * f_ck
    r = eps / 0.002
    return 0.447 * f_ck * (2*r - r*r)

# ——— Rectangular section stress-block factors as before ———
def compute_g(x_u, D):
    return 16.0 * (D - 30.0)**2 / (7.0 * x_u)**2

def compute_a(x_u, D):
    if x_u <= D:
        return 0.362 * x_u / D
    g = compute_g(x_u, D)
    return 0.447 * (1.0 - 4.0 * g / 21.0)

def compute_x1(x_u, D):
    if x_u <= D:
        return 0.416 * x_u
    g = compute_g(x_u, D)
    return (0.5 - 8.0 * g / 49.0) * (D / (1.0 - 4.0 * g / 21.0))

# ——— Strain distribution over full depth D_j ———
def epsilon_si(x_u, D, y_i):
    if x_u <= D:
        return 0.0035 * ((x_u - D/2.0 + y_i) / x_u)
    return 0.002 * (1.0 + (y_i - D/14.0) / (x_u - 3.0*D/7.0))

# ——— Core: slice‐by‐slice circular stress block ———
def C_c_circular(x_u, D, f_ck, N_slices):
    Δx = D / N_slices
    total_Cc = 0.0
    total_Mc = 0.0
    for i in range(N_slices):
        x_i = (i + 0.5) * Δx
        half_chord = np.sqrt((D/2)**2 - (x_i - D/2)**2)
        y_min = D/2 - half_chord
        y_max = D/2 + half_chord
        d = y_max - y_min

        if x_u <= y_max:
            a_i     = 0.362 * x_u / d
            x_prime = 0.416 * x_u
        else:
            g        = 16*(d - 30)**2 / (7*x_u)**2
            a_i      = 0.447 * (1 - 4*g/21)
            x1_strip = (0.5 - 8*g/49) * (d / (1 - 4*g/21))
            x_prime  = y_max - x1_strip

        Cc_i  = a_i * f_ck * (Δx * d)
        lever = d/2 - x_prime
        total_Cc += Cc_i
        total_Mc += Cc_i * lever

    return total_Cc, total_Mc

# ——— Rectangular full‐section compressive and moment ———
def C_c_rect(x_u, b, D, f_ck):
    return compute_a(x_u, D) * f_ck * b * D

def M_c_rect(x_u, b, D, f_ck):
    Cc = C_c_rect(x_u, b, D, f_ck)
    x1 = compute_x1(x_u, D)
    return Cc * (D/2.0 - x1)

# ——— Rebar positions ———
def get_bar_positions_circular(D, cover, dia, num_bars):
    R = D/2.0 - cover
    C = D/2.0
    return [(C + R*np.cos(2*np.pi*i/num_bars),
             C + R*np.sin(2*np.pi*i/num_bars))
            for i in range(num_bars)]

def get_bar_positions_rectangular(b, D, cover, dia, num_bars):
    pos = []
    # corners first
    corners = [(cover,cover), (b-cover,cover),
               (b-cover,D-cover), (cover,D-cover)]
    pos += corners[:min(4,num_bars)]
    rem = num_bars - len(pos)
    sides = [0,0,0,0]
    for i in range(rem):
        sides[i%4] += 1
    # distribute remaining along edges
    if sides[0]:
        xs = np.linspace(cover+dia/2, b-cover-dia/2, sides[0]+2)[1:-1]
        pos += [(x, cover) for x in xs]
    if sides[1]:
        ys = np.linspace(cover+dia/2, D-cover-dia/2, sides[1]+2)[1:-1]
        pos += [(b-cover, y) for y in ys]
    if sides[2]:
        xs = np.linspace(cover+dia/2, b-cover-dia/2, sides[2]+2)[1:-1]
        pos += [(x, D-cover) for x in xs]
    if sides[3]:
        ys = np.linspace(cover+dia/2, D-cover-dia/2, sides[3]+2)[1:-1]
        pos += [(cover, y) for y in ys]
    return pos

# ——— Plot helper ———
def plot_section(b, D, bars):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.add_patch(plt.Rectangle((0,0), b, D,
                  fill=False, edgecolor='black'))
    for x, y, dia, A, fy in bars:
        ax.add_patch(plt.Circle((x, y), dia/2, color='red'))
    ax.set_xlim(0, b); ax.set_ylim(0, D); ax.set_aspect('equal')
    plt.xlabel("mm"); plt.ylabel("mm")
    plt.title("Retrofitted Column Section")
    plt.grid(True)
    plt.show()

# ——— Extrapolation helper (only for plotting) ———
def _extrapolate_to_axes(Mu_plot, Pu_plot):
    """
    Given plotted arrays (same units as axes), extrapolate linearly from the
    endpoint nearest each axis to intersect M=0 and/or P=0 if the curve never
    touches that axis. Returns possibly-extended arrays. Does not modify inputs.
    """
    import numpy as _np

    M = _np.asarray(Mu_plot, dtype=float)
    P = _np.asarray(Pu_plot, dtype=float)
    if M.size < 2 or P.size < 2:
        return M, P

    Mext = M.tolist()
    Pext = P.tolist()

    # helper: extrapolate from endpoint i toward neighbour j to hit M=0
    def _hit_M0(i, j):
        Mi, Mj = M[i], M[j]
        Pi, Pj = P[i], P[j]
        denom = (Mi - Mj)
        if denom == 0:
            return None
        t = -Mi / denom  # move from i toward j (t>0 means go past i toward j)
        if not _np.isfinite(t) or t <= 0:
            return None
        P0 = Pi + t * (Pi - Pj)
        if not _np.isfinite(P0):
            return None
        return (0.0, float(P0), i)

    # helper: extrapolate from endpoint i toward neighbour j to hit P=0
    def _hit_P0(i, j):
        Pi, Pj = P[i], P[j]
        Mi, Mj = M[i], M[j]
        denom = (Pi - Pj)
        if denom == 0:
            return None
        t = -Pi / denom
        if not _np.isfinite(t) or t <= 0:
            return None
        M0 = Mi + t * (Mi - Mj)
        if not _np.isfinite(M0):
            return None
        return (float(M0), 0.0, i)

    # If curve never reaches M=0 (all M>0), extend to M=0
    if _np.all(M > 0):
        i = int(_np.argmin(M))                # endpoint closest to M=0
        j = i-1 if i > 0 else 1               # neighbour
        hit = _hit_M0(i, j)
        if hit:
            M0, P0, insert_at = hit
            Mext.insert(insert_at, M0)
            Pext.insert(insert_at, P0)

    # If curve never reaches P=0 (all P>0), extend to P=0
    if _np.all(P > 0):
        i = int(_np.argmin(P))                # endpoint closest to P=0
        j = i-1 if i > 0 else 1
        hit = _hit_P0(i, j)
        if hit:
            M0, P0, insert_at = hit
            Mext.insert(insert_at, M0)
            Pext.insert(insert_at, P0)

    return _np.array(Mext, dtype=float), _np.array(Pext, dtype=float)

# ——— Updated rebar force functions ———
def C_s(x_u, D_j, bar_list):
    total = 0.0
    for y_i, A_s, f_y, f_ck in bar_list:
        eps = epsilon_si(x_u, D_j, y_i)
        total += (f_si(eps, f_y) - f_ci(eps, f_ck)) * A_s
    return total

def M_s(x_u, D_j, bar_list):
    total = 0.0
    for y_i, A_s, f_y, f_ck in bar_list:
        eps = epsilon_si(x_u, D_j, y_i)
        total += (f_si(eps, f_y) - f_ci(eps, f_ck)) * A_s * y_i
    return total

# ——— Primary function (literal script logic) ———
def _generate_jacketed_pm_detailed(D_c, cover_c, dia_c, n_c, fy_c, f_ck_core,
                                   b, D_j, cover_j, dia_j, n_j, fy_j, f_ck_jacket,
                                   Es=200000.0, nondim=False, Pu_input='', outpath=None):
    # build bar geometry exactly like your script
    core_raw   = get_bar_positions_circular(D_c, cover_c, dia_c, n_c)
    x_off = (b - D_c)/2.0
    y_off = (D_j - D_c)/2.0
    core_bars = [
        (x + x_off, y + y_off, dia_c,
         np.pi*dia_c**2/4.0, fy_c)
        for x,y in core_raw
    ]
    rect_raw  = get_bar_positions_rectangular(b, D_j, cover_j, dia_j, n_j)
    rect_bars = [
        (x, y, dia_j,
         np.pi*dia_j**2/4.0, fy_j)
        for x,y in rect_raw
    ]
    bars = core_bars + rect_bars

    # Prepare bar_list as in script: (y_i, A_s, f_y, f_ck_region)
    bar_list = []
    for x,y,dia,A,fy in core_bars:
        bar_list.append((y - D_j/2.0, A, fy, f_ck_core))
    for x,y,dia,A,fy in rect_bars:
        bar_list.append((y - D_j/2.0, A, fy, f_ck_jacket))

    # Interaction curve loop (exactly as your script)
    step    = 0.1 * D_j
    xu_vals = np.arange(0.001, 10*D_j + step, step)
    Pu_vals, Mu_vals = [], []

    # Use exactly the same N_slices and calculations as your script
    for xu in xu_vals:
        # core concrete
        Cc_core, Mc_core = C_c_circular(xu, D_c, f_ck_core, N_slices=200)
        # full rectangle @ f_ck_jacket
        Cc_rect_full   = C_c_rect(xu, b, D_j, f_ck_jacket)
        Mc_rect_full   = M_c_rect(xu, b, D_j, f_ck_jacket)
        # subtract out that same circular region @ f_ck_jacket
        Cc_sub, Mc_sub = C_c_circular(xu, D_c, f_ck_jacket, N_slices=200)
        # jacket = full rect minus core region
        Cc_jacket = Cc_rect_full - Cc_sub
        Mc_jacket = Mc_rect_full - Mc_sub

        Cc_total = Cc_core + Cc_jacket
        Mc_total = Mc_core + Mc_jacket

        Cs = C_s(xu, D_j, bar_list)
        Ms = M_s(xu, D_j, bar_list)

        Pu = Cc_total + Cs
        Mu = Mc_total + Ms
        Pu_vals.append(Pu)
        Mu_vals.append(Mu)

    # plot first‐quadrant interaction curve
    # --- build arrays and apply first-quadrant mask ---
    Mu_arr_raw = np.array(Mu_vals)    # N·mm
    Pu_arr_raw = np.array(Pu_vals)    # N

    # filter finite entries
    finite_mask_raw = np.isfinite(Mu_arr_raw) & np.isfinite(Pu_arr_raw)
    Mu_arr_raw = Mu_arr_raw[finite_mask_raw]
    Pu_arr_raw = Pu_arr_raw[finite_mask_raw]

    # plotting arrays in display units
    Mu_arr_plot_kNm = Mu_arr_raw / 1e6   # kN·m
    Pu_arr_plot_kN  = Pu_arr_raw / 1e3   # kN

    # first-quadrant mask (use dimensional sign)
    mask = (Mu_arr_raw >= 0) & (Pu_arr_raw >= 0)

    # make sure we have a defined normalization for nondim case
    fck_norm = float(f_ck_jacket)

    # initialize return values
    mu_at_P = None
    mu_max = None
    BF_plot = None
    saved_path = None

    # --- NON-DIMENSIONALIZATION (correct for jacketed outer dims) ---
    if nondim:
        # use RAW dimensional arrays (N, N·mm) to compute non-dimensional quantities
        P_star = Pu_arr_raw / (fck_norm * b * D_j)           # P / (f_ck * b * D_j)
        M_star = Mu_arr_raw / (fck_norm * b * (D_j**2))     # M / (f_ck * b * D_j^2)
        Mu_plot = M_star[mask]
        Pu_plot = P_star[mask]
        x_label = r'$M/(f_{ck}\, b \, D_j^{2})$'
        y_label = r'$P/(f_{ck}\, b \, D_j)$'
    else:
        # use plotting arrays converted to kN / kN·m
        Mu_plot = Mu_arr_plot_kNm[mask]
        Pu_plot = Pu_arr_plot_kN[mask]
        x_label = 'M_u (kN·m)'
        y_label = 'P_u (kN)'

    # --- Plotting (interaction & cross-section) ---
    import matplotlib.pyplot as _plt
    import matplotlib.patches as patches

    fig, axes = _plt.subplots(1, 2, figsize=(12, 6))

    # Left: Interaction curve
    ax = axes[0]

    # === EXTRAPOLATE TO AXES for plotting only ===
    # call helper to extend plotted curve to axes intersections when needed
    try:
        Mu_plot, Pu_plot = _extrapolate_to_axes(Mu_plot, Pu_plot)
    except Exception:
        # if anything goes wrong, just plot as-is
        pass

    # add label so legend can show the interaction curve
    ax.plot(Mu_plot, Pu_plot, linewidth=2, label='Interaction curve')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Interaction Curve (Jacketed)")
    ax.grid(True)

    # Balanced point snapping to plotted coords for return (if possible)
    if 'BalancedPoint' in locals() and BalancedPoint is not None:
        try:
            Mu_b_raw, Pu_b_raw = BalancedPoint
            # find nearest plotted index among masked raw arrays
            Mu_masked_raw = Mu_arr_raw[mask]   # N·mm
            Pu_masked_raw = Pu_arr_raw[mask]   # N
            if Mu_masked_raw.size > 0:
                dists = np.abs(Mu_masked_raw - Mu_b_raw) + np.abs(Pu_masked_raw - Pu_b_raw)
                idx_near = int(np.nanargmin(dists))
                Mu_b_plot_raw = Mu_masked_raw[idx_near]  # N·mm
                Pu_b_plot_raw = Pu_masked_raw[idx_near]  # N
                if nondim:
                    BF_plot = (Mu_b_plot_raw / (fck_norm * b * (D_j**2)), Pu_b_plot_raw / (fck_norm * b * D_j))
                else:
                    BF_plot = (Mu_b_plot_raw / 1e6, Pu_b_plot_raw / 1e3)  # convert to kN·m, kN
        except Exception:
            BF_plot = None

    # show interaction legend (top-right)
    ax.legend(loc='upper right')

    # Robust y-limit for visualization (use percentile to avoid a few spikes dominating)
    if Pu_plot.size > 0:
        try:
            y_max = float(np.nanpercentile(Pu_plot, 99.5))
            if np.isfinite(y_max) and y_max > 0:
                ax.set_ylim(0, y_max * 1.05)
            else:
                ax.set_ylim(0, float(np.nanmax(Pu_plot)) * 1.05)
        except Exception:
            ax.set_ylim(0, float(np.nanmax(Pu_plot)) * 1.05)

    # Right: Proportional cross-section
    ax2 = axes[1]
    ax2.set_aspect('equal')

    # Outer jacket rectangle
    outer = patches.Rectangle((0, 0), b, D_j, fill=False, edgecolor='black', linewidth=1.2)
    ax2.add_patch(outer)

    # Inner core circle center coordinates
    core_cx = (b - D_c)/2.0 + D_c/2.0
    core_cy = (D_j - D_c)/2.0 + D_c/2.0
    inner = _plt.Circle((core_cx, core_cy), D_c/2.0, fill=False, edgecolor='gray', linestyle='--')
    ax2.add_patch(inner)

    # draw core reinforcement (core_bars is list of (x,y,dia,A,fy))
    try:
        for x, y, dia_c_item, A_c_item, fy_c_item in core_bars:
            ax2.add_patch(_plt.Circle((x, y), dia_c_item/2.0, color='red'))
    except Exception:
        try:
            for x, y in core_raw:
                ax2.add_patch(_plt.Circle((x + (b - D_c)/2.0, y + (D_j - D_c)/2.0), dia_c/2.0, color='red'))
        except Exception:
            pass

    # draw jacket reinforcement (rect_raw list of (x,y) or rect_bars list of 5-tuples)
    try:
        for x, y in rect_raw:
            ax2.add_patch(_plt.Circle((x, y), dia_j/2.0, color='red'))
    except Exception:
        try:
            for x, y, dia_j_item, A_j_item, fy_j_item in rect_bars:
                ax2.add_patch(_plt.Circle((x, y), dia_j_item/2.0, color='red'))
        except Exception:
            pass

    # If Pu_input provided, estimate xu and draw on cross-section, shade compression and plot point on interaction.
    if Pu_input:
        try:
            # interpret user input depending on nondim flag
            user_val = float(Pu_input)
            if nondim:
                # user provided P* -> convert to dimensional using jacket normalization
                P_dim_user = user_val * fck_norm * b * D_j
            else:
                P_dim_user = user_val

            # sort raw Pu_arr_raw (dimensional N) and corresponding xu_vals for interpolation
            idx_sort = np.argsort(Pu_arr_raw)
            Pu_sorted_raw = Pu_arr_raw[idx_sort]              # dimensional (N)
            xu_sorted = xu_vals[:len(Pu_vals)][idx_sort]

            if (P_dim_user >= Pu_sorted_raw.min()) and (P_dim_user <= Pu_sorted_raw.max()):
                xu_est = float(np.interp(P_dim_user, Pu_sorted_raw, xu_sorted))

                # compute interpolated Mu at this provided P using RAW dimensional arrays (N·mm)
                Mu_sorted_raw = Mu_arr_raw[idx_sort]   # N·mm
                Mu_at_P_raw = float(np.interp(P_dim_user, Pu_sorted_raw, Mu_sorted_raw))  # N·mm

                # record for return in consistent display units:
                if nondim:
                    mu_at_P = Mu_at_P_raw / (fck_norm * b * (D_j**2))   # non-dimensional M*
                else:
                    mu_at_P = Mu_at_P_raw / 1e6                         # convert to kN·m for template

                # record mu_max similarly (use raw max)
                try:
                    mu_max_raw = float(np.nanmax(Mu_arr_raw[mask])) if Mu_arr_raw[mask].size > 0 else None
                    if mu_max_raw is not None:
                        mu_max = (mu_max_raw / (fck_norm * b * (D_j**2))) if nondim else (mu_max_raw / 1e6)
                    else:
                        mu_max = None
                except Exception:
                    mu_max = None

                # print results to console (preserve original script behaviour) — consistent units
                try:
                    if nondim:
                        Mu_at_P_n = Mu_at_P_raw / (fck_norm * b * (D_j**2))
                        mu_max_n = None if mu_max_raw is None else (mu_max_raw / (fck_norm * b * (D_j**2)))
                        print(f"Interpolated Mu at given P*: {Mu_at_P_raw:.6g} N·mm (non-dimensional = {Mu_at_P_n:.6g})")
                        if mu_max_raw is not None:
                            print(f"Maximum Mu on curve: {mu_max_raw:.6g} N·mm (non-dimensional = {mu_max_n:.6g})")
                    else:
                        print(f"Interpolated Mu at given P: {mu_at_P:.6g} kN·m")
                        if mu_max is not None:
                            print(f"Maximum Mu on curve: {mu_max:.6g} kN·m")
                except Exception:
                    pass

                # Convert neutral axis depth (xu_est measured from top) to plotting y coordinate:
                # In the plot the bottom is y=0 and top is y=D_j, so neutral axis plotting y-coordinate is:
                y_line = D_j - xu_est
                ax2.hlines(y_line, xmin=0, xmax=b, colors='blue', linestyles='--')
                ax2.annotate(f"xu ≈ {xu_est:.1f} mm", xy=(b*0.55, y_line), xytext=(10, -10),
                             textcoords='offset points', color='blue')

                # Shade compression region from top (y = D_j) down to neutral axis y_line
                if y_line < D_j:
                    poly_x = [0, b, b, 0]
                    poly_y = [y_line, y_line, D_j, D_j]
                    ax2.fill(poly_x, poly_y, color='lightgrey', alpha=0.6, label='Area under compression')

                # --- Plot the point on interaction curve and annotate to the right ---
                if nondim:
                    plotted_M = Mu_at_P_raw / (fck_norm * b * (D_j**2))
                    plotted_P = P_dim_user / (fck_norm * b * D_j)
                    annot_text = f"P*={plotted_P:.3f}\nM*={plotted_M:.3f}"
                else:
                    plotted_M = Mu_at_P_raw / 1e6  # kN·m
                    plotted_P = P_dim_user / 1e3   # kN
                    annot_text = f"P={plotted_P:.1f} kN\nM={plotted_M:.2f} kN·m"

                ax.scatter([plotted_M], [plotted_P], s=80, edgecolor='k',
                           facecolor='orange', zorder=10, label='Stated load point')

                # ensure plotted point is visible by adjusting axes if necessary
                cur_xlim = ax.get_xlim(); cur_ylim = ax.get_ylim()
                minx = min(cur_xlim[0], plotted_M); maxx = max(cur_xlim[1], plotted_M)
                miny = min(cur_ylim[0], plotted_P); maxy = max(cur_ylim[1], plotted_P)
                x_margin = (maxx - minx) * 0.08 if (maxx - minx) != 0 else 0.1
                y_margin = (maxy - miny) * 0.08 if (maxy - miny) != 0 else 0.1
                ax.set_xlim(minx - x_margin, maxx + x_margin)
                ax.set_ylim(max(0, miny - y_margin), maxy + y_margin)

                # annotation placed to the right of the plotted point (or left if near right edge)
                x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
                if plotted_M + 0.12 * x_range < ax.get_xlim()[1]:
                    ann_x = plotted_M + 0.12 * x_range
                    ha = 'left'
                else:
                    ann_x = plotted_M - 0.12 * x_range
                    ha = 'right'
                ann_y = plotted_P

                ax.annotate(annot_text,
                            xy=(plotted_M, plotted_P),
                            xytext=(ann_x, ann_y),
                            arrowprops=dict(arrowstyle="->", color='black'),
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
                            fontsize=9,
                            horizontalalignment=ha,
                            verticalalignment='center')
        except Exception:
            pass

    # Fallback interpolation block (if earlier block didn't set mu_at_P)
    if Pu_input:
        try:
            user_Pu = float(Pu_input)
            # convert if nondim
            if nondim:
                P_dim_for_print = user_Pu * fck_norm * b * D_j
            else:
                P_dim_for_print = user_Pu

            idx = np.argsort(Pu_arr_raw)
            Pu_s_raw, Mu_s_raw = Pu_arr_raw[idx], Mu_arr_raw[idx]
            if P_dim_for_print < Pu_s_raw[0] or P_dim_for_print > Pu_s_raw[-1]:
                # out-of-range — mimic original script behaviour by printing a message to console
                print(f"Pu = {P_dim_for_print:.1f} N is outside range {Pu_s_raw[0]:.1f}–{Pu_s_raw[-1]:.1f}")
            else:
                Mu_interp_raw = np.interp(P_dim_for_print, Pu_s_raw, Mu_s_raw)  # N·mm
                print(f"Interpolated Mu for Pu = {P_dim_for_print:.1f} N → Mu = {Mu_interp_raw:.1f} N·mm")
                # ensure mu_at_P available and in same units as input
                if mu_at_P is None:
                    if nondim:
                        mu_at_P = float(Mu_interp_raw) / (fck_norm * b * (D_j**2))
                    else:
                        mu_at_P = float(Mu_interp_raw) / 1e6
        except Exception:
            pass

    # compute mu_max if user did not request a Pu_input or it wasn't set above
    if mu_max is None:
        try:
            mu_max_raw = float(np.nanmax(Mu_arr_raw[mask])) if Mu_arr_raw[mask].size > 0 else None
            if mu_max_raw is not None:
                mu_max = (mu_max_raw / (fck_norm * b * (D_j**2))) if nondim else (mu_max_raw / 1e6)
        except Exception:
            mu_max = None

    # If there are legend handles in the cross-section axes (e.g. shading added above),
    # move the legend outside the plotted rectangle so it does not overlap the section.
    handles, labels = ax2.get_legend_handles_labels()
    if handles:
        ax2.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05))

    ax2.set_xlim(-0.05*b, 1.05*b)
    ax2.set_ylim(-0.05*D_j, 1.05*D_j)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title('Cross-section (proportional)')

    plt.tight_layout()

    # Save or show (preserve original behaviour)
    if outpath is None:
        plt.show()
        saved_path = None
    else:
        outpath_parent = pathlib.Path(outpath).parent
        outpath_parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(outpath), dpi=200)
        plt.close(fig)
        saved_path = str(outpath)

    # ------------------ MINIMAL FIX: ensure returned values are kN·m for template ------------------
    # The template expects mu_at_P and mu_max to be in kN·m for dimensional display.
    # If earlier logic produced non-dimensional values (M*), convert them back to kN·m here.
    try:
        if mu_at_P is not None:
            # if nondimensional value was stored, convert to kN·m:
            # mu_at_P (non-dim) * (fck * b * D_j^2) gives N·mm -> divide by 1e6 -> kN·m
            if nondim:
                mu_at_P = float(mu_at_P) * (fck_norm * b * (D_j**2)) / 1e6
            # else: assume mu_at_P already in kN·m (leave unchanged)
        if mu_max is not None:
            if nondim:
                mu_max = float(mu_max) * (fck_norm * b * (D_j**2)) / 1e6
            # else: mu_max already in kN·m (leave unchanged)
    except Exception:
        # defensive: if conversion fails, leave values as-is (template will handle None)
        pass

    # Return dict compatible with app.py
    return {
        'outpath': saved_path,
        # mu_at_P and mu_max follow display units (kN·m) — template computes nondim from them if requested
        'mu_at_P': mu_at_P,
        'mu_max': mu_max,
        'BF_plot': BF_plot
    }

# ——— Backwards-compatible wrapper for older app signature ———
def generate_jacketed_pm(D_core, core_cover, core_num_bars, core_bar_dia,
                         B_j, D_j, jacket_cover, jacket_bar_dia, jacket_num_bars,
                         f_ck=None, f_y=None,
                         f_ck_core=None, fy_core=None, f_ck_jacket=None, fy_jacket=None,
                         Es=200000.0, nondim=False, Pu_input='', outpath=None):
    """
    Compatibility wrapper: accepts either the older simple signature (single f_ck/f_y)
    or explicit core/jacket material values. Minimal changes:
     - If f_ck_core/fy_core provided, use those; otherwise fall back to f_ck/f_y.
     - If f_ck_jacket/fy_jacket provided, use those; otherwise fall back to f_ck/f_y.
    """
    # map inputs (preserve behaviour if only f_ck/f_y supplied)
    D_c = D_core
    cover_c = core_cover
    dia_c = core_bar_dia
    n_c = core_num_bars

    # determine core material
    if fy_core is not None and fy_core != '':
        fy_c = int(float(fy_core))
    elif f_y is not None:
        fy_c = int(float(f_y))
    else:
        fy_c = 415

    if f_ck_core is not None and f_ck_core != '':
        f_ck_core_val = float(f_ck_core)
    elif f_ck is not None:
        f_ck_core_val = float(f_ck)
    else:
        f_ck_core_val = 30.0

    # determine jacket material
    b = float(B_j)
    cover_j = jacket_cover
    dia_j = jacket_bar_dia
    n_j = jacket_num_bars

    if fy_jacket is not None and fy_jacket != '':
        fy_j = int(float(fy_jacket))
    elif f_y is not None:
        fy_j = int(float(f_y))
    else:
        fy_j = 415

    if f_ck_jacket is not None and f_ck_jacket != '':
        f_ck_jacket_val = float(f_ck_jacket)
    elif f_ck is not None:
        f_ck_jacket_val = float(f_ck)
    else:
        f_ck_jacket_val = 30.0

    return _generate_jacketed_pm_detailed(D_c, cover_c, dia_c, n_c, fy_c, f_ck_core_val,
                                          b, D_j, cover_j, dia_j, n_j, fy_j, f_ck_jacket_val,
                                          Es=Es, nondim=nondim, Pu_input=Pu_input, outpath=outpath)
