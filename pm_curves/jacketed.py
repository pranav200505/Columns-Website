# -*- coding: utf-8 -*-
"""
Jacketed (retrofitted) column P-M curve generation.
Provides:
 - generate_jacketed_pm(...) compatibility wrapper used by app.py
 - _generate_jacketed_pm_detailed(...) main implementation
"""

import numpy as np
import matplotlib.pyplot as plt
import pathlib

from ._style import apply_theme, ACCENT, POINT, DANGER, STEEL, CONCRETE_EDGE, COMPRESSION_FILL

# ——— Steel stress–strain tables (IS456 design curves for HYSD bars) ———
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

def _steel_curve(f_y):
    """
    Return (strain, stress) arrays for an arbitrary steel grade.
    Grades present in steel_tables are used directly; any other grade uses the
    Fe415 design curve scaled by f_y/415 (stress scaled linearly, the strain
    axis shifted so the yield plateau starts at 0.002 + 0.87*f_y/Es).
    Previously, any grade other than 415/500 raised a KeyError.
    """
    if f_y in steel_tables:
        tbl = steel_tables[f_y]
        return tbl['strain'], tbl['stress']
    ref = steel_tables[415]
    scale = float(f_y) / 415.0
    stress = ref['stress'] * scale
    Es = 200000.0
    elastic = stress / Es                  # elastic part of strain at each point
    inelastic = ref['strain'] - ref['stress'] / Es
    strain = elastic + inelastic           # keep inelastic offsets of the reference curve
    return strain, stress

def f_si(eps, f_y):
    strain, stress = _steel_curve(f_y)
    eps_c = np.clip(abs(eps), strain[0], strain[-1])
    s = np.interp(eps_c, strain, stress)
    return np.sign(eps) * s

def _yield_strain(f_y):
    strain, _ = _steel_curve(f_y)
    return float(strain[-1])

def f_ci(eps, f_ck):
    if eps <= 0:
        return 0.0
    if eps >= 0.002:
        return 0.447 * f_ck
    r = eps / 0.002
    return 0.447 * f_ck * (2*r - r*r)

# ——— Rectangular section stress-block factors ———
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

# ——— Core: slice-by-slice circular stress block (vectorised) ———
def C_c_circular(x_u, D, f_ck, N_slices):
    dx = D / N_slices
    x_i = (np.arange(N_slices) + 0.5) * dx
    half_chord = np.sqrt(np.clip((D/2.0)**2 - (x_i - D/2.0)**2, 0.0, None))
    y_max = D/2.0 + half_chord
    d = 2.0 * half_chord

    with np.errstate(divide='ignore', invalid='ignore'):
        a_small = 0.362 * x_u / d
        g = 16.0 * (d - 30.0)**2 / (7.0 * x_u)**2
        a_big = 0.447 * (1.0 - 4.0*g/21.0)
        x1_strip = (0.5 - 8.0*g/49.0) * (d / (1.0 - 4.0*g/21.0))

    cond = x_u <= y_max
    a_i = np.where(cond, a_small, a_big)
    x_prime = np.where(cond, 0.416 * x_u, y_max - x1_strip)

    Cc_i = a_i * f_ck * (dx * d)
    lever = d/2.0 - x_prime
    return float(Cc_i.sum()), float((Cc_i * lever).sum())

# ——— Rectangular full-section compressive and moment ———
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
    corners = [(cover,cover), (b-cover,cover),
               (b-cover,D-cover), (cover,D-cover)]
    pos += corners[:min(4,num_bars)]
    rem = num_bars - len(pos)
    sides = [0,0,0,0]
    for i in range(rem):
        sides[i%4] += 1
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

# ——— Extrapolation helper (only for plotting) ———
def _extrapolate_to_axes(Mu_plot, Pu_plot):
    """
    Given plotted arrays (same units as axes), extrapolate linearly from the
    endpoint nearest each axis to intersect M=0 and/or P=0 if the curve never
    touches that axis. Returns possibly-extended arrays. Does not modify inputs.
    """
    M = np.asarray(Mu_plot, dtype=float)
    P = np.asarray(Pu_plot, dtype=float)
    if M.size < 2 or P.size < 2:
        return M, P

    Mext = M.tolist()
    Pext = P.tolist()

    def _hit_M0(i, j):
        Mi, Mj = M[i], M[j]
        Pi, Pj = P[i], P[j]
        denom = (Mi - Mj)
        if denom == 0:
            return None
        t = -Mi / denom
        if not np.isfinite(t) or t <= 0:
            return None
        P0 = Pi + t * (Pi - Pj)
        if not np.isfinite(P0):
            return None
        return (0.0, float(P0), i)

    def _hit_P0(i, j):
        Pi, Pj = P[i], P[j]
        Mi, Mj = M[i], M[j]
        denom = (Pi - Pj)
        if denom == 0:
            return None
        t = -Pi / denom
        if not np.isfinite(t) or t <= 0:
            return None
        M0 = Mi + t * (Mi - Mj)
        if not np.isfinite(M0):
            return None
        return (float(M0), 0.0, i)

    if np.all(M > 0):
        i = int(np.argmin(M))
        j = i-1 if i > 0 else 1
        hit = _hit_M0(i, j)
        if hit:
            M0, P0, insert_at = hit
            Mext.insert(insert_at, M0)
            Pext.insert(insert_at, P0)

    if np.all(P > 0):
        i = int(np.argmin(P))
        j = i-1 if i > 0 else 1
        hit = _hit_P0(i, j)
        if hit:
            M0, P0, insert_at = hit
            Mext.insert(insert_at, M0)
            Pext.insert(insert_at, P0)

    return np.array(Mext, dtype=float), np.array(Pext, dtype=float)

# ——— Rebar force functions ———
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

# ——— Primary function ———
def _generate_jacketed_pm_detailed(D_c, cover_c, dia_c, n_c, fy_c, f_ck_core,
                                   b, D_j, cover_j, dia_j, n_j, fy_j, f_ck_jacket,
                                   Es=200000.0, nondim=False, Pu_input='', outpath=None):
    apply_theme()

    # build bar geometry
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

    # bar_list: (y_i, A_s, f_y, f_ck_region)
    bar_list = []
    for x,y,dia,A,fy in core_bars:
        bar_list.append((y - D_j/2.0, A, fy, f_ck_core))
    for x,y,dia,A,fy in rect_bars:
        bar_list.append((y - D_j/2.0, A, fy, f_ck_jacket))

    # Balanced failure: extreme tension bar reaches its yield strain.
    # 0.0035*(xu - D_j/2 + y_min)/xu = -ey  =>  xu = 0.0035*(D_j/2 - y_min)/(0.0035 + ey)
    bottom_bar = min(bar_list, key=lambda t: t[0])
    ey_bal = _yield_strain(bottom_bar[2])
    xu_bal = 0.0035 * (D_j/2.0 - bottom_bar[0]) / (0.0035 + ey_bal)
    BalancedPoint = None

    # Interaction curve sweep (finer resolution than the original 0.1*D_j step)
    step    = 0.025 * D_j
    xu_vals = np.arange(0.001, 10*D_j + step, step)
    xu_vals = np.sort(np.append(xu_vals, xu_bal))
    Pu_vals, Mu_vals = [], []

    for xu in xu_vals:
        # core concrete
        Cc_core, Mc_core = C_c_circular(xu, D_c, f_ck_core, N_slices=200)
        # full rectangle @ f_ck_jacket
        Cc_rect_full   = C_c_rect(xu, b, D_j, f_ck_jacket)
        Mc_rect_full   = M_c_rect(xu, b, D_j, f_ck_jacket)
        # subtract out that same circular region @ f_ck_jacket
        Cc_sub, Mc_sub = C_c_circular(xu, D_c, f_ck_jacket, N_slices=200)
        Cc_jacket = Cc_rect_full - Cc_sub
        Mc_jacket = Mc_rect_full - Mc_sub

        Cc_total = Cc_core + Cc_jacket
        Mc_total = Mc_core + Mc_jacket

        Cs = C_s(xu, D_j, bar_list)
        Ms = M_s(xu, D_j, bar_list)

        Pu = Cc_total + Cs
        Mu = Mc_total + Ms

        if BalancedPoint is None and xu == xu_bal:
            BalancedPoint = (Mu, Pu)

        Pu_vals.append(Pu)
        Mu_vals.append(Mu)

    Mu_arr_raw = np.array(Mu_vals)    # N·mm
    Pu_arr_raw = np.array(Pu_vals)    # N

    finite_mask_raw = np.isfinite(Mu_arr_raw) & np.isfinite(Pu_arr_raw)
    Mu_arr_raw = Mu_arr_raw[finite_mask_raw]
    Pu_arr_raw = Pu_arr_raw[finite_mask_raw]
    xu_arr = xu_vals[finite_mask_raw]

    mask = (Mu_arr_raw >= 0) & (Pu_arr_raw >= 0)
    fck_norm = float(f_ck_jacket)

    mu_at_P = None
    mu_max = None
    mu_at_P_nd = None
    mu_max_nd = None
    BF_plot = None
    saved_path = None

    if Mu_arr_raw[mask].size > 0:
        mu_max_raw = float(np.nanmax(Mu_arr_raw[mask]))
        mu_max = mu_max_raw / 1e6                              # kN·m
        mu_max_nd = mu_max_raw / (fck_norm * b * (D_j**2))

    if nondim:
        Mu_plot = (Mu_arr_raw / (fck_norm * b * (D_j**2)))[mask]
        Pu_plot = (Pu_arr_raw / (fck_norm * b * D_j))[mask]
        x_label = r'$M/(f_{ck}\, b \, D_j^{2})$'
        y_label = r'$P/(f_{ck}\, b \, D_j)$'
    else:
        Mu_plot = (Mu_arr_raw / 1e6)[mask]
        Pu_plot = (Pu_arr_raw / 1e3)[mask]
        x_label = 'M$_u$ (kN·m)'
        y_label = 'P$_u$ (kN)'

    import matplotlib.patches as patches

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax = axes[0]

    # extrapolate plotted curve to axis intersections (plotting only)
    try:
        Mu_plot, Pu_plot = _extrapolate_to_axes(Mu_plot, Pu_plot)
    except Exception:
        pass

    ax.plot(Mu_plot, Pu_plot, linewidth=2.4, color=ACCENT, label='Interaction curve')
    ax.fill_between(Mu_plot, Pu_plot, 0, color=ACCENT, alpha=0.07)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("P–M Interaction Curve — Jacketed Section")

    # Balanced point marker
    if BalancedPoint is not None and BalancedPoint[0] >= 0 and BalancedPoint[1] >= 0:
        Mu_b_raw, Pu_b_raw = BalancedPoint
        if nondim:
            BF_plot = (Mu_b_raw / (fck_norm * b * (D_j**2)), Pu_b_raw / (fck_norm * b * D_j))
        else:
            BF_plot = (Mu_b_raw / 1e6, Pu_b_raw / 1e3)
        ax.plot([BF_plot[0]], [BF_plot[1]], marker='*', color=DANGER, markersize=14,
                linestyle='None', label='Balanced failure')
        ax.annotate(f"BF ({BF_plot[0]:.4g}, {BF_plot[1]:.4g})", xy=BF_plot, xytext=(10, 10),
                    textcoords='offset points', color=DANGER, fontsize=9)

    # Robust y-limit
    if Pu_plot.size > 0:
        try:
            y_max = float(np.nanpercentile(Pu_plot, 99.5))
            if np.isfinite(y_max) and y_max > 0:
                ax.set_ylim(0, y_max * 1.05)
            else:
                ax.set_ylim(0, float(np.nanmax(Pu_plot)) * 1.05)
        except Exception:
            ax.set_ylim(0, float(np.nanmax(Pu_plot)) * 1.05)

    # Right: proportional cross-section
    ax2 = axes[1]
    ax2.set_aspect('equal')

    ax2.add_patch(patches.Rectangle((0, 0), b, D_j, fill=False,
                                    edgecolor=CONCRETE_EDGE, linewidth=1.6))
    core_cx = (b - D_c)/2.0 + D_c/2.0
    core_cy = (D_j - D_c)/2.0 + D_c/2.0
    ax2.add_patch(plt.Circle((core_cx, core_cy), D_c/2.0, fill=False,
                             edgecolor='#94a3b8', linestyle='--'))

    for x, y, dia_item, _A, _fy in core_bars:
        ax2.add_patch(plt.Circle((x, y), dia_item/2.0, color=STEEL))
    for x, y in rect_raw:
        ax2.add_patch(plt.Circle((x, y), dia_j/2.0, color=STEEL))

    # If Pu_input provided, estimate xu, shade compression, mark interaction point.
    if Pu_input:
        try:
            user_val = float(Pu_input)
            if nondim:
                P_dim_user = user_val * fck_norm * b * D_j
            else:
                # dimensional input arrives in N (app converts kN -> N)
                P_dim_user = user_val

            idx_sort = np.argsort(Pu_arr_raw)
            Pu_sorted_raw = Pu_arr_raw[idx_sort]
            Mu_sorted_raw = Mu_arr_raw[idx_sort]
            xu_sorted = xu_arr[idx_sort]

            if (P_dim_user >= Pu_sorted_raw.min()) and (P_dim_user <= Pu_sorted_raw.max()):
                xu_est = float(np.interp(P_dim_user, Pu_sorted_raw, xu_sorted))
                Mu_at_P_raw = float(np.interp(P_dim_user, Pu_sorted_raw, Mu_sorted_raw))

                mu_at_P = Mu_at_P_raw / 1e6                          # kN·m always
                mu_at_P_nd = Mu_at_P_raw / (fck_norm * b * (D_j**2))

                # neutral axis + compression shading
                y_line = D_j - xu_est
                ax2.hlines(y_line, xmin=0, xmax=b, colors=ACCENT, linestyles='--')
                ax2.annotate(f"x$_u$ ≈ {xu_est:.1f} mm", xy=(b*0.55, y_line),
                             xytext=(10, -10), textcoords='offset points', color=ACCENT)
                if y_line < D_j:
                    poly_x = [0, b, b, 0]
                    poly_y = [max(0, y_line), max(0, y_line), D_j, D_j]
                    ax2.fill(poly_x, poly_y, color=COMPRESSION_FILL, alpha=0.6,
                             label='Compression zone')

                if nondim:
                    plotted_M = mu_at_P_nd
                    plotted_P = P_dim_user / (fck_norm * b * D_j)
                    annot_text = f"P*={plotted_P:.3f}\nM*={plotted_M:.3f}"
                else:
                    plotted_M = mu_at_P
                    plotted_P = P_dim_user / 1e3
                    annot_text = f"P={plotted_P:.1f} kN\nM={plotted_M:.2f} kN·m"

                ax.scatter([plotted_M], [plotted_P], s=90, edgecolor='k',
                           facecolor=POINT, zorder=10, label='Stated load point')

                cur_xlim = ax.get_xlim(); cur_ylim = ax.get_ylim()
                minx = min(cur_xlim[0], plotted_M); maxx = max(cur_xlim[1], plotted_M)
                miny = min(cur_ylim[0], plotted_P); maxy = max(cur_ylim[1], plotted_P)
                x_margin = (maxx - minx) * 0.08 if (maxx - minx) != 0 else 0.1
                y_margin = (maxy - miny) * 0.08 if (maxy - miny) != 0 else 0.1
                ax.set_xlim(minx - x_margin, maxx + x_margin)
                ax.set_ylim(max(0, miny - y_margin), maxy + y_margin)

                x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
                if plotted_M + 0.12 * x_range < ax.get_xlim()[1]:
                    ann_x = plotted_M + 0.12 * x_range; ha = 'left'
                else:
                    ann_x = plotted_M - 0.12 * x_range; ha = 'right'

                ax.annotate(annot_text, xy=(plotted_M, plotted_P), xytext=(ann_x, plotted_P),
                            arrowprops=dict(arrowstyle="->", color='black'),
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
                            fontsize=9, horizontalalignment=ha, verticalalignment='center')
            else:
                print(f"Pu = {P_dim_user:.1f} N is outside range "
                      f"{Pu_sorted_raw.min():.1f}–{Pu_sorted_raw.max():.1f}")
        except Exception as e:
            print(f"[generate_jacketed_pm] Pu_input handling error: {e}")

    ax.legend(loc='upper right')

    handles, labels = ax2.get_legend_handles_labels()
    if handles:
        ax2.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05))

    ax2.set_xlim(-0.05*b, 1.05*b)
    ax2.set_ylim(-0.05*D_j, 1.05*D_j)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.grid(False)
    ax2.set_title('Cross-section (proportional)')

    plt.tight_layout()

    if outpath is None:
        plt.show()
        saved_path = None
    else:
        outpath_parent = pathlib.Path(outpath).parent
        outpath_parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(outpath), dpi=200)
        plt.close(fig)
        saved_path = str(outpath)

    return {
        'outpath': saved_path,
        'mu_at_P': mu_at_P,        # kN·m
        'mu_max': mu_max,          # kN·m
        'mu_at_P_nd': mu_at_P_nd,  # M/(fck b Dj^2)
        'mu_max_nd': mu_max_nd,
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
    or explicit core/jacket material values.
    """
    D_c = D_core
    cover_c = core_cover
    dia_c = core_bar_dia
    n_c = core_num_bars

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
