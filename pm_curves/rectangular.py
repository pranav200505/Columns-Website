# rectangular.py
import numpy as np
import matplotlib.pyplot as plt
import pathlib

from ._style import apply_theme, ACCENT, POINT, DANGER, STEEL, CONCRETE_EDGE, COMPRESSION_FILL

# ------------------ Material / constitutive helpers ------------------

def f_si(eps, f_y, Es=200000.0):
    """Steel stress-strain: bilinear up to f_y."""
    return np.sign(eps) * min(abs(eps) * Es, f_y)

def f_ci(epsilon, f_ck):
    """Simplified concrete stress-strain (parabolic to a cap at 0.447*fck)."""
    if epsilon <= 0:
        return 0.0
    if epsilon >= 0.002:
        return 0.447 * f_ck
    r = epsilon / 0.002
    return 0.447 * f_ck * (2 * r - r * r)

# ------------------ Stress block helper functions ------------------

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

# ------------------ Strain distribution ------------------

def epsilon_si(x_u, D, y_i):
    """
    Strain at a reinforcement located at distance y_i from section centerline
    given neutral axis depth x_u (measured from top).
    """
    if x_u <= D:
        return 0.0035 * ((x_u - D/2.0 + y_i) / x_u)
    else:
        return 0.002 * (1.0 + (y_i - D/14.0) / (x_u - 3.0 * D / 7.0))

# ------------------ Concrete & steel force / moment contributions ------------------

def C_c(x_u, b, D, f_ck):
    a_val = compute_a(x_u, D)
    return a_val * f_ck * b * D

def C_s(x_u, D, y_list, A_s, f_ck, f_y, Es=200000.0):
    total = 0.0
    for y_i in y_list:
        eps = epsilon_si(x_u, D, y_i)
        total += (f_si(eps, f_y, Es) - f_ci(eps, f_ck)) * A_s
    return total

def M_c(x_u, b, D, f_ck):
    Cc = C_c(x_u, b, D, f_ck)
    x1 = compute_x1(x_u, D)
    return Cc * (D/2.0 - x1)

def M_s(x_u, D, y_list, A_s, f_ck, f_y, Es=200000.0):
    total = 0.0
    for y_i in y_list:
        eps = epsilon_si(x_u, D, y_i)
        total += (f_si(eps, f_y, Es) - f_ci(eps, f_ck)) * A_s * y_i
    return total

# ------------------ Rebar placement helper ----------------

def get_bar_positions(b, D, cover, dia, num_bars):
    """
    Place reinforcement around rectangular perimeter:
    - corners first, then distribute remaining bars on edges evenly.
    Coordinates: origin at top-left; y increases downward.
    """
    corners = [(cover, cover), (b - cover, cover), (b - cover, D - cover), (cover, D - cover)]
    if num_bars <= 4:
        return corners[:num_bars]

    bar_positions = corners.copy()
    rem = num_bars - 4
    sides = [0, 0, 0, 0]
    for i in range(rem):
        sides[i % 4] += 1

    # top edge (between top-left and top-right)
    if sides[0]:
        xs = np.linspace(cover + dia/2, b - cover - dia/2, sides[0] + 2)[1:-1]
        for x in xs:
            bar_positions.append((x, cover))
    # right edge
    if sides[1]:
        ys = np.linspace(cover + dia/2, D - cover - dia/2, sides[1] + 2)[1:-1]
        for y in ys:
            bar_positions.append((b - cover, y))
    # bottom edge
    if sides[2]:
        xs = np.linspace(cover + dia/2, b - cover - dia/2, sides[2] + 2)[1:-1]
        for x in xs:
            bar_positions.append((x, D - cover))
    # left edge
    if sides[3]:
        ys = np.linspace(cover + dia/2, D - cover - dia/2, sides[3] + 2)[1:-1]
        for y in ys:
            bar_positions.append((cover, y))

    return bar_positions

# ------------------ Main generator (rectangular) ------------------

def generate_rectangular_pm(b, D, cover, bar_dia, num_bars, f_ck, f_y, Es,
                            nondim=False, Pu_input='', outpath=None):
    """
    Compute and plot rectangular PM curve and proportional cross-section.
    """
    apply_theme()

    # Place bars & prepare lists
    bars = get_bar_positions(b, D, cover, bar_dia, num_bars)
    y_list = [y - D/2.0 for (_, y) in bars]  # y distances measured from mid-height
    A_s = np.pi * (bar_dia**2) / 4.0

    # Balanced failure: extreme tension steel reaches the yield strain exactly.
    # Solving 0.0035*(xu - D/2 + y_min)/xu = -ey for xu:
    ey = f_y / Es
    y_min_bar = min(y_list)
    xu_bal = 0.0035 * (D/2.0 - y_min_bar) / (0.0035 + ey)

    # compute Mu, Pu over xu sampling (include the exact balanced xu in the sweep)
    xu_vals = np.linspace(1e-3, 10 * D, 300)
    xu_vals = np.sort(np.append(xu_vals, xu_bal))
    Mu_vals = []
    Pu_vals = []
    BalancedPoint = None

    mu_at_P = None
    mu_max = None
    mu_at_P_nd = None
    mu_max_nd = None

    for xu in xu_vals:
        Cc = C_c(x_u=xu, b=b, D=D, f_ck=f_ck)
        Cs = C_s(x_u=xu, D=D, y_list=y_list, A_s=A_s, f_ck=f_ck, f_y=f_y, Es=Es)
        Pu = Cc + Cs

        Mc = M_c(x_u=xu, b=b, D=D, f_ck=f_ck)
        Ms = M_s(x_u=xu, D=D, y_list=y_list, A_s=A_s, f_ck=f_ck, f_y=f_y, Es=Es)
        Mu = Mc + Ms

        if BalancedPoint is None and xu == xu_bal:
            BalancedPoint = (Mu, Pu)

        Mu_vals.append(Mu)
        Pu_vals.append(Pu)

    # Convert arrays (preserve raw arrays for interpolation)
    Mu_arr = np.array(Mu_vals)
    Pu_arr = np.array(Pu_vals)

    # Mask first-quadrant branch for plotting
    finite_mask = np.isfinite(Mu_arr) & np.isfinite(Pu_arr)
    Mu_fin = Mu_arr[finite_mask]; Pu_fin = Pu_arr[finite_mask]
    xu_fin = xu_vals[finite_mask]
    mask = (Mu_fin >= 0) & (Pu_fin >= 0)
    Mu_plot_arr = Mu_fin[mask]
    Pu_plot_arr = Pu_fin[mask]

    if Mu_plot_arr.size > 0:
        mu_max_raw = float(np.nanmax(Mu_plot_arr))
        mu_max = mu_max_raw * 1e-6                          # kN·m
        mu_max_nd = mu_max_raw / (f_ck * b * D**2)

    # nondimensionalize for plotting if requested
    if nondim:
        Mu_plot = Mu_plot_arr / (f_ck * b * D**2)    # M/(fck B D^2)
        Pu_plot = Pu_plot_arr / (f_ck * b * D)       # P/(fck B D)
        x_label = r'$M/(f_{ck}\,B\,D^2)$'
        y_label = r'$P/(f_{ck}\,B\,D)$'
    else:
        Mu_plot = Mu_plot_arr / 10**6
        Pu_plot = Pu_plot_arr / 10**3
        x_label = 'M$_u$ (kN·m)'
        y_label = 'P$_u$ (kN)'

    # Balanced point plotting coordinates (converted if nondim)
    BF_plot = None
    if BalancedPoint is not None and BalancedPoint[0] >= 0 and BalancedPoint[1] >= 0:
        Mu_b_raw, Pu_b_raw = BalancedPoint
        if nondim:
            Mu_b = Mu_b_raw / (f_ck * b * D**2)
            Pu_b = Pu_b_raw / (f_ck * b * D)
        else:
            Mu_b, Pu_b = Mu_b_raw / 10**6, Pu_b_raw / 10**3
        BF_plot = (Mu_b, Pu_b)

    # ------------------ Plotting ------------------
    import matplotlib.patches as patches
    import numpy as _np

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax = axes[0]

    ax.plot(Mu_plot, Pu_plot, linewidth=2.4, color=ACCENT, label='Interaction curve')
    ax.fill_between(Mu_plot, Pu_plot, 0, color=ACCENT, alpha=0.07)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title('P–M Interaction Curve — Rectangular Section')

    # Plot Balanced Failure marker if present
    if BF_plot is not None:
        ax.plot([BF_plot[0]], [BF_plot[1]], marker='*', color=DANGER, markersize=14,
                linestyle='None', label='Balanced failure')
        ax.annotate(f"BF ({BF_plot[0]:.4g}, {BF_plot[1]:.4g})",
                    xy=(BF_plot[0], BF_plot[1]),
                    xytext=(10, 10),
                    textcoords='offset points',
                    color=DANGER, fontsize=9)
        try:
            cur_x0, cur_x1 = ax.get_xlim()
            cur_y0, cur_y1 = ax.get_ylim()
            ax.set_xlim(cur_x0, max(cur_x1, BF_plot[0] * 1.1))
            ax.set_ylim(cur_y0, max(cur_y1, BF_plot[1] * 1.1))
        except Exception:
            pass

    # Robust y-limit to avoid spikes hiding the curve
    if Pu_plot.size > 0:
        try:
            ymax = float(_np.nanpercentile(Pu_plot, 99.5))
            if _np.isfinite(ymax) and ymax > 0:
                ax.set_ylim(0, ymax * 1.05)
            else:
                ax.set_ylim(0, float(_np.nanmax(Pu_plot)) * 1.05)
        except Exception:
            ax.set_ylim(0, float(_np.nanmax(Pu_plot)) * 1.05)

    # ------------------ Cross-section drawing ------------------
    ax2 = axes[1]
    ax2.set_aspect('equal')

    ax2.add_patch(patches.Rectangle((0, 0), b, D, fill=False, edgecolor=CONCRETE_EDGE, linewidth=1.6))
    ax2.add_patch(patches.Rectangle((cover, cover), b - 2 * cover, D - 2 * cover,
                                    fill=False, edgecolor='#94a3b8', linestyle='--'))

    for x, y in bars:
        ax2.add_patch(plt.Circle((x, y), bar_dia / 2.0, color=STEEL))

    # If Pu_input provided: interpret depending on nondim and draw xu line + capture Mu values
    if Pu_input:
        try:
            user_val = float(Pu_input)

            valid = np.isfinite(Pu_arr) & np.isfinite(Mu_arr)
            if valid.sum() > 1:
                Pu_raw = Pu_arr[valid]
                Mu_raw = Mu_arr[valid]
                xu_for_interp = xu_vals[valid]

                sort_idx = np.argsort(Pu_raw)
                Pu_sorted = Pu_raw[sort_idx]
                Mu_sorted = Mu_raw[sort_idx]
                xu_sorted = xu_for_interp[sort_idx]

                # dimensional input arrives in N (app converts kN -> N); nondim arrives as P*
                if nondim:
                    P_dim = user_val * f_ck * b * D
                else:
                    P_dim = user_val

                if (P_dim >= Pu_sorted.min()) and (P_dim <= Pu_sorted.max()):
                    xu_est = float(np.interp(P_dim, Pu_sorted, xu_sorted))
                    Mu_at_P_raw = float(np.interp(P_dim, Pu_sorted, Mu_sorted))

                    mu_at_P = Mu_at_P_raw * 1e-6                       # kN·m always
                    mu_at_P_nd = Mu_at_P_raw / (f_ck * b * D**2)

                    # neutral axis line on the section (bottom y=0, top y=D)
                    y_line = D - xu_est
                    ax2.hlines(y_line, xmin=0, xmax=b, colors=ACCENT, linestyles='--')
                    ax2.annotate(f"x$_u$ ≈ {xu_est:.1f} mm", xy=(b * 0.5, y_line),
                                 xytext=(10, -20), textcoords='offset points', color=ACCENT)

                    # shade compression region from top (y = D) down to neutral axis
                    if y_line < D:
                        poly_x = [0, b, b, 0]
                        poly_y = [max(0, y_line), max(0, y_line), D, D]
                        ax2.fill(poly_x, poly_y, color=COMPRESSION_FILL, alpha=0.6,
                                 label='Compression zone')

                    # plot the point on interaction curve and annotate
                    if nondim:
                        plotted_M = mu_at_P_nd
                        plotted_P = P_dim / (f_ck * b * D)
                        annot_text = f"P*={plotted_P:.3f}\nM*={plotted_M:.3f}"
                    else:
                        plotted_M = mu_at_P
                        plotted_P = P_dim * 1e-3
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
                        ann_x = plotted_M + 0.12 * x_range
                        ha = 'left'
                    else:
                        ann_x = plotted_M - 0.12 * x_range
                        ha = 'right'
                    ax.annotate(annot_text,
                                xy=(plotted_M, plotted_P),
                                xytext=(ann_x, plotted_P),
                                arrowprops=dict(arrowstyle="->", color='black'),
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
                                fontsize=9,
                                horizontalalignment=ha,
                                verticalalignment='center')
        except Exception as e:
            # do not crash the plotting if user input parsing fails
            print(f"[generate_rectangular_pm] Pu_input handling error: {e}")

    # Add legend for interaction plot
    ax.legend(loc='upper right')

    ax2.set_xlim(-0.05 * b, 1.05 * b)
    ax2.set_ylim(-0.05 * D, 1.05 * D)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.grid(False)
    ax2.set_title('Cross-section (proportional)')

    plt.tight_layout()

    # Save or show
    if outpath is None:
        plt.show()
        saved_path = None
    else:
        outpath_path = pathlib.Path(outpath)
        outpath_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(outpath_path), dpi=200)
        plt.close(fig)
        saved_path = str(outpath_path)

    return {
        'outpath': saved_path,
        'mu_at_P': mu_at_P,        # kN·m
        'mu_max': mu_max,          # kN·m
        'mu_at_P_nd': mu_at_P_nd,  # M/(fck B D^2)
        'mu_max_nd': mu_max_nd,
        'BF_plot': BF_plot
    }
