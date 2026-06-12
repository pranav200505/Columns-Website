import numpy as np
import math
import matplotlib.pyplot as plt
import pathlib
from matplotlib.ticker import FuncFormatter

from ._style import apply_theme, ACCENT, POINT, DANGER, STEEL, CONCRETE_EDGE, COMPRESSION_FILL

# Steel stress-strain (simple bilinear cap at fy)
def f_si(eps, f_y, Es=200000.0):
    return np.sign(eps) * min(abs(eps) * Es, f_y)

# Concrete stress-strain simplified parabola/constant cap
def f_ci(epsilon, f_ck):
    if epsilon <= 0:
        return 0.0
    elif epsilon >= 0.002:
        return 0.447 * f_ck
    else:
        r = epsilon / 0.002
        return 0.447 * f_ck * (2 * r - r * r)

# strain at a bar inside a circular strip of local depth d (y_i measured from centerline)
def epsilon_si(x_u, d, y_i):
    if x_u <= d:
        return 0.0035 * ((x_u - d/2.0 + y_i) / x_u)
    else:
        return 0.002 * (1.0 + (y_i - d/14.0) / (x_u - 3.0 * d / 7.0))

# place bars evenly on a circle of diameter D with cover offset
def get_bar_positions_circular(D, cover, dia, num_bars):
    bar_positions = []
    R = D / 2.0 - cover
    C = D / 2.0
    for i in range(num_bars):
        theta = 2 * math.pi * i / num_bars
        x = C + R * math.cos(theta)
        y = C + R * math.sin(theta)
        bar_positions.append((x, y))
    return bar_positions

# Compute compression force and moment for circular section by slicing (vectorised)
def C_c_circular(x_u, D, f_ck, N_slices=400):
    dx = D / N_slices
    x_i = (np.arange(N_slices) + 0.5) * dx
    half_chord = np.sqrt(np.clip((D / 2.0) ** 2 - (x_i - D / 2.0) ** 2, 0.0, None))
    y_max = D / 2.0 + half_chord
    d = 2.0 * half_chord

    with np.errstate(divide='ignore', invalid='ignore'):
        a_small = 0.362 * x_u / d
        g = 16.0 * (d - 30.0) ** 2 / (7.0 * x_u) ** 2
        a_big = 0.447 * (1.0 - 4.0 * g / 21.0)
        x1_strip = (0.5 - 8.0 * g / 49.0) * (d / (1.0 - 4.0 * g / 21.0))

    cond = x_u <= y_max
    a_i = np.where(cond, a_small, a_big)
    x_prime = np.where(cond, 0.416 * x_u, y_max - x1_strip)

    Cc_i = a_i * f_ck * (dx * d)
    lever = d / 2.0 - x_prime
    return float(Cc_i.sum()), float((Cc_i * lever).sum())

# Steel axial + moment contribution in a single pass over the bars
def _steel_CsMs(x_u, D, bar_coords, A_s, f_ck, f_y, Es):
    Cs = 0.0
    Ms = 0.0
    for x_abs, y_abs in bar_coords:
        half_chord = math.sqrt(max(0.0, (D / 2.0) ** 2 - (x_abs - D / 2.0) ** 2))
        d_i = 2.0 * half_chord
        y_i = y_abs - D / 2.0
        eps = epsilon_si(x_u, d_i, y_i)
        sigma = (f_si(eps, f_y, Es) - f_ci(eps, f_ck)) * A_s
        Cs += sigma
        Ms += sigma * y_i
    return Cs, Ms

# kept for backwards compatibility
def C_s(x_u, D, bar_coords, A_s, f_ck, f_y, Es=200000.0):
    return _steel_CsMs(x_u, D, bar_coords, A_s, f_ck, f_y, Es)[0]

def M_s(x_u, D, bar_coords, A_s, f_ck, f_y, Es=200000.0):
    return _steel_CsMs(x_u, D, bar_coords, A_s, f_ck, f_y, Es)[1]


def generate_circular_pm(D, cover, num_bars, bar_dia, f_ck, f_y, Es,
                         nondim=False, Pu_input='', outpath=None):
    apply_theme()
    bars = get_bar_positions_circular(D, cover, bar_dia, num_bars)
    A_s = math.pi * bar_dia**2 / 4.0
    N_slices = 600

    # Balanced failure: extreme tension steel reaches yield strain exactly.
    # Solving 0.0035*(xu - D/2 + y_bot)/xu = -ey for xu (strain plane over full depth D):
    ey = f_y / Es
    xu_bal = 0.0035 * (D - cover) / (0.0035 + ey)

    xu_vals = np.linspace(0.001, 10 * D, 800)
    xu_vals = np.sort(np.append(xu_vals, xu_bal))
    Mu_vals, Pu_vals = [], []
    BalancedPoint = None
    mu_at_P = None
    mu_max = None
    mu_at_P_nd = None
    mu_max_nd = None
    BF_plot = None

    for xu in xu_vals:
        Cc, Mc = C_c_circular(xu, D, f_ck, N_slices=N_slices)
        Cs, Ms = _steel_CsMs(xu, D, bars, A_s, f_ck, f_y, Es)
        Pu = Cc + Cs
        Mu = Mc + Ms
        if BalancedPoint is None and xu == xu_bal:
            BalancedPoint = (Mu, Pu)
        Mu_vals.append(Mu)
        Pu_vals.append(Pu)

    Mu_arr = np.array(Mu_vals)
    Pu_arr = np.array(Pu_vals)

    mask = (Mu_arr >= 0) & (Pu_arr >= 0)
    Mu_plot_raw = Mu_arr[mask]
    Pu_plot_raw = Pu_arr[mask]

    if Mu_plot_raw.size > 0:
        mu_max_raw = float(np.nanmax(Mu_plot_raw))
        mu_max = mu_max_raw * 1e-6                      # kN·m
        mu_max_nd = mu_max_raw / (f_ck * D**3)          # M/(fck D^3)

    if nondim:
        Mu_plot = Mu_plot_raw / (f_ck * D**3)
        Pu_plot = Pu_plot_raw / (f_ck * D * D)
        x_label = r'$M/(f_{ck}\,D^3)$'
        y_label = r'$P/(f_{ck}\,D^2)$'
    else:
        Mu_plot = Mu_plot_raw * 1e-6   # kN·m
        Pu_plot = Pu_plot_raw * 1e-3   # kN
        x_label = 'M$_u$ (kN·m)'
        y_label = 'P$_u$ (kN)'

    # Plot interaction curve + proportional cross-section
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax = axes[0]
    ax.plot(Mu_plot, Pu_plot, linewidth=2.4, color=ACCENT, label='Interaction curve')
    ax.fill_between(Mu_plot, Pu_plot, 0, color=ACCENT, alpha=0.07)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title('P–M Interaction Curve — Circular Section')

    if Pu_plot.size > 0:
        ymax = float(np.nanmax(Pu_plot))
        if np.isfinite(ymax) and ymax > 0:
            ax.set_ylim(0, ymax * 1.05)

    if not nondim:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))

    # Balanced failure marker
    if BalancedPoint is not None:
        Mu_b_raw, Pu_b_raw = BalancedPoint
        if Mu_b_raw >= 0 and Pu_b_raw >= 0:
            if nondim:
                Mu_b = Mu_b_raw / (f_ck * D**3)
                Pu_b = Pu_b_raw / (f_ck * D * D)
            else:
                Mu_b = Mu_b_raw * 1e-6
                Pu_b = Pu_b_raw * 1e-3
            BF_plot = (Mu_b, Pu_b)
            ax.plot([Mu_b], [Pu_b], marker='*', color=DANGER, markersize=14,
                    linestyle='None', label='Balanced failure')
            ax.annotate(f"BF ({Mu_b:.3f}, {Pu_b:.3f})", xy=(Mu_b, Pu_b), xytext=(10, 10),
                        textcoords='offset points', color=DANGER, fontsize=9)

    # Cross-section
    ax2 = axes[1]
    ax2.set_aspect('equal')
    circle = plt.Circle((0, 0), D / 2.0, fill=False, edgecolor=CONCRETE_EDGE, linewidth=1.6)
    ax2.add_patch(circle)
    for x, y in bars:
        ax2.add_patch(plt.Circle((x - D / 2.0, y - D / 2.0), bar_dia / 2.0, color=STEEL))
    ax2.add_patch(plt.Circle((0, 0), D / 2.0 - cover, edgecolor='#94a3b8', linestyle='--', fill=False))

    # --- Mark and shade for provided Pu_input ---
    if Pu_input:
        try:
            val = float(Pu_input)
            # dimensional input arrives in N (app converts kN -> N); nondim arrives as P*
            P_dim = val * (f_ck * D * D) if nondim else val
            valid = np.isfinite(Pu_arr) & np.isfinite(Mu_arr)
            Pu_for_interp = Pu_arr[valid]
            Mu_for_interp = Mu_arr[valid]
            xu_for_interp = xu_vals[valid]
            if Pu_for_interp.size > 1:
                sort_idx = np.argsort(Pu_for_interp)
                Pu_sorted = Pu_for_interp[sort_idx]
                Mu_sorted = Mu_for_interp[sort_idx]
                xu_sorted = xu_for_interp[sort_idx]
                if (P_dim >= Pu_sorted.min()) and (P_dim <= Pu_sorted.max()):
                    xu_est = float(np.interp(P_dim, Pu_sorted, xu_sorted))
                    Mu_at_P_raw = float(np.interp(P_dim, Pu_sorted, Mu_sorted))

                    mu_at_P = Mu_at_P_raw * 1e-6                  # kN·m always
                    mu_at_P_nd = Mu_at_P_raw / (f_ck * D**3)

                    if nondim:
                        plotted_M = mu_at_P_nd
                        plotted_P = P_dim / (f_ck * D * D)
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
                        ann_x = plotted_M + 0.12 * x_range; ha = 'left'
                    else:
                        ann_x = plotted_M - 0.12 * x_range; ha = 'right'
                    ax.annotate(annot_text, xy=(plotted_M, plotted_P), xytext=(ann_x, plotted_P),
                                arrowprops=dict(arrowstyle="->", color='black'),
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
                                fontsize=9, horizontalalignment=ha, verticalalignment='center')

                    # neutral axis line + compression shading on the section
                    y_line = D / 2.0 - xu_est
                    ax2.hlines(y_line, xmin=-D / 2.0, xmax=D / 2.0, colors=ACCENT, linestyles='--')
                    ax2.annotate(f"x$_u$ ≈ {xu_est:.1f} mm", xy=(0, y_line),
                                 xytext=(10, 10), textcoords='offset points', color=ACCENT)

                    R = D / 2.0
                    if y_line >= R:
                        ax2.add_patch(plt.Circle((0, 0), R, color=COMPRESSION_FILL,
                                                 alpha=0.6, label='Compression zone'))
                    elif y_line <= -R:
                        pass
                    else:
                        t0 = np.arcsin(y_line / R)
                        t1 = math.pi - t0
                        theta = np.linspace(t0, t1, 300)
                        x_arc = R * np.cos(theta)
                        y_arc = R * np.sin(theta)
                        x_chord = math.sqrt(max(0.0, R * R - y_line * y_line))
                        poly_x = np.concatenate(([x_chord], x_arc, [-x_chord]))
                        poly_y = np.concatenate(([y_line], y_arc, [y_line]))
                        ax2.fill(poly_x, poly_y, color=COMPRESSION_FILL, alpha=0.6,
                                 label='Compression zone')
        except Exception as e:
            print(f"[generate_circular_pm] Pu_input handling error: {e}")

    ax.legend(loc='upper right')

    ax2.set_xlim(-D / 2.0 * 1.1, D / 2.0 * 1.1)
    ax2.set_ylim(-D / 2.0 * 1.1, D / 2.0 * 1.1)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.grid(False)
    ax2.set_title('Cross-section (proportional)')

    plt.tight_layout()
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
        'mu_at_P_nd': mu_at_P_nd,  # M/(fck D^3)
        'mu_max_nd': mu_max_nd,
        'BF_plot': BF_plot
    }
