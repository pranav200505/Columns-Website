import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pathlib
from matplotlib.ticker import FuncFormatter  # >>> CHANGE 1: added import for tick formatting

# Steel stress-strain (simple bilinear cap at fy)
def f_si(eps, f_y):
    Es = 200000.0
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

# Compute compression force and moment for circular section by slicing
def C_c_circular(x_u, D, f_ck, N_slices=400):
    Δx = D / N_slices
    total_Cc = 0.0
    total_Mc = 0.0
    for i in range(N_slices):
        x_i = (i + 0.5) * Δx
        half_chord = math.sqrt(max(0.0, (D/2.0)**2 - (x_i - D/2.0)**2))
        y_min = D/2.0 - half_chord
        y_max = D/2.0 + half_chord
        d = y_max - y_min
        if x_u <= y_max:
            a_i = 0.362 * x_u / d
            x_prime = 0.416 * x_u
        else:
            g = 16 * (d - 30.0)**2 / (7.0 * x_u)**2
            a_i = 0.447 * (1.0 - 4.0 * g / 21.0)
            x1_strip = (0.5 - 8.0 * g / 49.0) * (d / (1.0 - 4.0 * g / 21.0))
            x_prime = y_max - x1_strip
        Cc_i = a_i * f_ck * (Δx * d)
        lever = d / 2.0 - x_prime
        total_Cc += Cc_i
        total_Mc += Cc_i * lever
    return total_Cc, total_Mc

# Steel contribution (axial)
def C_s(x_u, D, bar_coords, A_s, f_ck, f_y):
    total = 0.0
    for x_abs, y_abs in bar_coords:
        half_chord = math.sqrt(max(0.0, (D/2.0)**2 - (x_abs - D/2.0)**2))
        y_min = D/2.0 - half_chord
        y_max = D/2.0 + half_chord
        d_i = y_max - y_min
        y_i = y_abs - D/2.0
        eps = epsilon_si(x_u, d_i, y_i)
        f_s = f_si(eps, f_y)
        f_c = f_ci(eps, f_ck)
        total += (f_s - f_c) * A_s
    return total

# Steel moment contribution
def M_s(x_u, D, bar_coords, A_s, f_ck, f_y):
    total = 0.0
    for x_abs, y_abs in bar_coords:
        half_chord = math.sqrt(max(0.0, (D/2.0)**2 - (x_abs - D/2.0)**2))
        y_min = D/2.0 - half_chord
        y_max = D/2.0 + half_chord
        d_i = y_max - y_min
        y_i = y_abs - D/2.0
        eps = epsilon_si(x_u, d_i, y_i)
        f_s = f_si(eps, f_y)
        f_c = f_ci(eps, f_ck)
        total += (f_s - f_c) * A_s * y_i
    return total


def generate_circular_pm(D, cover, num_bars, bar_dia, f_ck, f_y, Es,
                         nondim=False, Pu_input='', outpath=None):
    bars = get_bar_positions_circular(D, cover, bar_dia, num_bars)
    A_s = math.pi * bar_dia**2 / 4.0

    xu_vals = np.linspace(0.001, 10 * D, 800)
    Mu_vals, Pu_vals = [], []
    BalancedPoint = None
    mu_at_P = None
    mu_max = None
    BF_plot = None
    N_slices = 600

    for xu in xu_vals:
        Cc, Mc = C_c_circular(xu, D, f_ck, N_slices=N_slices)
        Cs = C_s(xu, D, bars, A_s, f_ck, f_y)
        Ms = M_s(x_u := xu, D, bars, A_s, f_ck, f_y)
        Pu = Cc + Cs
        Mu = Mc + Ms
        es_bottom = epsilon_si(xu, D/2.0, - (D/2.0 - cover))
        if BalancedPoint is None and es_bottom < 0 and abs(abs(es_bottom) - f_y / Es) <= 1e-4:
            BalancedPoint = (Mu, Pu)
        Mu_vals.append(Mu)
        Pu_vals.append(Pu)

    Mu_arr = np.array(Mu_vals)
    Pu_arr = np.array(Pu_vals)

    mask = (Mu_arr >= 0) & (Pu_arr >= 0)
    Mu_plot_raw = Mu_arr[mask]
    Pu_plot_raw = Pu_arr[mask]

    if Mu_plot_raw.size > 0:
        mu_max = float(np.nanmax(Mu_plot_raw))

    # >>> CHANGE 2: apply scaling for dimensional plots (N·mm → kN·m, N → kN)
    if nondim:
        P_star = Pu_plot_raw / (f_ck * D * D)
        M_star = Mu_plot_raw / (f_ck * D**3)
        Mu_plot, Pu_plot = M_star, P_star
        x_label = r'$M/(f_{ck}D^3)$'
        y_label = r'$P/(f_{ck}BD)$'
    else:
        Mu_plot = Mu_plot_raw * 1e-6   # convert to kN·m
        Pu_plot = Pu_plot_raw * 1e-3   # convert to kN
        x_label = 'M_u (kN·m)'
        y_label = 'P_u (kN)'

    # Plot interaction curve + proportional cross-section
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax = axes[0]
    ax.plot(Mu_plot, Pu_plot, linewidth=2, label='Interaction curve')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title('Interaction Curve (Circular)')
    ax.grid(True)

    if Pu_plot.size > 0:
        ymax = float(np.nanmax(Pu_plot))
        if np.isfinite(ymax) and ymax > 0:
            ax.set_ylim(0, ymax * 1.05)

    # >>> CHANGE 3: apply tick formatting to remove scientific notation
    if not nondim:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))

    # Balanced failure marker (no other change)
    if BalancedPoint is not None and Mu_plot_raw.size > 0:
        Mu_b_raw, Pu_b_raw = BalancedPoint
        dists = np.abs(Mu_plot_raw - Mu_b_raw) + np.abs(Pu_plot_raw - Pu_b_raw)
        idx_near = int(np.nanargmin(dists))
        Mu_b_plot_raw = Mu_plot_raw[idx_near]
        Pu_b_plot_raw = Pu_plot_raw[idx_near]

        if nondim:
            Mu_b = Mu_b_plot_raw / (f_ck * D**3)
            Pu_b = Pu_b_plot_raw / (f_ck * D * D)
        else:
            Mu_b = Mu_b_plot_raw * 1e-6
            Pu_b = Pu_b_plot_raw * 1e-3

        BF_plot = (Mu_b, Pu_b)
        ax.plot([Mu_b], [Pu_b], marker='*', color='red', markersize=12, label='Balanced failure')
        ax.annotate(f"BF ({Mu_b:.3f}, {Pu_b:.3f})", xy=(Mu_b, Pu_b), xytext=(10, 10),
                    textcoords='offset points', color='red')

    # Cross-section and rest unchanged … 
    ax2 = axes[1]
    ax2.set_aspect('equal')
    circle = plt.Circle((0, 0), D / 2.0, fill=False, edgecolor='black')
    ax2.add_patch(circle)
    for x, y in bars:
        ax2.add_patch(plt.Circle((x - D / 2.0, y - D / 2.0), bar_dia / 2.0, color='red'))
    ax2.add_patch(plt.Circle((0, 0), D / 2.0 - cover, edgecolor='gray', linestyle='--', fill=False))

    # --- Mark and shade for provided Pu_input (minimal edits only) ---
    if Pu_input:
        try:
            val = float(Pu_input)
            # convert provided input to dimensional P for interpolation (consistent with earlier logic)
            P_dim = val * (f_ck * D * D) if nondim else val
            Pu_for_interp = Pu_arr.copy()
            Mu_for_interp = Mu_arr.copy()
            valid = np.isfinite(Pu_for_interp) & np.isfinite(Mu_for_interp)
            Pu_for_interp = Pu_for_interp[valid]
            Mu_for_interp = Mu_for_interp[valid]
            xu_for_interp = xu_vals[:len(Pu_for_interp)]
            if Pu_for_interp.size > 1:
                sort_idx = np.argsort(Pu_for_interp)
                Pu_sorted = Pu_for_interp[sort_idx]
                Mu_sorted = Mu_for_interp[sort_idx]
                xu_sorted = np.array(xu_for_interp)[sort_idx]
                if (P_dim >= Pu_sorted.min()) and (P_dim <= Pu_sorted.max()):
                    xu_est = float(np.interp(P_dim, Pu_sorted, xu_sorted))
                    Mu_at_P = float(np.interp(P_dim, Pu_sorted, Mu_sorted))

                    # compute plotting coordinates depending on nondim/dimensional mode
                    if nondim:
                        plotted_M = Mu_at_P / (f_ck * D**3)
                        plotted_P = P_dim / (f_ck * D * D)
                        mu_at_P = plotted_M
                        annot_text = f"P*={plotted_P:.3f}\nM*={plotted_M:.3f}"
                    else:
                        plotted_M = Mu_at_P * 1e-6  # kN·m
                        plotted_P = P_dim  * 1e-3 # kN (already in kN)
                        mu_at_P = plotted_M
                        annot_text = f"P={plotted_P:.1f} kN\nM={plotted_M:.2f} kN·m"

                    # draw the point more visibly (scatter) and ensure legend entry
                    ax.scatter([plotted_M], [plotted_P], s=80, edgecolor='k',
                               facecolor='orange', zorder=10, label='Stated load point')

                    # ensure the interaction axis includes the plotted point (expand limits if needed)
                    cur_xlim = ax.get_xlim()
                    cur_ylim = ax.get_ylim()
                    minx = min(cur_xlim[0], plotted_M)
                    maxx = max(cur_xlim[1], plotted_M)
                    miny = min(cur_ylim[0], plotted_P)
                    maxy = max(cur_ylim[1], plotted_P)
                    # add a small margin
                    x_margin = (maxx - minx) * 0.08 if (maxx - minx) != 0 else 0.1
                    y_margin = (maxy - miny) * 0.08 if (maxy - miny) != 0 else 0.1
                    ax.set_xlim(minx - x_margin, maxx + x_margin)
                    ax.set_ylim(max(0, miny - y_margin), maxy + y_margin)

                    # annotation placed to the right of the point (offset based on axis range)
                    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
                    # put annotation slightly to the right; if plotted_M is near right edge, force left placement
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

                    # determine neutral axis line (y_line) in section coords and draw hline + shading
                    # Note: section is centered at (0,0) with radius R=D/2, and y increases upward in axes coords
                    y_line = D / 2.0 - xu_est
                    ax2.hlines(y_line, xmin=-D / 2.0, xmax=D / 2.0, colors='blue', linestyles='--')
                    ax2.annotate(f"xu ≈ {xu_est:.1f} mm", xy=(0, y_line),
                                 xytext=(10, 10), textcoords='offset points', color='blue')

                    # --- NEW: shade the circular segment correctly (top portion y >= y_line) ---
                    R = D / 2.0
                    if y_line >= R:
                        # neutral axis at/above top => full circle compressed
                        ax2.add_patch(plt.Circle((0, 0), R, color='lightgrey', alpha=0.6, label='Area under compression'))
                    elif y_line <= -R:
                        # neutral axis below bottom => no compression region
                        pass
                    else:
                        # compute arc angles for the top circular segment
                        # arcsin gives the right-side intersection angle in [-pi/2, pi/2]
                        t0 = np.arcsin(y_line / R)
                        t1 = math.pi - t0
                        theta = np.linspace(t0, t1, 300)  # sweep from right intersection -> left intersection
                        x_arc = R * np.cos(theta)
                        y_arc = R * np.sin(theta)
                        # build polygon that follows the arc and closes along the chord at y=y_line
                        # ensure chord endpoints are exact intersection points
                        x_chord = math.sqrt(max(0.0, R * R - y_line * y_line))
                        poly_x = np.concatenate(([x_chord], x_arc, [-x_chord]))
                        poly_y = np.concatenate(([y_line], y_arc, [y_line]))
                        ax2.fill(poly_x, poly_y, color='lightgrey', alpha=0.6, label='Area under compression')

                    
                    
        except Exception:
            pass

    # finalize interaction plot legend (so markers show)
    ax.legend(loc='upper right')

    ax2.set_xlim(-D / 2.0 * 1.1, D / 2.0 * 1.1)
    ax2.set_ylim(-D / 2.0 * 1.1, D / 2.0 * 1.1)
    ax2.set_xticks([]); ax2.set_yticks([])
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
        'mu_at_P': mu_at_P,
        'mu_max': mu_max * 1e-6 if mu_max else None,
        'BF_plot': BF_plot
    }
