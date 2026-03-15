# rectangular.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pathlib

# ------------------ Material / constitutive helpers ------------------

def f_si(eps, f_y):
    """Steel stress-strain: bilinear up to f_y with Es = 200000 MPa (units consistent)."""
    Es = 200000.0
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

def C_s(x_u, D, y_list, A_s, f_ck, f_y):
    total = 0.0
    for y_i in y_list:
        eps = epsilon_si(x_u, D, y_i)
        total += (f_si(eps, f_y) - f_ci(eps, f_ck)) * A_s
    return total

def M_c(x_u, b, D, f_ck):
    Cc = C_c(x_u, b, D, f_ck)
    x1 = compute_x1(x_u, D)
    return Cc * (D/2.0 - x1)

def M_s(x_u, D, y_list, A_s, f_ck, f_y):
    total = 0.0
    for y_i in y_list:
        eps = epsilon_si(x_u, D, y_i)
        total += (f_si(eps, f_y) - f_ci(eps, f_ck)) * A_s * y_i
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
    # Place bars & prepare lists
    bars = get_bar_positions(b, D, cover, bar_dia, num_bars)
    y_list = [y - D/2.0 for (_, y) in bars]  # y distances measured from mid-height
    A_s = np.pi * (bar_dia**2) / 4.0

    # compute Mu, Pu over xu sampling (keep full raw lists for interpolation)
    xu_vals = np.linspace(1e-3, 10 * D, 300)
    Mu_vals = []
    Pu_vals = []
    BalancedPoint = None

    # Initialize outputs that we will return later
    mu_at_P = None
    mu_max = None

    for xu in xu_vals:
        Cc = C_c(x_u=xu, b=b, D=D, f_ck=f_ck)
        Cs = C_s(x_u=xu, D=D, y_list=y_list, A_s=A_s, f_ck=f_ck, f_y=f_y)
        Pu = Cc + Cs

        Mc = M_c(x_u=xu, b=b, D=D, f_ck=f_ck)
        Ms = M_s(x_u=xu, D=D, y_list=y_list, A_s=A_s, f_ck=f_ck, f_y=f_y)
        Mu = Mc + Ms

        # Balanced failure detection (bottom bar reaches yield)
        # Relaxed tolerance slightly to be robust to numerical differences
        es_bottom = epsilon_si(xu, D, y_list[0])
        if BalancedPoint is None and abs(es_bottom + f_y / Es) < 5e-4 and es_bottom < 0:
            BalancedPoint = (Mu, Pu)
            # debug info to confirm detection (safe to leave; helpful in console)
            print(f"[rectangular] Balanced point detected at xu={xu:.4f}, es_bottom={es_bottom:.6g}, Mu={Mu:.6g}, Pu={Pu:.6g}")

        Mu_vals.append(Mu)
        Pu_vals.append(Pu)

    # Convert arrays (preserve raw arrays for interpolation)
    Mu_arr = np.array(Mu_vals)
    Pu_arr = np.array(Pu_vals)

    # Mask first-quadrant branch for plotting
    finite_mask = np.isfinite(Mu_arr) & np.isfinite(Pu_arr)
    Mu_arr = Mu_arr[finite_mask]; Pu_arr = Pu_arr[finite_mask]
    mask = (Mu_arr >= 0) & (Pu_arr >= 0)
    Mu_plot_arr = Mu_arr[mask]
    Pu_plot_arr = Pu_arr[mask]

    # nondimensionalize for plotting if requested
    if nondim:
        Mu_plot = Mu_plot_arr / (f_ck * b * D**2)    # M/(fck B D^2)
        Pu_plot = Pu_plot_arr / (f_ck * b * D)       # P/(fck B D)
        x_label = r'$M/(f_{ck} B D^2)$'
        y_label = r'$P/(f_{ck} B D)$'
    else:
        Mu_plot = Mu_plot_arr / 10**6
        Pu_plot = Pu_plot_arr / 10**3
        x_label = 'M_u (kN·m)'
        y_label = 'P_u (kN)'

    # Balanced point plotting coordinates (converted if nondim)
    BF_plot = None
    if BalancedPoint is not None:
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

    # Interaction curve (label added so legend can show it)
    ax.plot(Mu_plot, Pu_plot, linewidth=2, label='Interaction curve')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title('Interaction Curve (Rectangular)')
    ax.grid(True)

    # Plot Balanced Failure marker if present
    if BF_plot is not None:
        ax.plot([BF_plot[0]], [BF_plot[1]], marker='*', color='red', markersize=12, label='Balanced failure')
        ax.annotate(f"BF ({BF_plot[0]:.4g}, {BF_plot[1]:.4g})",
                    xy=(BF_plot[0], BF_plot[1]),
                    xytext=(10, 10),
                    textcoords='offset points',
                    color='red', fontsize=9)
        # debug print and ensure marker is visible by expanding limits if needed
        try:
            print(f"[rectangular] Plotting BF at {BF_plot}")
            cur_x0, cur_x1 = ax.get_xlim()
            cur_y0, cur_y1 = ax.get_ylim()
            new_x1 = max(cur_x1, BF_plot[0] * 1.1)
            new_y1 = max(cur_y1, BF_plot[1] * 1.1)
            ax.set_xlim(cur_x0, new_x1)
            ax.set_ylim(cur_y0, new_y1)
        except Exception:
            pass

    # Add legend for interaction plot (top-right)
    ax.legend(loc='upper right')

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

    # Outer rectangle and cover rectangle
    ax2.add_patch(patches.Rectangle((0, 0), b, D, fill=False, edgecolor='black', linewidth=1.2))
    ax2.add_patch(patches.Rectangle((cover, cover), b - 2 * cover, D - 2 * cover,
                                    fill=False, edgecolor='gray', linestyle='--'))

    # Reinforcement circles
    for x, y in bars:
        ax2.add_patch(plt.Circle((x, y), bar_dia / 2.0, color='red'))

    # If Pu_input provided: interpret depending on nondim and draw xu line + capture Mu values
    if Pu_input:
        try:
            # Parse input
            user_val = float(Pu_input)

            # Prepare raw arrays for interpolation (raw values are in same units as Pu_vals)
            Pu_raw = np.array(Pu_vals)
            Mu_raw = np.array(Mu_vals)

            valid = np.isfinite(Pu_raw) & np.isfinite(Mu_raw)
            if valid.sum() > 0:
                Pu_raw = Pu_raw[valid]
                Mu_raw = Mu_raw[valid]
                xu_for_interp = xu_vals[:len(Pu_raw)]

                # sort by Pu to build monotonic interpolation
                sort_idx = np.argsort(Pu_raw)
                Pu_sorted = Pu_raw[sort_idx]
                Mu_sorted = Mu_raw[sort_idx]
                xu_sorted = np.array(xu_for_interp)[sort_idx]

                # --- Robust interpretation of user input ---
                # If nondim, user_val is P* and should be converted to dimensional P directly
                if nondim:
                    P_dim = user_val * f_ck * b * D
                else:
                    # try interpreting user input as kN first (common), convert to N
                    P_dim_try = user_val * 1e3
                    if (P_dim_try >= Pu_sorted.min()) and (P_dim_try <= Pu_sorted.max()):
                        P_dim = P_dim_try
                    else:
                        # fall back to interpreting as N
                        P_dim = user_val

                # compute interpolated xu and Mu at P_dim if inside range
                if (P_dim >= Pu_sorted.min()) and (P_dim <= Pu_sorted.max()):
                    xu_est = float(np.interp(P_dim, Pu_sorted, xu_sorted))
                    Mu_at_P = float(np.interp(P_dim, Pu_sorted, Mu_sorted))
                    Mu_max = float(np.nanmax(Mu_raw))

                    # store results into lowercase variables for return (consistent units as earlier)
                    mu_at_P = Mu_at_P * 10**(-6)
                    mu_max = Mu_max * 10**(-6)

                    # print results similar to previous behavior
                    if nondim:
                        Mu_at_P_nondim = Mu_at_P / (f_ck * b * D**2)
                        Mu_max_nondim = (Mu_max * 10**(-6))/ (f_ck * b * D**2)
                        print(f"Interpolated Mu at given P*: {Mu_at_P:.6g} kN·m (non-dimensional = {Mu_at_P_nondim:.6g})")
                        print(f"Maximum Mu on curve: {Mu_max:.6g} kN·m (non-dimensional = {Mu_max_nondim:.6g})")
                    else:
                        print(f"Interpolated Mu at given P: {Mu_at_P:.6g} kN·m")
                        print(f"Maximum Mu on curve: {Mu_max:.6g} kN·m")

                    # Convert neutral axis depth (xu_est measured from top) to plotting y coordinate:
                    # In the plot the bottom is y=0 and top is y=D, so neutral axis y-coordinate is:
                    y_line = D - xu_est
                    ax2.hlines(y_line, xmin=0, xmax=b, colors='blue', linestyles='--')
                    ax2.annotate(f"xu ≈ {xu_est:.1f} mm", xy=(b * 0.5, y_line),
                                 xytext=(10, -20), textcoords='offset points', color='blue')

                    # --- Shade compression region from top (y = D) down to neutral axis y_line ---
                    if y_line < D:
                        poly_x = [0, b, b, 0]
                        poly_y = [y_line, y_line, D, D]
                        ax2.fill(poly_x, poly_y, color='lightgrey', alpha=0.6, label='Area under compression')

                    # --- Plot the point on interaction curve and annotate to the right ---
                    if nondim:
                        plotted_M = Mu_at_P / (f_ck * b * D**2)
                        plotted_P = P_dim / (f_ck * b * D)
                        annot_text = f"P*={plotted_P:.3f}\nM*={plotted_M:.3f}"
                    else:
                        plotted_M = Mu_at_P * 1e-6  # to kN·m
                        plotted_P = P_dim * 1e-3    # to kN (Pu_plot is in kN)
                        annot_text = f"P={plotted_P:.1f} kN\nM={plotted_M:.2f} kN·m"

                    # draw the point and ensure it is visible
                    ax.scatter([plotted_M], [plotted_P], s=80, edgecolor='k', facecolor='orange', zorder=10, label='Stated load point')

                    # expand axis limits if required
                    cur_xlim = ax.get_xlim(); cur_ylim = ax.get_ylim()
                    minx = min(cur_xlim[0], plotted_M); maxx = max(cur_xlim[1], plotted_M)
                    miny = min(cur_ylim[0], plotted_P); maxy = max(cur_ylim[1], plotted_P)
                    x_margin = (maxx - minx) * 0.08 if (maxx - minx) != 0 else 0.1
                    y_margin = (maxy - miny) * 0.08 if (maxy - miny) != 0 else 0.1
                    ax.set_xlim(minx - x_margin, maxx + x_margin)
                    ax.set_ylim(max(0, miny - y_margin), maxy + y_margin)

                    # annotation to the right (or left if near edge)
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
                else:
                    # out-of-range: still capture Mu_max for return and inform
                    Mu_max = float(np.nanmax(Mu_raw))
                    mu_max = Mu_max
                    if nondim:
                        Mu_max_nondim = Mu_max / (f_ck * b * D**2)
                        print(f"Provided P is outside computed Pu range. Maximum Mu on curve: {Mu_max:.6g} N·mm (non-dimensional = {Mu_max_nondim:.6g})")
                    else:
                        print(f"Provided P is outside computed Pu range. Maximum Mu on curve: {Mu_max:.6g} N·mm")
        except Exception as e:
            # do not crash the plotting if user input parsing fails
            print(f"[generate_rectangular_pm] Pu_input handling error: {e}")

    
    ax2.set_xlim(-0.05 * b, 1.05 * b)
    ax2.set_ylim(-0.05 * D, 1.05 * D)
    ax2.set_xticks([]); ax2.set_yticks([])
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

    # Return both the plot path and the interpolated Mu (if available)
    return {
        'outpath': saved_path,
        'mu_at_P': mu_at_P,
        'mu_max': mu_max,
        'BF_plot': BF_plot
    }
