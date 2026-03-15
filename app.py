from flask import Flask, render_template, request
import os, uuid, traceback
from pathlib import Path

app = Flask(__name__)
BASE = Path(__file__).resolve().parent
PLOTS = BASE / "static" / "plots"

# import pm_curve backends
from pm_curves.rectangular import generate_rectangular_pm
from pm_curves.circular import generate_circular_pm
from pm_curves.jacketed import generate_jacketed_pm

# import reliability main directly (no subprocess needed)
from reliability.Reliability_Analysis_of_PM_Curves_literal import main as reliability_main

import matplotlib
matplotlib.use('Agg')

# ------------------ Cross-section diagram helpers ------------------
def _ensure_plots_dir():
    PLOTS.mkdir(parents=True, exist_ok=True)

def plot_rect_section(b, D, cover, bar_dia, num_bars, outpath):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(4, 5))
    ax.add_patch(plt.Rectangle((0, 0), b, D, fill=False, edgecolor='black', linewidth=1.4))
    ax.add_patch(plt.Rectangle((cover, cover), b - 2*cover, D - 2*cover,
                              fill=False, edgecolor='gray', linestyle='--', linewidth=1.0))
    perim_positions = []
    if num_bars <= 4:
        corners = [(cover, cover), (b - cover, cover), (b - cover, D - cover), (cover, D - cover)]
        perim_positions = corners[:num_bars]
    else:
        corners = [(cover, cover), (b - cover, cover), (b - cover, D - cover), (cover, D - cover)]
        perim_positions = corners.copy()
        rem = num_bars - 4
        sides = [0,0,0,0]
        for i in range(rem):
            sides[i % 4] += 1
        if sides[0]:
            xs = np.linspace(cover + bar_dia/2, b - cover - bar_dia/2, sides[0] + 2)[1:-1]
            perim_positions += [(x, cover) for x in xs]
        if sides[1]:
            ys = np.linspace(cover + bar_dia/2, D - cover - bar_dia/2, sides[1] + 2)[1:-1]
            perim_positions += [(b - cover, y) for y in ys]
        if sides[2]:
            xs = np.linspace(cover + bar_dia/2, b - cover - bar_dia/2, sides[2] + 2)[1:-1]
            perim_positions += [(x, D - cover) for x in xs]
        if sides[3]:
            ys = np.linspace(cover + bar_dia/2, D - cover - bar_dia/2, sides[3] + 2)[1:-1]
            perim_positions += [(cover, y) for y in ys]

    for x, y in perim_positions:
        ax.add_patch(plt.Circle((x, y), bar_dia/2.0, color='red'))

    ax.annotate("b (width)", xy=(b/2.0, -0.03*D), ha='center', va='top', fontsize=9)
    ax.annotate("D (depth)", xy=(b + 0.03*b, D/2.0), rotation=90, va='center', ha='left', fontsize=9)
    ax.annotate("", xy=(cover, D + 0.02*D), xytext=(0, D + 0.02*D), arrowprops=dict(arrowstyle="<->"))
    ax.text(cover/2.0, D + 0.025*D, "cover", ha='center', va='bottom', fontsize=8)

    ax.set_xlim(-0.05*b, 1.05*b)
    ax.set_ylim(-0.05*D, 1.05*D)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Rectangular section (schematic)", fontsize=10)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_circ_section(D, cover, bar_dia, num_bars, outpath):
    import matplotlib.pyplot as plt
    import numpy as np

    R = D / 2.0; C = R
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.add_patch(plt.Circle((C, C), R, fill=False, edgecolor='black', linewidth=1.4))
    ax.add_patch(plt.Circle((C, C), R - cover, fill=False, edgecolor='gray', linestyle='--', linewidth=1.0))

    for i in range(num_bars):
        theta = 2 * 3.14159265 * i / num_bars
        x = C + (R - cover) * __import__('math').cos(theta)
        y = C + (R - cover) * __import__('math').sin(theta)
        ax.add_patch(plt.Circle((x, y), bar_dia/2.0, color='red'))

    ax.annotate("D (diameter)", xy=(2*R + 0.03*D, C), rotation=90, va='center', ha='left', fontsize=9)
    ax.text(C, -0.07*D, "circle diameter D", ha='center', fontsize=9)
    ax.set_xlim(-0.05*D, 1.05*D + 0.05*D)
    ax.set_ylim(-0.05*D, 1.05*D)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Circular section (schematic)", fontsize=10)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_jacket_section(D_core, cover_core, dia_core, n_core, B_j, D_j, jacket_cover, dia_j, n_j, outpath):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.add_patch(plt.Rectangle((0, 0), B_j, D_j, fill=False, edgecolor='black', linewidth=1.4))
    core_cx = (B_j - D_core)/2.0 + D_core/2.0
    core_cy = (D_j - D_core)/2.0 + D_core/2.0
    ax.add_patch(plt.Circle((core_cx, core_cy), D_core/2.0, fill=False, edgecolor='gray', linestyle='--'))

    R_core = D_core / 2.0 - cover_core
    for i in range(n_core):
        theta = 2 * 3.14159265 * i / n_core
        x = core_cx + R_core * __import__('math').cos(theta)
        y = core_cy + R_core * __import__('math').sin(theta)
        ax.add_patch(plt.Circle((x, y), dia_core/2.0, color='red'))

    perim_angles = [2 * 3.14159265 * i / n_j for i in range(n_j)]
    R_j = min(B_j, D_j) / 2.0 - jacket_cover
    for theta in perim_angles:
        x = B_j/2.0 + R_j * __import__('math').cos(theta)
        y = D_j/2.0 + R_j * __import__('math').sin(theta)
        ax.add_patch(plt.Circle((x, y), dia_j/2.0, color='red'))

    ax.annotate("B_j (width)", xy=(B_j/2.0, -0.06*D_j), ha='center', fontsize=9)
    ax.annotate("D_j (depth)", xy=(B_j + 0.04*B_j, D_j/2.0), rotation=90, ha='left', fontsize=9)
    ax.set_xlim(-0.05*B_j, 1.05*B_j)
    ax.set_ylim(-0.05*D_j, 1.05*D_j)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Jacketed section (schematic)", fontsize=10)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)

# ----------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pm_curves')
def pm_curves():
    return render_template('pm_curves.html')

@app.route('/run_pm_curves', methods=['POST'])
def run_pm_curves():
    form = request.form
    geom = form.get('geometry')
    plot_type = form.get('plot_type')
    fck = float(form.get('fck', 30))
    fy = int(form.get('fy', 415))
    Es = float(form.get('Es', 200000))
    Pu_input = form.get('Pu_input', '').strip()
    if Pu_input:
        try:
            Pu_input_val = float(Pu_input)
            if plot_type == 'nondimensional':
                Pu_input = str(Pu_input_val)
            else:
                Pu_input = str(Pu_input_val * 1e3)
        except ValueError:
            pass

    outfn = f"pm_{uuid.uuid4().hex}.png"
    outpath = PLOTS / outfn

    pm_image = None
    mu_at_P = None
    mu_max = None
    bf_plot = None
    section_image = None

    try:
        _ensure_plots_dir()

        if geom == 'rectangular':
            b = float(form.get('b', 300))
            D = float(form.get('D', 600))
            cover = float(form.get('cover', 40))
            bar_dia = float(form.get('bar_dia', 16))
            num_bars = int(form.get('num_bars', 8))

            result = generate_rectangular_pm(b, D, cover, bar_dia, num_bars, fck, fy, Es,
                                            nondim=(plot_type == 'nondimensional'),
                                            Pu_input=Pu_input, outpath=outpath)

            section_fn = f"section_rect_{uuid.uuid4().hex}.png"
            section_fp = PLOTS / section_fn
            try:
                plot_rect_section(b, D, cover, bar_dia, num_bars, str(section_fp))
                section_image = section_fn
            except Exception:
                section_image = None

            if isinstance(result, dict):
                pm_path = result.get('outpath')
                mu_at_P = result.get('mu_at_P')
                mu_max = result.get('mu_max')
                bf_plot = result.get('BF_plot')
                pm_image = Path(pm_path).name if pm_path else None
            else:
                pm_image = outfn

            extra = {'b': b, 'D': D}

        elif geom == 'circular':
            D_c = float(form.get('D_circ', 600))
            cover_c = float(form.get('cover_c', 40))
            num_bars_c = int(form.get('num_bars_c', 8))
            bar_dia_c = float(form.get('bar_dia_c', 16))

            result = generate_circular_pm(D_c, cover_c, num_bars_c, bar_dia_c, fck, fy, Es,
                                         nondim=(plot_type == 'nondimensional'),
                                         Pu_input=Pu_input, outpath=outpath)

            section_fn = f"section_circ_{uuid.uuid4().hex}.png"
            section_fp = PLOTS / section_fn
            try:
                plot_circ_section(D_c, cover_c, bar_dia_c, num_bars_c, str(section_fp))
                section_image = section_fn
            except Exception:
                section_image = None

            if isinstance(result, dict):
                pm_path = result.get('outpath')
                mu_at_P = result.get('mu_at_P')
                mu_max = result.get('mu_max')
                bf_plot = result.get('BF_plot')
                pm_image = Path(pm_path).name if pm_path else None
            else:
                pm_image = outfn

            extra = {'b': D_c, 'D': D_c}

        else:  # jacketed
            D_core = float(form.get('D_core', 300))
            B_j = float(form.get('B_j', 500))
            D_j = float(form.get('D_j', 600))
            core_cover = float(form.get('core_cover', 30))
            jacket_cover = float(form.get('jacket_cover', 40))
            core_bar_dia = float(form.get('core_bar_dia', 12))
            core_num_bars = int(form.get('core_num_bars', 6))
            jacket_bar_dia = float(form.get('jacket_bar_dia', 16))
            jacket_num_bars = int(form.get('jacket_num_bars', 8))

            fck_core_val = form.get('fck_core', '').strip()
            fy_core_val = form.get('fy_core', '').strip()
            fck_jacket_val = form.get('fck_jacket', '').strip()
            fy_jacket_val = form.get('fy_jacket', '').strip()

            fck_core = float(fck_core_val) if fck_core_val != '' else fck
            fy_core = int(float(fy_core_val)) if fy_core_val != '' else fy
            fck_jacket = float(fck_jacket_val) if fck_jacket_val != '' else fck
            fy_jacket = int(float(fy_jacket_val)) if fy_jacket_val != '' else fy

            result = generate_jacketed_pm(D_core, core_cover, core_num_bars, core_bar_dia,
                                         B_j, D_j, jacket_cover, jacket_bar_dia, jacket_num_bars,
                                         f_ck_core=fck_core, fy_core=fy_core,
                                         f_ck_jacket=fck_jacket, fy_jacket=fy_jacket,
                                         Es=Es, nondim=(plot_type == 'nondimensional'),
                                         Pu_input=Pu_input, outpath=outpath)

            section_fn = f"section_jacket_{uuid.uuid4().hex}.png"
            section_fp = PLOTS / section_fn
            try:
                plot_jacket_section(D_core, core_cover, core_bar_dia, core_num_bars,
                                    B_j, D_j, jacket_cover, jacket_bar_dia, jacket_num_bars,
                                    str(section_fp))
                section_image = section_fn
            except Exception:
                section_image = None

            if isinstance(result, dict):
                pm_path = result.get('outpath')
                mu_at_P = result.get('mu_at_P')
                mu_max = result.get('mu_max')
                bf_plot = result.get('BF_plot')
                pm_image = Path(pm_path).name if pm_path else None
            else:
                pm_image = outfn

            extra = {'b': B_j, 'D': D_j}

    except Exception as e:
        traceback.print_exc()
        return render_template('results.html', pm_image=None, error=str(e),
                               mu_at_P=None, mu_max=None, bf_plot=None)

    context = {
        'pm_image': pm_image,
        'mu_at_P': mu_at_P,
        'mu_max': mu_max,
        'nondim': (plot_type == 'nondimensional'),
        'f_ck': fck,
        'error': None,
        'bf_plot': bf_plot,
        'section_image': section_image
    }
    context.update(extra)
    return render_template('results.html', **context)


@app.route('/probabilistic')
def probabilistic():
    return render_template('probabilistic.html')


@app.route('/run_probabilistic', methods=['POST'])
def run_probabilistic():
    form = request.form

    def get_float(name, default):
        v = form.get(name, '').strip()
        try:
            return float(v) if v != '' else default
        except Exception:
            return default

    def get_int(name, default, vmin=None, vmax=None):
        v = form.get(name, '').strip()
        try:
            vi = int(v) if v != '' else default
            if vmin is not None and vi < vmin: vi = vmin
            if vmax is not None and vi > vmax: vi = vmax
            return vi
        except Exception:
            return default

    # ── Simulation controls ──────────────────────────────────────────────────
    N = get_int('N', 300, vmin=10, vmax=5000)

    # ── Concrete fck ─────────────────────────────────────────────────────────
    fck_nom = get_float('fck_nom', 30.0)
    fck_vary = form.get('fck_vary') == '1'
    fck_cov  = get_float('fck_cov', 0.18)
    fck_std  = fck_nom * fck_cov if fck_vary else 0.0

    # ── Steel fy ─────────────────────────────────────────────────────────────
    fy_nom  = get_float('fy_nom', 415.0)
    fy_vary = form.get('fy_vary') == '1'
    fy_cov  = get_float('fy_cov', 0.06)
    fy_std  = fy_nom * fy_cov if fy_vary else 0.0

    # ── Reinforcement ratio p (%) ─────────────────────────────────────────────
    p_nom   = get_float('p_nom', 2.0)
    p_vary  = form.get('p_vary') == '1'
    p_cov   = get_float('p_cov', 0.10)
    p_std   = p_nom * p_cov if p_vary else 0.0

    # ── Bar layout (fixed) ───────────────────────────────────────────────────
    n_rows      = get_int('n_rows', 3, vmin=2)
    nbars_total = get_int('nbars_total', 8, vmin=4)
    Es          = get_float('Es', 200000.0)

    # ── Ensure output folder exists ──────────────────────────────────────────
    _ensure_plots_dir()

    try:
        result = reliability_main(
            N           = N,
            fck_nom     = fck_nom,
            fck_std     = fck_std,
            fy_nom      = fy_nom,
            fy_std      = fy_std,
            p_nom       = p_nom,
            p_std       = p_std,
            n           = n_rows,
            nbars_total = nbars_total,
            Es          = Es,
            use_parallel= False,   # safer inside Flask dev server
            verbose     = True,
        )
    except Exception as e:
        traceback.print_exc()
        return render_template('results.html',
                               error=str(e),
                               beta_image=None,
                               pf_image=None,
                               hist_images=[],
                               excel_link=None,
                               rel_summary=None)

    # ── Collect output paths ─────────────────────────────────────────────────
    beta_image  = Path(result['beta_image']).name  if result.get('beta_image')  else None
    pf_image    = Path(result['pf_image']).name    if result.get('pf_image')    else None
    hist_images = [Path(p).name for p in result.get('hist_images', []) if p]
    excel_link  = '/static/plots/' + Path(result['fragility_file']).name \
                  if result.get('fragility_file') else None

    rel_summary = {
        'N'        : N,
        'fck'      : f"{fck_nom:.1f} ± {fck_std:.2f} MPa" if fck_vary else f"{fck_nom:.1f} MPa (fixed)",
        'fy'       : f"{fy_nom:.1f} ± {fy_std:.2f} MPa"   if fy_vary  else f"{fy_nom:.1f} MPa (fixed)",
        'p'        : f"{p_nom:.2f} ± {p_std:.3f} %"        if p_vary   else f"{p_nom:.2f} % (fixed)",
        'elapsed'  : f"{result.get('elapsed_s', 0):.1f} s",
    }

    return render_template('results.html',
                           beta_image  = beta_image,
                           pf_image    = pf_image,
                           hist_images = hist_images,
                           excel_link  = excel_link,
                           rel_summary = rel_summary,
                           error       = None)


if __name__ == '__main__':
    app.run(debug=True)
