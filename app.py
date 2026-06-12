from flask import Flask, render_template, request
import os, time, uuid, traceback
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


# ------------------ Helpers ------------------
def _ensure_plots_dir():
    PLOTS.mkdir(parents=True, exist_ok=True)


def _cleanup_old_plots(max_age_hours=24):
    """Remove generated artefacts older than max_age_hours so the folder
    doesn't grow without bound."""
    try:
        cutoff = time.time() - max_age_hours * 3600
        for f in PLOTS.glob("*"):
            if f.suffix.lower() in ('.png', '.xlsx') and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
    except Exception:
        pass


def _get_float(form, name, default):
    v = form.get(name, '')
    v = v.strip() if v else ''
    try:
        return float(v) if v != '' else default
    except Exception:
        return default


def _get_int(form, name, default, vmin=None, vmax=None):
    v = form.get(name, '')
    v = v.strip() if v else ''
    try:
        vi = int(float(v)) if v != '' else default
    except Exception:
        vi = default
    if vmin is not None and vi < vmin:
        vi = vmin
    if vmax is not None and vi > vmax:
        vi = vmax
    return vi


class ValidationError(Exception):
    pass


def _require_positive(value, label):
    if not (value > 0):
        raise ValidationError(f"{label} must be a positive number.")
    return value


def _check_section(width, depth, cover, bar_dia, num_bars, prefix=""):
    p = f"{prefix} " if prefix else ""
    _require_positive(width, f"{p}width")
    _require_positive(depth, f"{p}depth")
    _require_positive(cover, f"{p}cover")
    _require_positive(bar_dia, f"{p}bar diameter")
    if num_bars < 2:
        raise ValidationError(f"{p}number of bars must be at least 2.")
    if 2 * cover >= min(width, depth):
        raise ValidationError(
            f"{p}cover ({cover:g} mm) is too large for the section — "
            f"twice the cover must be smaller than the least dimension.")
    if bar_dia >= min(width, depth) / 4:
        raise ValidationError(f"{p}bar diameter ({bar_dia:g} mm) is unreasonably "
                              f"large for the section size.")


# ----------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html', active='home')


@app.route('/pm_curves')
def pm_curves():
    return render_template('pm_curves.html', active='pm')


@app.route('/run_pm_curves', methods=['POST'])
def run_pm_curves():
    form = request.form
    geom = form.get('geometry', 'rectangular')
    plot_type = form.get('plot_type', 'dimensional')
    nondim = (plot_type == 'nondimensional')

    fck = _get_float(form, 'fck', 30.0)
    fy = _get_float(form, 'fy', 415.0)
    Es = _get_float(form, 'Es', 200000.0)

    # Load point: kN for dimensional plots (converted to N here, once),
    # P* for nondimensional plots (passed through unchanged).
    Pu_raw = (form.get('Pu_input', '') or '').strip()
    Pu_input = ''
    Pu_display = None
    if Pu_raw:
        try:
            Pu_val = float(Pu_raw)
            Pu_display = Pu_val
            Pu_input = str(Pu_val) if nondim else str(Pu_val * 1e3)
        except ValueError:
            return render_template('pm_curves.html', active='pm',
                                   error=f"Could not read the load point “{Pu_raw}” — "
                                         "please enter a number.")

    outfn = f"pm_{uuid.uuid4().hex}.png"
    outpath = PLOTS / outfn

    result = {}
    params = []

    try:
        _ensure_plots_dir()
        _cleanup_old_plots()

        _require_positive(fck, "f_ck")
        _require_positive(fy, "f_y")
        _require_positive(Es, "E_s")

        if geom == 'rectangular':
            b = _get_float(form, 'b', 300.0)
            D = _get_float(form, 'D', 600.0)
            cover = _get_float(form, 'cover', 40.0)
            bar_dia = _get_float(form, 'bar_dia', 16.0)
            num_bars = _get_int(form, 'num_bars', 8)
            _check_section(b, D, cover, bar_dia, num_bars)

            result = generate_rectangular_pm(b, D, cover, bar_dia, num_bars, fck, fy, Es,
                                             nondim=nondim, Pu_input=Pu_input,
                                             outpath=outpath)
            geom_label = 'Rectangular'
            params = [('B × D', f'{b:g} × {D:g} mm'),
                      ('Cover', f'{cover:g} mm'),
                      ('Reinforcement', f'{num_bars} × ⌀{bar_dia:g} mm')]

        elif geom == 'circular':
            D_c = _get_float(form, 'D_circ', 600.0)
            cover_c = _get_float(form, 'cover_c', 40.0)
            num_bars_c = _get_int(form, 'num_bars_c', 8)
            bar_dia_c = _get_float(form, 'bar_dia_c', 16.0)
            _check_section(D_c, D_c, cover_c, bar_dia_c, num_bars_c)

            result = generate_circular_pm(D_c, cover_c, num_bars_c, bar_dia_c, fck, fy, Es,
                                          nondim=nondim, Pu_input=Pu_input,
                                          outpath=outpath)
            geom_label = 'Circular'
            params = [('Diameter', f'{D_c:g} mm'),
                      ('Cover', f'{cover_c:g} mm'),
                      ('Reinforcement', f'{num_bars_c} × ⌀{bar_dia_c:g} mm')]

        elif geom == 'jacketed':
            D_core = _get_float(form, 'D_core', 300.0)
            B_j = _get_float(form, 'B_j', 500.0)
            D_j = _get_float(form, 'D_j', 600.0)
            core_cover = _get_float(form, 'core_cover', 30.0)
            jacket_cover = _get_float(form, 'jacket_cover', 40.0)
            core_bar_dia = _get_float(form, 'core_bar_dia', 12.0)
            core_num_bars = _get_int(form, 'core_num_bars', 6)
            jacket_bar_dia = _get_float(form, 'jacket_bar_dia', 16.0)
            jacket_num_bars = _get_int(form, 'jacket_num_bars', 8)

            _check_section(D_core, D_core, core_cover, core_bar_dia, core_num_bars,
                           prefix="core")
            _check_section(B_j, D_j, jacket_cover, jacket_bar_dia, jacket_num_bars,
                           prefix="jacket")
            if D_core >= min(B_j, D_j):
                raise ValidationError("The core diameter must be smaller than the "
                                      "jacket outer dimensions.")

            fck_core = _get_float(form, 'fck_core', fck)
            fy_core = _get_float(form, 'fy_core', fy)
            fck_jacket = _get_float(form, 'fck_jacket', fck)
            fy_jacket = _get_float(form, 'fy_jacket', fy)

            result = generate_jacketed_pm(D_core, core_cover, core_num_bars, core_bar_dia,
                                          B_j, D_j, jacket_cover, jacket_bar_dia,
                                          jacket_num_bars,
                                          f_ck_core=fck_core, fy_core=fy_core,
                                          f_ck_jacket=fck_jacket, fy_jacket=fy_jacket,
                                          Es=Es, nondim=nondim,
                                          Pu_input=Pu_input, outpath=outpath)
            geom_label = 'Jacketed'
            params = [('Core ⌀', f'{D_core:g} mm'),
                      ('Jacket', f'{B_j:g} × {D_j:g} mm'),
                      ('Core bars', f'{core_num_bars} × ⌀{core_bar_dia:g} mm'),
                      ('Jacket bars', f'{jacket_num_bars} × ⌀{jacket_bar_dia:g} mm')]
        else:
            raise ValidationError("Unknown geometry selected.")

    except ValidationError as e:
        return render_template('pm_curves.html', active='pm', error=str(e))
    except Exception as e:
        traceback.print_exc()
        return render_template('pm_curves.html', active='pm',
                               error=f"Computation failed: {e}")

    params = [('Geometry', geom_label),
              ('Concrete', f'{fck:g} MPa'),
              ('Steel', f'{fy:g} MPa')] + params
    if Pu_display is not None:
        params.append(('Load point P', f'{Pu_display:g} ' + ('(P*)' if nondim else 'kN')))

    pm_path = result.get('outpath')
    context = {
        'kind': 'pm',
        'pm_image': Path(pm_path).name if pm_path else None,
        'mu_at_P': result.get('mu_at_P'),
        'mu_max': result.get('mu_max'),
        'mu_at_P_nd': result.get('mu_at_P_nd'),
        'mu_max_nd': result.get('mu_max_nd'),
        'bf_plot': result.get('BF_plot'),
        'nondim': nondim,
        'params': params,
        'error': None,
    }
    return render_template('results.html', **context)


@app.route('/probabilistic')
def probabilistic():
    return render_template('probabilistic.html', active='prob')


@app.route('/run_probabilistic', methods=['POST'])
def run_probabilistic():
    form = request.form

    # ── Simulation controls ──────────────────────────────────────────────────
    N = _get_int(form, 'N', 300, vmin=10, vmax=5000)
    seed_raw = (form.get('seed', '') or '').strip()
    seed = None
    if seed_raw:
        try:
            seed = int(seed_raw)
        except ValueError:
            seed = None

    # ── Concrete fck ─────────────────────────────────────────────────────────
    fck_nom = _get_float(form, 'fck_nom', 30.0)
    fck_vary = form.get('fck_vary') == '1'
    fck_cov = _get_float(form, 'fck_cov', 0.18)
    fck_std = fck_nom * fck_cov if fck_vary else 0.0

    # ── Steel fy ─────────────────────────────────────────────────────────────
    fy_nom = _get_float(form, 'fy_nom', 415.0)
    fy_vary = form.get('fy_vary') == '1'
    fy_cov = _get_float(form, 'fy_cov', 0.06)
    fy_std = fy_nom * fy_cov if fy_vary else 0.0

    # ── Reinforcement ratio p (%) ─────────────────────────────────────────────
    p_nom = _get_float(form, 'p_nom', 2.0)
    p_vary = form.get('p_vary') == '1'
    p_cov = _get_float(form, 'p_cov', 0.10)
    p_std = p_nom * p_cov if p_vary else 0.0

    # ── Bar layout (fixed) ───────────────────────────────────────────────────
    n_rows = _get_int(form, 'n_rows', 3, vmin=2)
    nbars_total = _get_int(form, 'nbars_total', 8, vmin=4)
    Es = _get_float(form, 'Es', 200000.0)

    _ensure_plots_dir()
    _cleanup_old_plots()

    try:
        if fck_nom <= 0 or fy_nom <= 0 or p_nom <= 0 or Es <= 0:
            raise ValidationError("All nominal values must be positive.")

        result = reliability_main(
            N=N,
            fck_nom=fck_nom, fck_std=fck_std,
            fy_nom=fy_nom, fy_std=fy_std,
            p_nom=p_nom, p_std=p_std,
            n=n_rows, nbars_total=nbars_total,
            Es=Es,
            seed=seed,
            use_parallel=False,   # safer inside Flask dev server
            verbose=True,
        )
    except ValidationError as e:
        return render_template('probabilistic.html', active='prob', error=str(e))
    except Exception as e:
        traceback.print_exc()
        return render_template('probabilistic.html', active='prob',
                               error=f"Simulation failed: {e}")

    # ── Collect output paths ─────────────────────────────────────────────────
    beta_image = Path(result['beta_image']).name if result.get('beta_image') else None
    pf_image = Path(result['pf_image']).name if result.get('pf_image') else None
    hist_images = [Path(p).name for p in result.get('hist_images', []) if p]
    excel_link = '/static/plots/' + Path(result['fragility_file']).name \
                 if result.get('fragility_file') else None

    rel_summary = {
        'N': N,
        'n_curves': result.get('n_curves'),
        'fck': f"{fck_nom:.1f} ± {fck_std:.2f} MPa" if fck_vary else f"{fck_nom:.1f} MPa (fixed)",
        'fy': f"{fy_nom:.1f} ± {fy_std:.2f} MPa" if fy_vary else f"{fy_nom:.1f} MPa (fixed)",
        'p': f"{p_nom:.2f} ± {p_std:.3f} %" if p_vary else f"{p_nom:.2f} % (fixed)",
        'seed': seed if seed is not None else '—',
        'elapsed': f"{result.get('elapsed_s', 0):.1f} s",
        'beta_min': result.get('beta_min'),
        'beta_max': result.get('beta_max'),
        'pf_max': result.get('pf_max'),
    }

    return render_template('results.html',
                           kind='reliability',
                           beta_image=beta_image,
                           pf_image=pf_image,
                           hist_images=hist_images,
                           excel_link=excel_link,
                           rel_summary=rel_summary,
                           error=None)


if __name__ == '__main__':
    app.run(debug=True)
