# Shared matplotlib theme so every plot produced by the app looks consistent.
import matplotlib

matplotlib.use('Agg')

ACCENT = '#4f46e5'       # indigo  – primary curves
ACCENT_2 = '#0891b2'     # cyan    – secondary lines
POINT = '#f59e0b'        # amber   – user load point
DANGER = '#dc2626'       # red     – balanced failure marker
STEEL = '#dc2626'        # rebar circles
CONCRETE_EDGE = '#1e293b'
COMPRESSION_FILL = '#c7d2fe'

THEME = {
    'figure.facecolor': 'white',
    'axes.facecolor': '#fbfcfe',
    'axes.edgecolor': '#cbd5e1',
    'axes.linewidth': 1.0,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.titlecolor': '#0f172a',
    'axes.labelsize': 10.5,
    'axes.labelcolor': '#334155',
    'axes.grid': True,
    'grid.color': '#e2e8f0',
    'grid.linewidth': 0.8,
    'xtick.color': '#475569',
    'ytick.color': '#475569',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.frameon': True,
    'legend.framealpha': 0.92,
    'legend.edgecolor': '#e2e8f0',
    'legend.fontsize': 9,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
}


def apply_theme():
    matplotlib.rcParams.update(THEME)
