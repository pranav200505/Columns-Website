# run_reliability.py
"""
Simple runner for the reliability main.
Edit the parameters below, save, and run:
    python run_reliability.py
"""

import time
from pathlib import Path

# Import the main driver (adjust path if your package layout differs)
from reliability.Reliability_Analysis_of_PM_Curves_literal import main

def run():
    # ------------------- User-editable parameters -------------------
    N = 1000                      # Monte Carlo samples (change to 10, 50, 100, 1000, etc.)
    fck_nom = 30.0               # mean concrete fck (MPa)
    fck_std = 5.0                # std deviation of fck (MPa)
    fy_nom = 415.0               # mean steel fy (MPa)
    fy_std = 25.0                # std deviation of fy (MPa)

    # reinforcement template params (use None to accept script defaults)
    p = None                     # reinforcement percentage (or None)
    n = None                     # number of rows (or None)
    nbars_total = None           # total number of bars (or None)

    # RNG seed for reproducibility (use None for random)
    seed = 12345

    # Extra kwargs forwarded if your main supports others (leave empty normally)
    extra_kwargs = {}

    # ------------------- Run and time -------------------
    print(f"Running reliability main with N={N}, fck=({fck_nom}±{fck_std}), fy=({fy_nom}±{fy_std}), seed={seed}")
    t0 = time.time()
    # call main with named args
    main(N=N,
         fck_nom=fck_nom, fck_std=fck_std,
         fy_nom=fy_nom, fy_std=fy_std,
         p=p, n=n, nbars_total=nbars_total,
         seed=seed,
         **extra_kwargs)
    elapsed = time.time() - t0
    print(f"Elapsed (s): {elapsed:.2f}")

    # Where to look for outputs
    outdir = Path.cwd() / "static" / "plots"
    print("\nCheck output files in:", outdir.resolve())
    if outdir.exists():
        files = sorted([p.name for p in outdir.iterdir() if p.is_file()])
        if files:
            print("Files written (latest):")
            for fname in files[-10:]:
                print("  ", fname)
        else:
            print("No files found yet in the plots folder.")
    else:
        print("Plots folder does not exist (the script should create it when saving).")

if __name__ == "__main__":
    run()
