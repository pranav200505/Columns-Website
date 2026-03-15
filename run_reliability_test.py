# run_reliability_test.py
import sys, time, traceback, pathlib, os, glob

project_root = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Force headless matplotlib backend
try:
    import matplotlib
    matplotlib.use("Agg")
except Exception as e:
    print("Could not set Agg backend:", e)

plot_folder = project_root / "static" / "plots"
plot_folder.mkdir(parents=True, exist_ok=True)
os.chdir(str(plot_folder))

print("Starting reliability.main() at", time.strftime("%Y-%m-%d %H:%M:%S"))
try:
    from reliability.Reliability_Analysis_of_PM_Curves_literal import main
except Exception:
    print("FAILED to import reliability.main():")
    traceback.print_exc()
    sys.exit(1)

t0 = time.time()
try:
    main()
    print("\nmain() returned normally")
except Exception:
    print("\nmain() raised exception:")
    traceback.print_exc()
t1 = time.time()
print("\nElapsed (s):", round(t1 - t0, 2))
print("Files written to", os.getcwd())
for p in sorted(glob.glob("*")):
    print("  ", p)
