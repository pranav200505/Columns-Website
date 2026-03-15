# patch_reliability.py
import pathlib, re

rdir = pathlib.Path("reliability")
if not rdir.exists():
    print("reliability/ folder not found in current directory:", pathlib.Path.cwd())
    raise SystemExit(1)

pyfiles = sorted(rdir.glob("*.py"))
print("Files to inspect:", [p.name for p in pyfiles])

for p in pyfiles:
    txt = p.read_text(encoding="utf8")
    orig = txt
    # write backup
    bak = p.with_suffix(".py.bak")
    bak.write_text(orig, encoding="utf8")
    changed = False

    # Ensure headless backend set at top
    if "matplotlib.use('Agg')" not in txt and 'matplotlib.use("Agg")' not in txt:
        # insert at very top
        txt = "import matplotlib\nmatplotlib.use('Agg')\n" + txt
        changed = True

    # Replace plt.show() with plt.close()
    txt2 = re.sub(r"\bplt\.show\(\s*\)", "plt.close()", txt)
    if txt2 != txt:
        txt = txt2
        changed = True

    if changed:
        p.write_text(txt, encoding="utf8")
        print("Patched:", p.name, " (backup ->", bak.name + ")")
    else:
        print("No change needed:", p.name)

print("Patching complete. Backups created with .py.bak extensions.")
