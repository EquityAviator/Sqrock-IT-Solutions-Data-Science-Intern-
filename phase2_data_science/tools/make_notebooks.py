"""Turn each pipeline script into an executed Jupyter notebook next to it.

The scripts stay the single source of truth. This splits them into cells at their
"# ---- N. STEP" section markers, shows every figure a cell saves, executes the
notebook and writes <script>.ipynb beside the script.

    python tools/make_notebooks.py            # all pipelines
    python tools/make_notebooks.py task4      # just one
"""
import os
import re
import sys

import nbformat
from nbclient import NotebookClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = [
    "task1_stock_prediction/train.py",
    "task3_inventory/pipeline.py",
    "task4_loan_approval/train.py",
]
MARKER = re.compile(r"^# -{10,}\s*(.+?)\s*$")

SETUP = '''import os, time
from IPython.display import Image, display
__file__ = os.path.abspath("{script}")   # run from the task folder so the script's relative paths resolve
_seen = time.time()

def show_new_figures(folder=os.path.join("artifacts", "figures")):
    """Display the charts saved since the previous cell."""
    global _seen
    if os.path.isdir(folder):
        for f in sorted(os.listdir(folder), key=lambda f: os.path.getmtime(os.path.join(folder, f))):
            p = os.path.join(folder, f)
            if f.endswith(".png") and os.path.getmtime(p) >= _seen:
                display(Image(p))
    _seen = time.time()'''


def build(script_rel: str) -> str:
    path = os.path.join(ROOT, script_rel)
    src = open(path, encoding="utf-8").read()
    doc = re.match(r'\s*"""(.*?)"""\s*', src, re.S)
    body = src[doc.end():] if doc else src
    intro = doc.group(1).strip().splitlines() if doc else []
    title = intro[0].title() if intro else os.path.basename(script_rel)

    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells.append(nbformat.v4.new_markdown_cell(f"# {title}\n\n```\n" + "\n".join(intro[1:]) + "\n```"))
    nb.cells.append(nbformat.v4.new_code_cell(SETUP.replace("{script}", os.path.basename(path))))

    chunk, heading = [], None

    def flush():
        code = "\n".join(chunk).strip("\n")
        if heading:
            nb.cells.append(nbformat.v4.new_markdown_cell(f"## {heading.title()}"))
        if code.strip():
            if "savefig" in code:
                code += "\nshow_new_figures()"
            nb.cells.append(nbformat.v4.new_code_cell(code))

    for line in body.splitlines():
        m = MARKER.match(line)
        if m:
            flush()
            chunk, heading = [], m.group(1)
        else:
            chunk.append(line)
    flush()

    NotebookClient(nb, timeout=3600, kernel_name="python3",
                   resources={"metadata": {"path": os.path.dirname(path)}}).execute()
    out = os.path.splitext(path)[0] + ".ipynb"
    nbformat.write(nb, out)
    return out


if __name__ == "__main__":
    wanted = sys.argv[1:]
    for s in SCRIPTS:
        if not wanted or any(w in s for w in wanted):
            print("built", build(s))
