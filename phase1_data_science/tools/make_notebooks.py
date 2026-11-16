"""Turn each task script into an executed Jupyter notebook next to it.

The scripts stay the single source of truth; this splits them into cells at
their "# ---- N. STEP" section markers, makes every saved chart also display
inline, executes the notebook and writes <script>.ipynb beside the script.

    python tools/make_notebooks.py            # all four tasks
    python tools/make_notebooks.py task2      # just one
"""
import os
import re
import sys

import nbformat
from nbclient import NotebookClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = [
    "task1_sales_forecasting/sales_forecasting.py",
    "task2_student_performance/student_performance.py",
    "task3_resume_screening/resume_screening.py",
    "task4_churn_prediction/churn_prediction.py",
]
MARKER = re.compile(r"^# -{10,}\s*(.+?)\s*$")

SHOW_CHARTS = '''# Show every chart inline as well as saving it to artifacts/
from IPython.display import Image, display
_save_to_disk = save
def save(fig, name):
    _save_to_disk(fig, name)
    display(Image(os.path.join(ART, name)))'''


def build(script_rel: str) -> str:
    path = os.path.join(ROOT, script_rel)
    src = open(path, encoding="utf-8").read()

    # module docstring -> markdown intro
    doc = re.match(r'\s*"""(.*?)"""\s*', src, re.S)
    body = src[doc.end():] if doc else src
    intro = doc.group(1).strip().splitlines() if doc else []
    title = intro[0].title() if intro else os.path.basename(script_rel)
    md = f"# {title}\n\n" + "\n".join(intro[1:]).replace("Run:  python", "Script version:")

    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells.append(nbformat.v4.new_markdown_cell(md))
    # scripts locate data relative to __file__, which a notebook does not have
    nb.cells.append(nbformat.v4.new_code_cell(
        "import os\n"
        f"__file__ = os.path.abspath(\"{os.path.basename(path)}\")  "
        "# run from the task folder; lets the script's relative paths resolve"))

    chunk, heading = [], None

    def flush():
        code = "\n".join(chunk).strip("\n")
        if heading:
            nb.cells.append(nbformat.v4.new_markdown_cell(f"## {heading.title()}"))
        if code.strip():
            nb.cells.append(nbformat.v4.new_code_cell(code))
            if "def save(" in code:
                nb.cells.append(nbformat.v4.new_code_cell(SHOW_CHARTS))

    for line in body.splitlines():
        m = MARKER.match(line)
        if m:
            flush()
            chunk, heading = [], m.group(1)
        else:
            chunk.append(line)
    flush()

    NotebookClient(nb, timeout=1800, kernel_name="python3",
                   resources={"metadata": {"path": os.path.dirname(path)}}).execute()
    out = os.path.splitext(path)[0] + ".ipynb"
    nbformat.write(nb, out)
    return out


if __name__ == "__main__":
    wanted = sys.argv[1:]
    for s in SCRIPTS:
        if not wanted or any(w in s for w in wanted):
            print("built", build(s))
