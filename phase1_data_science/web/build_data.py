"""Bundle the four per-task payloads (assets/t1..t4_data.json) into assets/data.js.

The dashboard is opened straight from disk (file://), where fetch() of a sibling
JSON file is blocked, so the payloads ship as one JavaScript global instead.
Run this after re-running any task script:

    python web/build_data.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")

bundle = {}
for key in ("t1", "t2", "t3", "t4"):
    with open(os.path.join(ASSETS, f"{key}_data.json"), encoding="utf-8") as f:
        bundle[key] = json.load(f)

with open(os.path.join(ASSETS, "data.js"), "w", encoding="utf-8") as f:
    f.write("window.PHASE1_DATA = " + json.dumps(bundle, separators=(",", ":")) + ";\n")

print("wrote", os.path.join(ASSETS, "data.js"))
