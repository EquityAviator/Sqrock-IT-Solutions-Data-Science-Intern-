"""Capture a PNG of every dashboard page into reports/screenshots/ (needs Google Chrome).

Start the app first, with the login screen off:
    set PHASE2_AUTH=off && streamlit run app.py --server.port 8765
then:
    python tools/screenshots.py
"""
from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "screenshots"
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765"
CHROME = next((p for p in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                           r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                           r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe") if Path(p).exists()), "chrome")
PORT = 9333
# name, path, page height, text of a tab to open first (or None)
PAGES = [("1_overview", "", 1500, None), ("2_stocks", "stocks", 1400, None), ("2_stocks_portfolio", "stocks", 1500, "Portfolio"),
         ("3_ai_analysis", "analysis", 1500, None), ("3_ai_insights", "analysis", 1500, "AI insights"),
         ("4_inventory", "inventory", 1750, None), ("4_inventory_labels", "inventory", 1300, "Labels (QR / barcode)"),
         ("5_loans", "loans", 1700, None), ("5_loans_model", "loans", 1700, "Model performance")]

READY = ("(() => { const busy = document.querySelector('[data-testid=stStatusWidget]'); "
         "const skel = document.querySelector('[data-testid=stSkeleton]'); "
         "return !!document.querySelector('[data-testid=stMetric], h1') && !busy && !skel; })()")


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
                             f"--remote-debugging-port={PORT}", f"--user-data-dir={tempfile.mkdtemp()}", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(40):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
                break
            except Exception:
                time.sleep(0.5)
        ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
        async with websockets.connect(ws_url, max_size=None) as ws:
            n = 0

            async def cmd(method: str, **params):
                nonlocal n
                n += 1
                await ws.send(json.dumps({"id": n, "method": method, "params": params}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == n:
                        return msg.get("result", {})

            async def js(expr: str):
                return (await cmd("Runtime.evaluate", expression=expr, returnByValue=True)).get("result", {}).get("value")

            async def settle(seconds: int = 60) -> None:
                quiet = 0
                for _ in range(seconds * 2):
                    await asyncio.sleep(0.5)
                    quiet = quiet + 1 if await js(READY) else 0
                    if quiet >= 5:                       # ready for 2.5 s in a row
                        return

            await cmd("Page.enable")
            for name, path, height, tab in PAGES:
                await cmd("Emulation.setDeviceMetricsOverride", width=1500, height=height, deviceScaleFactor=1, mobile=False)
                await cmd("Page.navigate", url=f"{BASE}/{path}")
                await settle()
                if tab:
                    ok = await js("(() => { const t = [...document.querySelectorAll('[role=tab], [data-baseweb=tab]')]"
                                  ".find(b => (b.textContent || '').trim() === " + json.dumps(tab) + "); "
                                  "if (t) { t.scrollIntoView(); t.click(); } return !!t; })()")
                    if not ok:
                        print(f"  tab not found: {tab}", flush=True)
                    await asyncio.sleep(1.5)
                    await settle(30)
                await asyncio.sleep(1.0)
                shot = await cmd("Page.captureScreenshot", format="png")
                (OUT / f"{name}.png").write_bytes(base64.b64decode(shot["data"]))
                print(f"saved {name}.png ({(OUT / f'{name}.png').stat().st_size / 1024:.0f} KB)", flush=True)
    finally:
        proc.terminate()


if __name__ == "__main__":
    asyncio.run(main())
