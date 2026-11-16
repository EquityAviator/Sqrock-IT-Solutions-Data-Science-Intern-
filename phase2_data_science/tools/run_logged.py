"""Run any command with a crash-proof log of everything that happens.

    python tools/run_logged.py task4 python task4_loan_approval/train.py
    python tools/run_logged.py app   python -m streamlit run app.py

Writes logs/<date>_<time>_<name>.log containing, in time order:
  * every line the command prints (stdout and stderr)
  * a system snapshot every few seconds: CPU %, RAM, pagefile, disk free, disk and
    network throughput, GPU load / memory / temperature, the command's own memory
  * new Windows System-log errors and warnings (driver and hardware trouble) as they appear
  * the previous crashes Windows recorded, at the top, for comparison

Every line is flushed AND fsync'ed to disk immediately, so if the PC blue-screens the
last line in the file is what was happening at that moment.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
SAMPLE_SECONDS = 3
EVENT_SECONDS = 15


class Log:
    def __init__(self, name: str):
        LOG_DIR.mkdir(exist_ok=True)
        self.path = LOG_DIR / f"{datetime.now():%Y-%m-%d_%H%M%S}_{name}.log"
        self.f = open(self.path, "a", encoding="utf-8", buffering=1)
        self.lock = threading.Lock()

    def write(self, tag: str, msg: str) -> None:
        line = f"{datetime.now():%H:%M:%S.%f}"[:-3] + f" [{tag:5s}] {msg.rstrip()}\n"
        with self.lock:
            self.f.write(line)
            self.f.flush()
            os.fsync(self.f.fileno())          # force it onto the disk: survives a blue screen


def powershell(cmd: str, timeout: int = 40) -> str:
    try:
        return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                              timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
    except Exception as exc:
        return f"(powershell failed: {exc})"


def gpu() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
        u, mu, mt, t, p = [x.strip() for x in out.split(",")]
        return f"gpu {u}% mem {mu}/{mt}MB {t}C {p}W"
    except Exception:
        return "gpu n/a"


def header(log: Log, cmd: list[str]) -> None:
    vm = psutil.virtual_memory()
    log.write("START", f"command: {' '.join(cmd)}")
    log.write("START", f"cwd: {os.getcwd()} | python: {sys.executable}")
    log.write("START", f"cpu threads {psutil.cpu_count()} | ram {vm.total / 2**30:.1f} GB | "
                       f"boot {datetime.fromtimestamp(psutil.boot_time()):%Y-%m-%d %H:%M:%S} | {gpu()}")
    for d in ("C:\\", "D:\\"):
        try:
            log.write("START", f"disk {d} free {psutil.disk_usage(d).free / 2**30:.1f} GB")
        except Exception:
            pass
    if sys.platform == "win32":
        prev = powershell("Get-WinEvent -FilterHashtable @{LogName='System'; Id=1001; ProviderName='Microsoft-Windows-WER-SystemErrorReporting'} "
                          "-MaxEvents 6 -ErrorAction SilentlyContinue | ForEach-Object { '{0:yyyy-MM-dd HH:mm:ss} {1}' -f $_.TimeCreated, "
                          "(($_.Message -replace '\\s+',' ').Substring(0,110)) }")
        for line in prev.splitlines():
            log.write("CRASH", "previous: " + line)
        nets = powershell("Get-NetAdapter | ForEach-Object { \"$($_.Name) | $($_.InterfaceDescription) | $($_.Status)\" }")
        for line in nets.splitlines():
            log.write("START", "network adapter: " + line)


def monitor(log: Log, proc: subprocess.Popen, stop: threading.Event) -> None:
    psutil.cpu_percent(None)
    last_disk, last_net, last_t = psutil.disk_io_counters(), psutil.net_io_counters(), time.time()
    try:
        me = psutil.Process(proc.pid)
    except Exception:
        me = None
    while not stop.wait(SAMPLE_SECONDS):
        try:
            now = time.time()
            dt = max(now - last_t, 1e-6)
            vm, sw = psutil.virtual_memory(), psutil.swap_memory()
            disk, net = psutil.disk_io_counters(), psutil.net_io_counters()
            rss, kids = 0, 0
            if me is not None and me.is_running():
                procs = [me] + me.children(recursive=True)
                kids = len(procs) - 1
                for p in procs:
                    try:
                        rss += p.memory_info().rss
                    except Exception:
                        pass
            log.write("SYS", f"cpu {psutil.cpu_percent(None):5.1f}% | ram {vm.percent:4.1f}% ({vm.available / 2**30:.1f} GB free) | "
                             f"pagefile {sw.percent:4.1f}% | {gpu()} | "
                             f"disk r/w {(disk.read_bytes - last_disk.read_bytes) / dt / 2**20:6.1f}/{(disk.write_bytes - last_disk.write_bytes) / dt / 2**20:6.1f} MB/s | "
                             f"net in/out {(net.bytes_recv - last_net.bytes_recv) / dt / 1024:7.1f}/{(net.bytes_sent - last_net.bytes_sent) / dt / 1024:7.1f} KB/s | "
                             f"job ram {rss / 2**20:6.0f} MB, {kids} child proc")
            last_disk, last_net, last_t = disk, net, now
        except Exception as exc:
            log.write("SYS", f"monitor error: {exc}")


def windows_events(log: Log, stop: threading.Event) -> None:
    """Copy new System-log errors/warnings (drivers, hardware, power) into our log."""
    if sys.platform != "win32":
        return
    since = datetime.now()
    seen: set[str] = set()
    while not stop.wait(EVENT_SECONDS):
        q = ("Get-WinEvent -FilterHashtable @{LogName='System'; Level=1,2,3; StartTime=[datetime]'" + since.strftime("%Y-%m-%d %H:%M:%S") +
             "'} -ErrorAction SilentlyContinue | Sort-Object TimeCreated | ForEach-Object { '{0:HH:mm:ss}|{1}|{2}|{3}' -f $_.TimeCreated, "
             "$_.ProviderName, $_.Id, (($_.Message -replace '\\s+',' ').Substring(0, [Math]::Min(160, ($_.Message -replace '\\s+',' ').Length))) }")
        for line in powershell(q).splitlines():
            if line and line not in seen:
                seen.add(line)
                log.write("WINEV", line)


def pump(log: Log, stream, tag: str) -> None:
    for raw in iter(stream.readline, b""):
        text = raw.decode("utf-8", errors="replace").rstrip()
        if text:
            log.write(tag, text)
            print(text, flush=True)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    name, cmd = sys.argv[1], sys.argv[2:]
    if cmd[0] == "python":
        cmd[0] = sys.executable                          # same interpreter as this wrapper
    log = Log(name)
    print(f"logging to {log.path}", flush=True)
    header(log, cmd)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(ROOT), env=env)
    log.write("START", f"process id {proc.pid}")
    stop = threading.Event()
    threads = [threading.Thread(target=monitor, args=(log, proc, stop), daemon=True),
               threading.Thread(target=windows_events, args=(log, stop), daemon=True),
               threading.Thread(target=pump, args=(log, proc.stdout, "OUT"), daemon=True),
               threading.Thread(target=pump, args=(log, proc.stderr, "ERR"), daemon=True)]
    for t in threads:
        t.start()
    t0 = time.time()
    try:
        code = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        code = -2
        log.write("END", "interrupted by user")
    threads[2].join(timeout=5)
    threads[3].join(timeout=5)
    stop.set()
    log.write("END", f"exit code {code} after {time.time() - t0:.1f}s")
    return code


if __name__ == "__main__":
    sys.exit(main())
