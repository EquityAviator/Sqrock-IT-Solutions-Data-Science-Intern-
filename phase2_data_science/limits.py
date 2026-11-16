"""Keep the pipelines gentle on the machine. Import this BEFORE numpy / torch / sklearn.

    from limits import N_JOBS

What it does
  * uses about 80% of the CPU threads, so the PC stays responsive
  * runs the process at below-normal priority, so other programs always win the CPU
  * caps GPU memory at 90% of the card, leaving room for the desktop

Override the thread count with the environment variable PHASE2_JOBS, e.g. PHASE2_JOBS=2.
"""
from __future__ import annotations

import os
import sys

_cpus = os.cpu_count() or 2
N_JOBS = max(1, int(os.environ.get("PHASE2_JOBS", int(_cpus * 0.8))))

# numerical libraries read these when they are first imported
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, str(N_JOBS))

if sys.platform == "win32":
    try:
        import ctypes
        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
else:
    try:
        os.nice(5)
    except Exception:
        pass


def limit_torch(gpu_fraction: float = 0.9) -> None:
    """Call once after importing torch."""
    import torch

    torch.set_num_threads(N_JOBS)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(gpu_fraction)
