"""How many candidates may be probed at once, and why that number.

Reverse Linear Scan is optimal in *oracle calls*: finding the newest version
that passes requires probing every version above the answer, so no schedule asks
fewer questions. What it does not fix is that the questions are asked one at a
time, each one a full install-and-validate lasting minutes.

Batching asks k of them at once. Exactness survives because a batch is a
contiguous prefix of the newest-first order -- exactly the versions a serial
scan would have visited next -- and the highest passing member is taken. If a
serial scan would have stopped at v, every version above v is in some earlier or
the same batch and failed there, so the batch returns v too.

**k is a resource bound, not a tuned hyperparameter.** Each concurrent probe is
a full environment: its own virtualenv, its own installed copy of the package,
its own test process. So k is whatever the machine can hold, computed from

* the release's **measured wheel size** (from the index, not guessed),
* an **installed-footprint multiplier** measured at 3.8x for numpy and higher
  for small packages, where fixed per-package overhead dominates -- 4.0 is used
  because it is the large-package figure, and only large packages ever bind,
* the machine's detected cores, free disk and memory, minus a stated reserve.

Nothing here is fitted to a subject, and no value is chosen after seeing a
result. The same formula on a 2-core CI runner returns k=1, which is the honest
reason AURA runs faster on a workstation than in GitHub Actions.

The exactness claim is not left to argument. ``k=1`` versus ``k=production`` is
run as a paired experiment, and the finding that matters is not the speedup but
the row above it: identical final configurations. A saving nobody verified is a
saving nobody should believe.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

# Measured: installed bytes / wheel bytes. numpy 3.80x, requests 6.70x,
# click 8.47x -- small packages are dominated by dist-info and __pycache__
# overhead. Large packages are the only ones that ever bind the batch width,
# so their figure is the one used.
INSTALL_MULTIPLIER = 4.0

# Below this, per-package overhead swamps the wheel size and the ratio stops
# meaning anything; treat every small package as costing the same little.
MIN_FOOTPRINT_MB = 32.0

# A probe is an install plus a test run, so it is not purely CPU-bound; half the
# cores keeps the machine responsive and avoids each probe slowing the others
# enough to distort the wall-clock comparison.
CPU_FRACTION = 0.5


@dataclass(frozen=True)
class Resources:
    """What this machine can actually spare. Detected, then recorded in the log."""

    cpus: int = 1
    free_disk_mb: float = 0.0
    total_memory_mb: float = 0.0
    disk_reserve_mb: float = 5_000.0
    memory_reserve_mb: float = 4_000.0
    cpu_source: str = "os"
    memory_source: str = "os"

    @classmethod
    def detect(cls, path: str = ".", env: dict | None = None,
               cgroup_root: str = "/sys/fs/cgroup") -> "Resources":
        """Measure what *this job* may use, not what the machine has.

        On a shared node a scheduler hands the job a slice: 8 cores of a
        64-core machine, 32 GB of 512. The machine's totals are the wrong
        numbers -- sizing batches from them runs more concurrent installs than
        the job has cores or memory for. So every limit that applies is read
        and the tightest one wins: Slurm's allocation, the CPUs this process is
        pinned to, and any cgroup cap, falling back to the machine's totals only
        when nothing narrower is set.
        """
        env = os.environ if env is None else env
        # The disk the environments are built on, which is not the workspace's
        # when AURA_VENV_ROOT moves them to a node's local disk.
        path = env.get("AURA_VENV_ROOT") or path
        try:
            free_disk = shutil.disk_usage(path).free / 1e6
        except Exception:
            free_disk = 0.0
        cpus, cpu_source = allocated_cpus(env, cgroup_root)
        memory, memory_source = allocated_memory_mb(env, cgroup_root, cpus)
        return cls(cpus=cpus, free_disk_mb=free_disk, total_memory_mb=memory,
                   cpu_source=cpu_source, memory_source=memory_source)

    @property
    def usable_disk_mb(self) -> float:
        return max(0.0, self.free_disk_mb - self.disk_reserve_mb)

    @property
    def usable_memory_mb(self) -> float:
        return max(0.0, self.total_memory_mb - self.memory_reserve_mb)

    def to_dict(self) -> dict:
        return {"cpus": self.cpus, "cpu_source": self.cpu_source,
                "memory_source": self.memory_source,
                "free_disk_mb": round(self.free_disk_mb, 1),
                "total_memory_mb": round(self.total_memory_mb, 1),
                "usable_disk_mb": round(self.usable_disk_mb, 1),
                "usable_memory_mb": round(self.usable_memory_mb, 1)}


def _first_int(text: str | None) -> int | None:
    """First integer in a value such as "8", "8(x2)" or "4,8"."""
    if not text:
        return None
    digits = ""
    for ch in str(text):
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    return int(digits) if digits else None


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return None


def allocated_cpus(env: dict, cgroup_root: str = "/sys/fs/cgroup") -> tuple[int, str]:
    """The number of CPUs this process may use, and where that number came from."""
    limits: list[tuple[int, str]] = []

    per_task = _first_int(env.get("SLURM_CPUS_PER_TASK"))
    if per_task:
        limits.append((per_task, "slurm"))
    elif (on_node := _first_int(env.get("SLURM_JOB_CPUS_PER_NODE"))):
        limits.append((on_node, "slurm"))

    try:
        limits.append((len(os.sched_getaffinity(0)), "affinity"))   # Linux only
    except (AttributeError, OSError):
        pass

    quota = _read(f"{cgroup_root}/cpu.max")                      # cgroup v2
    if quota and not quota.startswith("max"):
        parts = quota.split()
        if len(parts) == 2 and parts[1].isdigit() and parts[0].isdigit():
            limits.append((max(1, -(-int(parts[0]) // int(parts[1]))), "cgroup"))

    if not limits:
        return max(1, os.cpu_count() or 1), "os"
    return min(limits)


def allocated_memory_mb(env: dict, cgroup_root: str = "/sys/fs/cgroup",
                        cpus: int = 1) -> tuple[float, str]:
    """The memory this process may use, in MB, and where that number came from."""
    limits: list[tuple[float, str]] = []
    try:
        limits.append(((os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 1e6,
                       "os"))
    except (ValueError, AttributeError, OSError):
        pass

    # Slurm states these in MB.
    if (per_node := _first_int(env.get("SLURM_MEM_PER_NODE"))):
        limits.append((float(per_node), "slurm"))
    elif (per_cpu := _first_int(env.get("SLURM_MEM_PER_CPU"))):
        limits.append((float(per_cpu * max(1, cpus)), "slurm"))

    for path in (f"{cgroup_root}/memory.max",                         # cgroup v2
                 f"{cgroup_root}/memory/memory.limit_in_bytes"):      # cgroup v1
        value = _read(path)
        if value and value.isdigit():
            limits.append((int(value) / 1e6, "cgroup"))

    if not limits:
        return 0.0, "unknown"
    return min(limits)


def footprint_mb(wheel_mb: float | None) -> float:
    """Installed size of one candidate, from its measured wheel size."""
    if not wheel_mb or wheel_mb <= 0:
        return MIN_FOOTPRINT_MB
    return max(MIN_FOOTPRINT_MB, wheel_mb * INSTALL_MULTIPLIER)


def size_bucket(wheel_mb: float | None) -> str:
    """Descriptive label for reporting. Does not feed the arithmetic."""
    if wheel_mb is None:
        return "unknown"
    if wheel_mb >= 50:
        return "large"
    if wheel_mb >= 5:
        return "medium"
    return "small"


@dataclass(frozen=True)
class BatchPolicy:
    """Turns a package's measured size into a batch width for this machine."""

    resources: Resources
    k_max: int = 6
    enabled: bool = True
    override: int = 0          # >0 pins k, for the k=1 determinism arm

    def plan(self, wheel_mb: float | None) -> dict:
        """Return the width and the arithmetic behind it, so the log can show both."""
        if not self.enabled:
            return {"k": 1, "reason": "batching disabled", "bucket": size_bucket(wheel_mb)}
        if self.override:
            return {"k": max(1, self.override), "reason": "pinned by configuration",
                    "bucket": size_bucket(wheel_mb)}

        per_probe = footprint_mb(wheel_mb)
        by_cpu = max(1, int(self.resources.cpus * CPU_FRACTION))
        by_disk = int(self.resources.usable_disk_mb // per_probe) if per_probe else 0
        by_memory = int(self.resources.usable_memory_mb // per_probe) if per_probe else 0

        limits = {"cpu": by_cpu, "disk": max(0, by_disk), "memory": max(0, by_memory),
                  "k_max": self.k_max}
        k = max(1, min(limits.values()))
        binding = min(limits, key=limits.get)
        return {"k": k, "bucket": size_bucket(wheel_mb),
                "wheel_mb": wheel_mb, "footprint_mb": round(per_probe, 1),
                "limits": limits, "binding_constraint": binding,
                "reason": f"limited by {binding}"}

    def k_for(self, wheel_mb: float | None) -> int:
        return self.plan(wheel_mb)["k"]


def batches(versions: list[str], k: int) -> list[list[str]]:
    """Split a newest-first candidate list into contiguous groups of k.

    Contiguity is what preserves exactness: each group is the block of versions a
    serial scan would visit next, so taking the highest passing member of the
    first group that contains one returns exactly the serial answer.
    """
    if k <= 1:
        return [[v] for v in versions]
    return [versions[i:i + k] for i in range(0, len(versions), k)]
