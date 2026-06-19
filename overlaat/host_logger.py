#!/usr/bin/env python3
"""Overlaat host sampler (OPTIONAL, macOS-only).

Samples CPU/GPU/RAM every INTERVAL_S seconds and writes one row to the
`host_samples` table (Postgres — the same DB the gateway logs to). Each row
carries host totals (mem/cpu/gpu) plus a per-backend RSS breakdown
(`backends_json`) — the memory holders that actually fill the machine's RAM.

This component is OPTIONAL. Overlaat's core value (fair queueing + one honest
lifecycle event per request) does not depend on it; the host sampler only adds
host-level context (who is holding memory, host GPU%) alongside the request
events. It is macOS-specific: it shells out to `powermetrics`, `vm_stat`, and
`ps`. On other platforms you would replace this module with an equivalent
sampler (e.g. nvidia-smi + /proc on Linux) writing the same `host_samples` rows.

Why RSS and not per-process GPU: on macOS, per-process GPU is not measurable for
Metal/MLX workloads (powermetrics --show-process-gpu reports 0 ms/s). RSS IS
measurable per process, and RSS is what causes the Metal OOM on a memory-bound
box. So we attribute MEMORY by RSS and leave GPU% host-wide.

To read powermetrics it needs root. Run it as root (e.g. via a system service /
launch daemon) or grant the invoking user passwordless sudo for powermetrics.
Stdlib-only (no venv) -> it writes to Postgres via the `psql` CLI. The DB is
expected to be up; there is no on-disk fallback.

Environment overrides:
  METRICS_DB_URL   Postgres connection URL (else read DATABASE_URL from the env
                   file at OVERLAAT_ENV, default ./overlaat.env).
  OVERLAAT_ENV     path to an env file holding DATABASE_URL (default
                   ./overlaat.env).
  PSQL             path to the psql binary (default: "psql" on PATH).
  SLOT_RUNNING_URL optional URL of a model-swap server's /running endpoint, used
                   only for cold-load tracking (default off).
"""

from __future__ import annotations  # support older Python 3.9 (PEP 604 unions)

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

INTERVAL_S = 5  # 5s sampling (small, but ~12x denser than 60s)
POWERMETRICS_DUR_MS = 1000  # 1s sample inside each loop
PAGE_SIZE = 16384  # Apple Silicon page size
TOTAL_MEM_GB = float(os.environ.get("TOTAL_MEM_GB", "256.0"))  # host RAM size
RSS_MIN_GB = 1.0  # only record processes holding >= this much RAM
RSS_TOP_N = 20  # cap the per-sample backend list

# psql binary: default to whatever is on PATH; override with PSQL.
PSQL = os.environ.get("PSQL", "psql")

# Env file holding DATABASE_URL (kept out of this repo). Override with OVERLAAT_ENV.
OVERLAAT_ENV = Path(os.environ.get("OVERLAAT_ENV", "./overlaat.env"))

# Process names whose RSS we care to attribute. macOS `ps` exposes the full
# command line, so we match on the executable basename derived in derive_name().
# Adapt this list to the backends you actually run (inference servers, runtimes,
# the gateway, the database). It is only used to recognize port-suffixed names;
# any process over RSS_MIN_GB is still recorded regardless.
BACKEND_EXE_PREFIXES = (
    "python",  # e.g. an mlx/transformers server launched via python
    "ollama",  # a local model server + its runner subprocesses
    "model-server",  # a custom inference engine binary (rename to yours)
    "mlx_lm",  # an MLX language-model server
)

# Optional: a model-swap server's /running endpoint for cold-load tracking.
# Empty (the default) disables cold-load tracking entirely.
SLOT_RUNNING_URL = os.environ.get("SLOT_RUNNING_URL", "")


def database_url() -> str:
    """METRICS_DB_URL env override, else DATABASE_URL from the env file."""
    env = os.environ.get("METRICS_DB_URL")
    if env:
        return env
    if OVERLAAT_ENV.exists():
        for line in OVERLAAT_ENV.read_text().splitlines():
            if line.startswith("DATABASE_URL="):
                return line.split("=", 1)[1].strip()
    return ""


DB_URL = database_url()


def vm_stat() -> dict:
    out = subprocess.check_output(["/usr/bin/vm_stat"], text=True)
    pages = {}
    for line in out.splitlines():
        m = re.match(r"^([^:]+):\s+(\d+)\.?$", line.strip())
        if m:
            pages[m.group(1).lower().replace(" ", "_")] = int(m.group(2))

    def gb(key: str) -> float:
        return round(pages.get(key, 0) * PAGE_SIZE / 1024**3, 2)

    return {
        "total_gb": TOTAL_MEM_GB,
        "free_gb": gb("pages_free"),
        "active_gb": gb("pages_active"),
        "inactive_gb": gb("pages_inactive"),
        "wired_gb": gb("pages_wired_down"),
        "compressed_gb": gb("pages_occupied_by_compressor"),
        "speculative_gb": gb("pages_speculative"),
    }


def loadavg() -> dict:
    out = subprocess.check_output(["/usr/bin/uptime"], text=True)
    m = re.search(r"load averages?:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", out)
    if not m:
        return {}
    return {"load1": float(m.group(1)), "load5": float(m.group(2)), "load15": float(m.group(3))}


def powermetrics_sample() -> tuple[dict, list[dict]]:
    """Return (gpu_summary, [per_process_gpu]). Per-process GPU is kept only for
    annotating the backend list; it is ~always 0 for Metal/MLX (see module doc)."""
    cmd = [
        "/usr/bin/powermetrics",
        "--samplers",
        "gpu_power,tasks",
        "--show-process-gpu",
        "-i",
        str(POWERMETRICS_DUR_MS),
        "-n",
        "1",
    ]
    if os.geteuid() != 0:
        # As a normal user, go through sudo -n (requires passwordless sudo for
        # powermetrics, e.g. a NOPASSWD sudoers entry).
        cmd = ["/usr/bin/sudo", "-n", *cmd]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=15)
    except Exception:
        return {"active_pct": None, "freq_mhz": None}, []

    gpu_pct = None
    gpu_freq = None
    for line in out.splitlines():
        if "GPU HW active residency" in line:
            m = re.search(r"(\d+\.\d+)%", line)
            if m:
                gpu_pct = float(m.group(1))
        elif "GPU HW active frequency" in line:
            m = re.search(r"(\d+)\s*MHz", line)
            if m:
                gpu_freq = int(m.group(1))

    procs = []
    in_table = False
    for line in out.splitlines():
        if re.match(r"^Name\s+ID\s+CPU", line):
            in_table = True
            continue
        if in_table:
            if not line.strip() or line.startswith("***") or line.startswith("ALL_"):
                in_table = False
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                pid = int(parts[1])
                gpu_ms_per_s = float(parts[-1])  # ms/s -> % via /10
                if gpu_ms_per_s > 0:
                    procs.append(
                        {"name": parts[0], "pid": pid, "gpu_pct": round(gpu_ms_per_s / 10, 1)}
                    )
            except (ValueError, IndexError):
                continue
    return {"active_pct": gpu_pct, "freq_mhz": gpu_freq}, procs


def derive_name(cmd: str) -> str:
    """Build a readable process name from a full command line.

    Examples:
      /System/Library/.../WindowServer -daemon            -> WindowServer
      /opt/model-server --host ... --port 8086            -> model-server-8086
      /Applications/Ollama.app/.../ollama runner --model X -> ollama-runner-X
      /.../python3 .../uvicorn server:app --port 8083     -> uvicorn-8083
    """
    if not cmd:
        return "?"
    if cmd.startswith("postgres:"):
        rest = cmd[len("postgres:") :].strip().split()
        return f"postgres-{rest[0]}" if rest else "postgres"
    if cmd.startswith("sshd-session"):
        return "sshd-session"
    tokens = cmd.split()
    exe = os.path.basename(tokens[0]) or "?"
    port = None
    for i, t in enumerate(tokens):
        if t == "--port" and i + 1 < len(tokens):
            port = tokens[i + 1]
            break
    if exe == "ollama" and len(tokens) > 1:
        if tokens[1] == "runner":
            for i, t in enumerate(tokens):
                if t == "--model" and i + 1 < len(tokens):
                    return f"ollama-runner-{tokens[i + 1]}"
            return "ollama-runner"
        if tokens[1] == "serve":
            return "ollama-serve"
    if port and exe.startswith(BACKEND_EXE_PREFIXES):
        if exe.startswith("python"):
            # A python-hosted server: try to recover a meaningful name from an
            # absolute script path ending in -server / -api.
            for t in tokens[1:]:
                if t.startswith("/"):
                    for d in t.split("/"):
                        if d.endswith(("-server", "-api")):
                            return f"{d}-{port}"
        return f"{exe}-{port}"
    return exe


def ps_snapshot() -> dict[int, dict]:
    """pid -> {pid, name, cpu_pct, rss_gb, gpu_pct}. Uses full `command` (macOS
    `comm` is truncated to 16 chars)."""
    out = subprocess.check_output(["/bin/ps", "-axo", "pid,%cpu,rss,command"], text=True)
    processes: dict[int, dict] = {}
    for line in out.splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
            cpu = float(parts[1])
            rss_kb = int(parts[2])
            cmd = parts[3]
        except ValueError:
            continue
        processes[pid] = {
            "pid": pid,
            "name": derive_name(cmd),
            "cpu_pct": cpu,
            "rss_gb": round(rss_kb / 1024**2, 2),
            "gpu_pct": 0.0,
        }
    return processes


def backend_breakdown(processes: dict[int, dict], gpu_procs: list[dict]) -> list[dict]:
    """Memory holders: processes with RSS >= RSS_MIN_GB, sorted desc, top N.
    GPU% merged in where powermetrics saw it (usually 0 for Metal/MLX)."""
    for gp in gpu_procs:
        p = processes.get(gp["pid"])
        if p:
            p["gpu_pct"] = gp["gpu_pct"]
    holders = [p for p in processes.values() if p["rss_gb"] >= RSS_MIN_GB]
    holders.sort(key=lambda p: -p["rss_gb"])
    return holders[:RSS_TOP_N]


def sample() -> dict:
    t_epoch = time.time()
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    mem = vm_stat()
    cpu = loadavg()
    gpu_stats, gpu_procs = powermetrics_sample()
    ps_data = ps_snapshot()
    backends = backend_breakdown(ps_data, gpu_procs)
    return {
        "ts": ts,
        "ts_epoch": round(t_epoch, 3),
        "interval_s": INTERVAL_S,
        "cpu": cpu,
        "mem": mem,
        "gpu": gpu_stats,
        "backends": backends,
    }


def _num(x) -> str:
    """SQL literal for a numeric column (NULL if None)."""
    return "NULL" if x is None else repr(float(x))


def write_sample_pg(rec: dict) -> None:
    """INSERT one host sample via psql. Raises on failure (caller logs)."""
    if not DB_URL:
        raise RuntimeError("no DATABASE_URL")
    mem, gpu, cpu = rec["mem"], rec["gpu"], rec["cpu"]
    backends = json.dumps(rec["backends"], ensure_ascii=False).replace("'", "''")
    sql = (
        "INSERT INTO host_samples "
        "(ts,gpu_pct,gpu_freq_mhz,ram_total_gb,ram_wired_gb,ram_active_gb,"
        "ram_inactive_gb,ram_compressed_gb,ram_free_gb,cpu_load1,cpu_load5,"
        "backends_json) VALUES ("
        f"{rec['ts_epoch']:.3f},{_num(gpu.get('active_pct'))},"
        f"{_num(gpu.get('freq_mhz'))},{_num(mem.get('total_gb'))},"
        f"{_num(mem.get('wired_gb'))},{_num(mem.get('active_gb'))},"
        f"{_num(mem.get('inactive_gb'))},{_num(mem.get('compressed_gb'))},"
        f"{_num(mem.get('free_gb'))},{_num(cpu.get('load1'))},"
        f"{_num(cpu.get('load5'))},'{backends}'::jsonb) "
        "ON CONFLICT (ts) DO NOTHING;"
    )
    subprocess.run(
        [PSQL, DB_URL, "-v", "ON_ERROR_STOP=1", "-q", "-c", sql],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )


# -- cold-load tracking (optional) ----------------------------------------------
# If you run a model-swap server that loads one large model at a time, its
# /running endpoint reports each member's state (starting -> ready -> [evicted]).
# We poll it every loop and emit an explicit load row on the starting->ready
# transition, so cold-load time stops hiding inside the first request's TTFT.
# State is held in module memory across loop iterations. Disabled when
# SLOT_RUNNING_URL is empty.
_load_state: dict[str, dict] = {}  # model -> {"state": str, "t_start": float|None}


def poll_slot_server() -> dict[str, str]:
    """{model: state} for currently-running swap members; {} if unreachable or
    if cold-load tracking is disabled (SLOT_RUNNING_URL unset)."""
    if not SLOT_RUNNING_URL:
        return {}
    try:
        with urllib.request.urlopen(SLOT_RUNNING_URL, timeout=2) as r:
            data = json.loads(r.read().decode())
        return {m["model"]: (m.get("state") or "") for m in data.get("running", [])}
    except Exception:
        return {}


def write_model_load_pg(model: str, t_start: float, t_ready: float | None, detail: str) -> None:
    """INSERT one cold-load row via psql. Raises on failure (caller logs)."""
    if not DB_URL:
        raise RuntimeError("no DATABASE_URL")
    load_s = None if t_ready is None else round(t_ready - t_start, 2)
    m = model.replace("'", "''")
    d = detail.replace("'", "''")
    sql = (
        "INSERT INTO model_loads (model,t_start,t_ready,load_s,detail) "
        f"VALUES ('{m}',{t_start:.3f},{_num(t_ready)},{_num(load_s)},'{d}');"
    )
    subprocess.run(
        [PSQL, DB_URL, "-v", "ON_ERROR_STOP=1", "-q", "-c", sql],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )


def track_model_loads(now_epoch: float) -> None:
    """Poll the swap server and emit a load row when a model reaches ready after
    a non-ready period. Best-effort: a failed poll or write is swallowed
    (logged). No-op when cold-load tracking is disabled."""
    cur = poll_slot_server()
    if not cur:
        return
    for model, state in cur.items():
        prev = _load_state.get(model)
        if prev is None:
            # First sighting: only arm a load if it's mid-startup right now.
            _load_state[model] = {
                "state": state,
                "t_start": None if state == "ready" else now_epoch,
            }
            continue
        if state == prev["state"]:
            continue
        if state != "ready" and prev["t_start"] is None:
            prev["t_start"] = now_epoch  # load began
        elif state == "ready" and prev["t_start"] is not None:
            try:
                write_model_load_pg(model, prev["t_start"], now_epoch, f"{prev['state']}->ready")
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(f"model-load write failed: {type(e).__name__}: {e}\n")
                sys.stderr.flush()
            prev["t_start"] = None  # load done
        prev["state"] = state
    # Drop models no longer running (evicted) so a future reload re-arms cleanly.
    for model in [m for m in _load_state if m not in cur]:
        del _load_state[model]


_stop = False


def _on_term(_signum, _frame):
    global _stop
    _stop = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--once", action="store_true", help="one sample -> pretty JSON to stdout (preview mode)"
    )
    args = p.parse_args()

    if args.once:
        print(json.dumps(sample(), ensure_ascii=False, indent=2))
        return

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    sys.stderr.write(
        f"host-logger started, interval={INTERVAL_S}s, db={'yes' if DB_URL else 'NO'}\n"
    )
    sys.stderr.flush()
    while not _stop:
        t0 = time.monotonic()
        try:
            write_sample_pg(sample())
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"sample failed: {type(e).__name__}: {e}\n")
            sys.stderr.flush()
        try:
            track_model_loads(time.time())
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"model-load track failed: {type(e).__name__}: {e}\n")
            sys.stderr.flush()
        elapsed = time.monotonic() - t0
        time.sleep(max(1.0, INTERVAL_S - elapsed))
    sys.stderr.write("host-logger stopping cleanly\n")


if __name__ == "__main__":
    main()
