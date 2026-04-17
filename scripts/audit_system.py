#!/usr/bin/env python3
"""Cross-platform system environment audit.

Writes ``docs/system_audit_report.json`` and prints a human-readable summary.
Intended to be pasted into handoff documents so reviewers know exactly what
environment produced the numbers.

No network. No elevated permissions. Safe to run anywhere.
"""
from __future__ import annotations

import json
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / "docs" / "system_audit_report.json"


def _run(cmd: list[str], timeout: float = 5.0) -> str:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
        return (out.stdout or "").strip() or (out.stderr or "").strip()
    except Exception as e:  # noqa: BLE001
        return f"<error: {e}>"


def _gpu_info() -> dict:
    if shutil.which("nvidia-smi") is None:
        return {"present": False, "summary": "nvidia-smi not on PATH"}
    raw = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader"])
    return {"present": bool(raw) and "<error" not in raw, "summary": raw}


def _torch_info() -> dict:
    try:
        import torch

        return {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count())
                if torch.cuda.is_available() else 0,
            "cuda_version": getattr(torch.version, "cuda", None),
        }
    except Exception as e:  # noqa: BLE001
        return {"version": None, "error": str(e)}


def _package_versions() -> dict:
    pkgs = ["numpy", "pandas", "pyyaml", "scipy", "ultralytics", "fastapi",
            "pytest", "ruff", "torch", "torchvision", "loguru"]
    out: dict[str, str] = {}
    for p in pkgs:
        try:
            mod = __import__(p.replace("-", "_"))
            out[p] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[p] = "not-installed"
    return out


def _git_info() -> dict:
    branch = _run(["git", "-C", str(PROJECT_ROOT), "branch", "--show-current"])
    short = _run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"])
    status = _run(["git", "-C", str(PROJECT_ROOT), "status", "--short"])
    ahead = _run(["git", "-C", str(PROJECT_ROOT), "rev-list",
                  "--count", "main..HEAD"])
    return {
        "branch": branch,
        "head": short,
        "dirty_lines": 0 if not status else len(status.splitlines()),
        "commits_ahead_of_main": ahead,
    }


def _port_status(port: int) -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.3)
    try:
        r = s.connect_ex(("127.0.0.1", port))
        return "in-use" if r == 0 else "free"
    finally:
        s.close()


def _disk_and_ram() -> dict:
    total, used, free = shutil.disk_usage(str(PROJECT_ROOT))
    ram: dict = {}
    try:
        import psutil  # optional

        v = psutil.virtual_memory()
        ram = {"total_gb": round(v.total / 1e9, 2),
               "available_gb": round(v.available / 1e9, 2)}
    except Exception:
        ram = {"note": "psutil not installed"}
    return {
        "disk_total_gb": round(total / 1e9, 2),
        "disk_free_gb": round(free / 1e9, 2),
        "ram": ram,
    }


def build_report() -> dict:
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        },
        "git": _git_info(),
        "gpu": _gpu_info(),
        "torch": _torch_info(),
        "packages": _package_versions(),
        "resources": _disk_and_ram(),
        "ports": {"8000": _port_status(8000), "9000": _port_status(9000)},
    }


def _print_summary(r: dict) -> None:
    print("=" * 60)
    print(f"  agri-drone system audit  -  {r['generated_at']}")
    print("=" * 60)
    print(f"  OS         : {r['os']['platform']}")
    print(f"  Python     : {r['python']['version']}  venv={r['python']['venv']}")
    print(f"  Git        : {r['git']['branch']}@{r['git']['head']}  "
          f"ahead-of-main={r['git']['commits_ahead_of_main']}  "
          f"dirty={r['git']['dirty_lines']}")
    print(f"  GPU        : {r['gpu']['summary']}")
    t = r["torch"]
    print(f"  torch      : {t.get('version')}  "
          f"cuda_available={t.get('cuda_available')}  "
          f"cuda={t.get('cuda_version')}")
    print(f"  disk free  : {r['resources']['disk_free_gb']} GB")
    print(f"  RAM        : {r['resources']['ram']}")
    print(f"  ports      : 8000={r['ports']['8000']}  9000={r['ports']['9000']}")
    print("=" * 60)


def main() -> int:
    r = build_report()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(r, indent=2), encoding="utf-8")
    _print_summary(r)
    print(f"  wrote {OUT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
