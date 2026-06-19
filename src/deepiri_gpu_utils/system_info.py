"""Read-only host facts (RAM, WSL, DMI, Docker) for doctor / ollama / detect."""

from __future__ import annotations

import os
import platform
import re
import shutil
from typing import Any

from ._subprocess import run_text

_NVIDIA_PCI_RE = re.compile(r"nvidia|vga.*nvidia", re.IGNORECASE)


def is_wsl() -> bool:
    rel = platform.release().lower()
    if "microsoft" in rel or "wsl" in rel:
        return True
    try:
        with open("/proc/version", encoding="utf-8", errors="replace") as f:
            if "microsoft" in f.read().lower():
                return True
    except OSError:
        pass
    return False


def system_ram_gb() -> int:
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        kb = int(parts[1])
                        return max(kb // (1024 * 1024), 0)
        except (OSError, ValueError, IndexError):
            return 0
    if platform.system() == "Darwin":
        res = run_text(["sysctl", "-n", "hw.memsize"], timeout=5)
        if res.ok and res.stdout.strip():
            try:
                return max(int(res.stdout.strip()) // (1024**3), 0)
            except ValueError:
                return 0
    return 0


def lspci_nvidia_present() -> bool | None:
    """Return True if lspci shows NVIDIA; None if lspci missing or failed."""

    if platform.system() != "Linux" or not shutil.which("lspci"):
        return None
    res = run_text(["lspci"], timeout=10)
    if not res.ok or not res.stdout:
        return None
    for line in res.stdout.splitlines():
        if _NVIDIA_PCI_RE.search(line):
            return True
    return False


def docker_cli_available() -> bool:
    return shutil.which("docker") is not None


def nvidia_container_toolkit_hint() -> dict[str, Any]:
    """Best-effort signals for NVIDIA Container Toolkit (no package manager calls)."""

    out: dict[str, Any] = {
        "nvidia_ctk_on_path": shutil.which("nvidia-ctk") is not None,
    }
    for path in (
        "/usr/bin/nvidia-container-runtime",
        "/usr/bin/nvidia-container-toolkit",
    ):
        if os.path.isfile(path):
            out["nvidia_container_binary"] = path
            break
    return out


def dmidecode_inventory() -> dict[str, Any]:
    """Best-effort SMBIOS/DMI strings (Linux, usually requires root)."""

    if platform.system() != "Linux" or not shutil.which("dmidecode"):
        return {"available": False, "reason": "dmidecode not installed or not Linux"}
    if os.geteuid() != 0:
        return {"available": False, "reason": "dmidecode typically requires root"}

    fields = (
        ("system_manufacturer", "system-manufacturer"),
        ("system_product_name", "system-product-name"),
        ("system_serial", "system-serial-number"),
    )
    data: dict[str, str] = {}
    for key, arg in fields:
        res = run_text(["dmidecode", "-s", arg], timeout=8)
        if res.ok and res.stdout.strip():
            data[key] = res.stdout.strip()
    if data:
        return {"available": True, **data}
    return {"available": False, "reason": "dmidecode produced no data"}
