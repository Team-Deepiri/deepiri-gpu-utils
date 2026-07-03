"""Terminal and HTML visualization for GPU hardware snapshots.

Read-only: renders :func:`summary.hardware_summary` and inventory data as ASCII
dashboards or a self-contained HTML report. No external plotting dependencies.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import UTC, datetime

from .install_check import install_readiness
from .inventory import GPUInfo
from .summary import HardwareSummary, hardware_summary

_BAR_WIDTH = 24


@dataclass(frozen=True)
class DashboardRender:
    """Rendered dashboard text plus metadata."""

    text: str
    backend: str
    gpu_count: int
    doctor_status: str


def _bar(used: float, total: float, width: int = _BAR_WIDTH) -> str:
    if total <= 0:
        return "[" + ("?" * width) + "]"
    ratio = max(0.0, min(used / total, 1.0))
    filled = int(round(ratio * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + f"] {used:.1f}/{total:.1f} GB"


def _gpu_vram_bars(gpus: list[GPUInfo]) -> list[str]:
    lines: list[str] = []
    for gpu in gpus:
        label = gpu.name or "unknown GPU"
        idx = gpu.index if gpu.index is not None else "-"
        total = float(gpu.memory_gb or 0)
        free = gpu.details.get("memory_free_gb") if isinstance(gpu.details, dict) else None
        if isinstance(free, (int, float)) and total > 0:
            used = max(total - float(free), 0.0)
            lines.append(f"  GPU[{idx}] {label}")
            lines.append(f"    VRAM {_bar(used, total)}")
            util = gpu.utilization_percent
            if util is not None:
                lines.append(f"    util {_bar(float(util), 100.0, width=16)} %")
        elif total > 0:
            lines.append(f"  GPU[{idx}] {label}")
            lines.append(f"    VRAM {_bar(total, total)} (total only)")
        else:
            lines.append(f"  GPU[{idx}] {gpu.backend} {label} (memory unknown)")
    return lines


def _install_status_line(backend: str) -> str:
    item = install_readiness(device="auto")
    if item.backend != backend and backend in ("cuda", "rocm", "mps", "cpu"):
        device_map = {"cuda": "nvidia", "rocm": "amd", "mps": "apple", "cpu": "cpu"}
        item = install_readiness(device=device_map[backend])  # type: ignore[arg-type]
    status = "READY" if item.ready else "MISSING"
    missing = ", ".join(item.missing_required) if item.missing_required else "none"
    return f"install: {status} (missing: {missing})"


def render_dashboard(*, snap: HardwareSummary | None = None) -> DashboardRender:
    """Render an ASCII hardware dashboard."""

    s = snap or hardware_summary()
    d = s.detect
    width = 62
    border = "=" * width

    lines = [
        border,
        " deepiri-gpu dashboard".center(width),
        border,
        f" backend  : {d.backend} (confidence {d.confidence:.2f})",
        f" doctor   : {s.doctor.status}",
        f" torch    : {s.torch_device.device}",
        f" RAM      : {s.system_ram_gb} GB",
        f" VRAM tot : {s.total_vram_gb if s.total_vram_gb is not None else '?'} GB",
        f" GPUs     : {s.gpu_count}",
        f" docker   : {s.build_args.device_type} / {s.build_args.base_image}",
        _install_status_line(d.backend),
        "",
        " VRAM / utilization",
    ]
    if s.inventory.gpus:
        lines.extend(_gpu_vram_bars(s.inventory.gpus))
    else:
        lines.append("  (no GPUs inventoried — CPU-only or drivers missing)")

    if s.doctor.runbook:
        lines.append("")
        lines.append(" runbook")
        for note in s.doctor.runbook[:4]:
            lines.append(f"  - {note}")
    if s.notes:
        lines.append("")
        lines.append(" notes")
        for note in s.notes[:3]:
            lines.append(f"  - {note}")

    lines.append(border)
    return DashboardRender(
        text="\n".join(lines),
        backend=d.backend,
        gpu_count=s.gpu_count,
        doctor_status=s.doctor.status,
    )


def render_html_report(*, snap: HardwareSummary | None = None) -> str:
    """Return a self-contained HTML hardware report."""

    s = snap or hardware_summary()
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    d = s.detect

    def esc(value: object) -> str:
        return html.escape(str(value))

    gpu_rows = ""
    for gpu in s.inventory.gpus:
        free = gpu.details.get("memory_free_gb") if isinstance(gpu.details, dict) else "?"
        gpu_rows += (
            "<tr>"
            f"<td>{esc(gpu.index)}</td>"
            f"<td>{esc(gpu.backend)}</td>"
            f"<td>{esc(gpu.name)}</td>"
            f"<td>{esc(gpu.memory_gb)}</td>"
            f"<td>{esc(free)}</td>"
            f"<td>{esc(gpu.utilization_percent)}</td>"
            f"<td>{esc(gpu.driver_version)}</td>"
            "</tr>"
        )
    if not gpu_rows:
        gpu_rows = '<tr><td colspan="7">No GPUs detected</td></tr>'

    runbook_items = "".join(f"<li>{esc(line)}</li>" for line in s.doctor.runbook[:8])
    note_items = "".join(f"<li>{esc(line)}</li>" for line in s.notes)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>deepiri-gpu report</title>
  <style>
    body {{
      font-family: system-ui, sans-serif;
      margin: 2rem;
      background: #0f1419;
      color: #e6edf3;
    }}
    h1, h2 {{ color: #58a6ff; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
    }}
    .card {{
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 1rem;
    }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
    th, td {{ border: 1px solid #30363d; padding: 0.5rem; text-align: left; }}
    .meta {{ color: #8b949e; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <h1>deepiri-gpu hardware report</h1>
  <p class="meta">Generated {esc(generated)}</p>
  <div class="grid">
    <div class="card"><strong>Backend</strong><br>{esc(d.backend)}</div>
    <div class="card"><strong>Doctor</strong><br>{esc(s.doctor.status)}</div>
    <div class="card"><strong>GPUs</strong><br>{esc(s.gpu_count)}</div>
    <div class="card"><strong>RAM</strong><br>{esc(s.system_ram_gb)} GB</div>
    <div class="card"><strong>VRAM total</strong><br>{esc(s.total_vram_gb)} GB</div>
    <div class="card"><strong>Torch device</strong><br>{esc(s.torch_device.device)}</div>
  </div>
  <h2>GPU inventory</h2>
  <table>
    <tr><th>Index</th><th>Backend</th><th>Name</th><th>VRAM GB</th><th>Free GB</th>
        <th>Util %</th><th>Driver</th></tr>
    {gpu_rows}
  </table>
  <h2>Runbook</h2>
  <ul>{runbook_items or '<li>None</li>'}</ul>
  <h2>Notes</h2>
  <ul>{note_items or '<li>None</li>'}</ul>
</body>
</html>"""
