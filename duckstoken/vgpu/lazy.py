"""Optional sibling modules.

The runtime modules (benchmark, scheduler, optimizer, prewarm) use the planner, telemetry, pressure and probe modules
when they can be imported and degrade to "unavailable" labels when they cannot, so one broken or missing piece never
takes the benchmark down with it.
"""

from __future__ import annotations

import importlib


def module(name: str):
    """``duckstoken.vgpu.<name>`` or None when it cannot be imported."""
    try:
        return importlib.import_module(f"{__package__}.{name}")
    except Exception:  # noqa: BLE001 - a syntax error in a sibling must not break the caller
        return None
