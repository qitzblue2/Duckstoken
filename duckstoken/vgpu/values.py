"""Honest values: every number shown to the user says whether it was measured, estimated, experimental or unavailable.

A value is a plain dict ``{"value": <number|str|None>, "status": <label>, "source": <where it came from>, "note": <str>}``.
Unavailable values carry ``value=None`` and a ``note`` that says why. Nothing in this package may invent a number to fill a
gap: when a source cannot be read the answer is `unavailable(...)`.
"""

from __future__ import annotations

MEASURED = "measured"        # read from the OS, a driver tool or the backend during this session
ESTIMATED = "estimated"      # computed from a formula or table; the assumptions are listed next to it
EXPERIMENTAL = "experimental"  # measured or computed by a method that is not yet validated on real hardware
UNAVAILABLE = "unavailable"  # the source does not exist on this machine or failed; value is None

STATUSES = (MEASURED, ESTIMATED, EXPERIMENTAL, UNAVAILABLE)


def val(value, status: str = MEASURED, source: str = "", note: str = "") -> dict:
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    if value is None and status != UNAVAILABLE:
        status = UNAVAILABLE
    return {"value": value, "status": status, "source": source, "note": note}


def measured(value, source: str = "", note: str = "") -> dict:
    return val(value, MEASURED, source, note)


def est(value, source: str = "", note: str = "") -> dict:
    return val(value, ESTIMATED, source, note)


def unavailable(reason: str = "", source: str = "") -> dict:
    return {"value": None, "status": UNAVAILABLE, "source": source, "note": reason or "Unavailable"}


def is_available(v) -> bool:
    return isinstance(v, dict) and v.get("status") != UNAVAILABLE and v.get("value") is not None


def num(v, default=None):
    """The number inside a value dict (or the number itself), else `default`."""
    if isinstance(v, dict):
        v = v.get("value")
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else default


def experimental_note(value, note: str = "", source: str = "") -> dict:
    """A value from a method that is plausible but not validated (e.g. counting stream chunks as tokens)."""
    return val(value if value is None else round(value, 2) if isinstance(value, float) else value, EXPERIMENTAL, source, note)
