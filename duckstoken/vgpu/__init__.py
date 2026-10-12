"""VirtualGPU: application-level coordination of VRAM, system RAM, memory-mapped storage and an optional disk cache.

This package never pretends system RAM is VRAM. It measures what the machine and the inference backend really expose,
plans placement within those limits, benchmarks configurations and reports every number with an honesty label
(see `values.py`): measured, estimated, experimental or unavailable.
"""

from .values import ESTIMATED, EXPERIMENTAL, MEASURED, UNAVAILABLE, est, measured, unavailable, val  # noqa: F401
