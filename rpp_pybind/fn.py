"""
Compatibility wrapper for the rocAL-style API.

Historically some code imported functions via:

    from rpp_pybind.fn import ...

The canonical implementation now lives under:

    rpp_pybind.amd.rpp.fn
"""

from __future__ import annotations

from .amd.rpp.fn import *  # noqa: F401,F403

# Re-export the same public API
from .amd.rpp.fn import __all__  # type: ignore  # noqa: E402
