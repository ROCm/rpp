"""
Compatibility wrapper for the rocAL-style API.

Historically some code imported functions via:

    from rpp_pybind.rpp_types import ...

The canonical implementation now lives under:

    rpp_pybind.amd.rpp.rpp_types
"""

from __future__ import annotations

from .amd.rpp.rpp_types import *  # noqa: F401,F403

# Re-export the same public API
from .amd.rpp.rpp_types import __all__  # type: ignore  # noqa: E402
