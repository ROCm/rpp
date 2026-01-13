# Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.
# MIT License
# Mukesh/rpp/rpp_pybind/amd/rpp/rpp_types.py

"""
RPP Type Definitions and Helpers
=================================

Re-exports C++ types and provides Python convenience functions.
"""

# Simple import - like rocAL
import rpp_pybind

# Re-export enums for easier access
RppBackend = rpp_pybind.types.RppBackend
RppStatus = rpp_pybind.types.RppStatus
RpptDataType = rpp_pybind.types.RpptDataType
RpptLayout = rpp_pybind.types.RpptLayout

# Shortcuts for common values
HOST = RppBackend.HOST
HIP = RppBackend.HIP

NCHW = RpptLayout.NCHW
NHWC = RpptLayout.NHWC

U8 = RpptDataType.U8
F32 = RpptDataType.F32
F16 = RpptDataType.F16

SUCCESS = RppStatus.SUCCESS
ERROR = RppStatus.ERROR

# Helper functions
def is_gpu_available():
    """Check if GPU is available for HIP backend."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def get_default_backend():
    """Get default backend based on GPU availability."""
    return HIP if is_gpu_available() else HOST

__all__ = [
    'RppBackend', 'RppStatus', 'RpptDataType', 'RpptLayout',
    'HOST', 'HIP',
    'NCHW', 'NHWC',
    'U8', 'F32', 'F16',
    'SUCCESS', 'ERROR',
    'is_gpu_available', 'get_default_backend'
]
