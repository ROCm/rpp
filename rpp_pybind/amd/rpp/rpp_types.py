# MIT License

# Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
