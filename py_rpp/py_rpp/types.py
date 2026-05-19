# Copyright (c) 2026 Advanced Micro Devices, Inc.
# RPP backend and layout constants — mirrors the C++ enums exposed in _py_rpp.types.

# Backend strings accepted by all py_rpp functions
HIP  = "hip"   # GPU via HIP/ROCm
CPU  = "cpu"   # Host CPU

BACKEND_HIP = HIP
BACKEND_CPU = CPU

# Layout strings used by fn.py for auto-detection and conversion
NHWC = "nhwc"  # Batch × Height × Width × Channel  (packed, default for uint8 images)
NCHW = "nchw"  # Batch × Channel × Height × Width  (planar, common in PyTorch)

__all__ = [
    "HIP", "CPU", "BACKEND_HIP", "BACKEND_CPU",
    "NHWC", "NCHW",
]
