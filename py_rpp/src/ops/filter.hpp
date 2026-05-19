#pragma once
#include "../common.hpp"

namespace ops {

// uniform box (mean) filter, kernel_size must be odd
py::array_t<uint8_t> box_filter(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    uint32_t kernel_size = 3,
    const std::string& backend = "hip");

// gaussian filter with per-image std_dev, kernel_size must be odd
py::array_t<uint8_t> gaussian_filter(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> std_dev,
    uint32_t kernel_size = 3,
    const std::string& backend = "hip");

// median filter (edge-preserving), kernel_size must be odd
py::array_t<uint8_t> median_filter(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    uint32_t kernel_size = 3,
    const std::string& backend = "hip");

} // namespace ops
