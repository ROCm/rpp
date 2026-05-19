#pragma once
#include "../common.hpp"

namespace ops {

// morphological erosion (min filter), kernel_size must be odd
py::array_t<uint8_t> erode(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    uint32_t kernel_size = 3,
    const std::string& backend = "hip");

// morphological dilation (max filter), kernel_size must be odd
py::array_t<uint8_t> dilate(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    uint32_t kernel_size = 3,
    const std::string& backend = "hip");

} // namespace ops
