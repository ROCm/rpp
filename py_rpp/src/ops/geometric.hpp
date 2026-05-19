#pragma once
#include "../common.hpp"

namespace ops {

// flip: horizontal=1 flips left-right, vertical=1 flips top-bottom (per image)
py::array_t<uint8_t> flip(
    py::array_t<uint8_t,  py::array::c_style | py::array::forcecast> src,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> horizontal,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> vertical,
    const std::string& backend = "hip");

// crop: extract a uniform-size region from each image.
// x, y: per-image top-left corner (shape (N,) uint32)
// out_h, out_w: uniform output size across the batch
py::array_t<uint8_t> crop(
    py::array_t<uint8_t,  py::array::c_style | py::array::forcecast> src,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> x,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> y,
    uint32_t out_h, uint32_t out_w,
    const std::string& backend = "hip");

// rotate: angle in degrees per image, output same size as input, bilinear interp
py::array_t<uint8_t> rotate(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> angle,
    const std::string& backend = "hip");

// resize: all images resized to (out_h, out_w), bilinear interpolation
py::array_t<uint8_t> resize(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    uint32_t out_h, uint32_t out_w,
    const std::string& backend = "hip");

} // namespace ops
