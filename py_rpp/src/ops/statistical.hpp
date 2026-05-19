#pragma once
#include "../common.hpp"

namespace ops {

// Per-image, per-channel mean.
// Returns float32 array of shape (N, C+1):
//   [:, 0..C-1] = per-channel mean,  [:, C] = overall mean across channels.
py::array_t<float> tensor_mean(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    const std::string& backend = "hip");

// Per-image, per-channel min pixel value.
// Returns uint8 array of shape (N, C+1): per-channel min + overall min.
py::array_t<uint8_t> tensor_min(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    const std::string& backend = "hip");

// Per-image, per-channel max pixel value.
// Returns uint8 array of shape (N, C+1): per-channel max + overall max.
py::array_t<uint8_t> tensor_max(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    const std::string& backend = "hip");

// Binary threshold: output[n,h,w,c] = 255 if min[n] <= src <= max[n] else 0
py::array_t<uint8_t> threshold(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> min_val,
    py::array_t<float,   py::array::c_style | py::array::forcecast> max_val,
    const std::string& backend = "hip");

} // namespace ops
