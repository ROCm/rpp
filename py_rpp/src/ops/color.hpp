#pragma once
#include "../common.hpp"

namespace ops {

// clamp(alpha*src + beta, 0, 255)
py::array_t<uint8_t> brightness(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> alpha,
    py::array_t<float,   py::array::c_style | py::array::forcecast> beta,
    const std::string& backend = "hip");

// src ^ gamma  (per-image)
py::array_t<uint8_t> gamma_correction(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> gamma,
    const std::string& backend = "hip");

// hue rotation in degrees [-180, 180] per image
py::array_t<uint8_t> hue(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> hue_shift,
    const std::string& backend = "hip");

// saturation factor per image (1.0 = identity)
py::array_t<uint8_t> saturation(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> factor,
    const std::string& backend = "hip");

// clamp(factor*(src - center) + center, 0, 255)
py::array_t<uint8_t> contrast(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> factor,
    py::array_t<float,   py::array::c_style | py::array::forcecast> center,
    const std::string& backend = "hip");

// exposure shift per image
py::array_t<uint8_t> exposure(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> factor,
    const std::string& backend = "hip");

// dst = alpha*src1 + (1-alpha)*src2
py::array_t<uint8_t> blend(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src1,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src2,
    py::array_t<float,   py::array::c_style | py::array::forcecast> alpha,
    const std::string& backend = "hip");

// combined brightness + contrast + hue + saturation adjustment
py::array_t<uint8_t> color_twist(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> brightness,
    py::array_t<float,   py::array::c_style | py::array::forcecast> contrast,
    py::array_t<float,   py::array::c_style | py::array::forcecast> hue_shift,
    py::array_t<float,   py::array::c_style | py::array::forcecast> saturation,
    const std::string& backend = "hip");

// per-channel histogram equalization (no params)
py::array_t<uint8_t> histogram_equalize(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    const std::string& backend = "hip");

} // namespace ops
