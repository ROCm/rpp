#pragma once
#include "../common.hpp"

namespace ops {

// Additive gaussian noise: mean and std_dev per image, seed is global
py::array_t<uint8_t> gaussian_noise(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> mean,
    py::array_t<float,   py::array::c_style | py::array::forcecast> std_dev,
    uint32_t seed = 0,
    const std::string& backend = "hip");

// Salt-and-pepper noise: four per-image float tensors + seed
py::array_t<uint8_t> salt_and_pepper_noise(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> noise_prob,
    py::array_t<float,   py::array::c_style | py::array::forcecast> salt_prob,
    py::array_t<float,   py::array::c_style | py::array::forcecast> salt_value,
    py::array_t<float,   py::array::c_style | py::array::forcecast> pepper_value,
    uint32_t seed = 0,
    const std::string& backend = "hip");

// Vignette: darkens edges; intensity per image (0 = no effect, 1 = full dark)
py::array_t<uint8_t> vignette(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> intensity,
    const std::string& backend = "hip");

// Pixelate: replaces blocks with block average; needs an internal scratch buffer
py::array_t<uint8_t> pixelate(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    float pixelation_percentage = 50.0f,
    const std::string& backend = "hip");

} // namespace ops
