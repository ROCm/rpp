#pragma once
#include "../common.hpp"

namespace ops {

py::array_t<uint8_t> bitwise_and(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src1,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src2,
    const std::string& backend = "hip");

py::array_t<uint8_t> bitwise_or(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src1,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src2,
    const std::string& backend = "hip");

py::array_t<uint8_t> bitwise_xor(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src1,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src2,
    const std::string& backend = "hip");

py::array_t<uint8_t> bitwise_not(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    const std::string& backend = "hip");

} // namespace ops
