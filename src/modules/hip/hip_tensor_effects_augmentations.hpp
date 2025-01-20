/*
MIT License

Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef HIP_TENSOR_EFFECTS_AUGMENTATIONS_HPP
#define HIP_TENSOR_EFFECTS_AUGMENTATIONS_HPP

#include "kernel/gridmask.hpp"
#include "kernel/spatter.hpp"
#include "kernel/noise_salt_and_pepper.hpp"
#include "kernel/noise_shot.hpp"
#include "kernel/noise_gaussian.hpp"
#include "kernel/non_linear_blend.hpp"
#include "kernel/jitter.hpp"
#include "kernel/glitch.hpp"
#include "kernel/water.hpp"
#include "kernel/ricap.hpp"
#include "kernel/vignette.hpp"
#include "kernel/resize.hpp"  //pixelate dependency
#include "kernel/erase.hpp"
#include "kernel/fog.hpp"
#include "kernel/rain.hpp"

#endif // HIP_TENSOR_EFFECTS_AUGMENTATIONS_HPP
