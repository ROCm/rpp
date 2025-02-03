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

/* Non Silent Region Detection requires Moving Mean Square (MMS) computation on input audio data
MMS buffer is a 1D buffer having same length as input audio. The algorithm used for MMS computation is explained with a sample use case

Example:
Input: [1, 2, 3, 4, 5, 6, 7, 8]
audio_length = 8
window_length = 3
reset_interval_length = 4

window_begin = -window_length + 1 = -2
window_factor = 1 / window_length = 1/3

MMS computation is divided into blocks of reset interval length
num_blocks = audio_length / reset_interval_length
For the above example we will have
    - 2 blocks (8 / 4)
    - each block runs for 4 iterations
    - in each iteration window begin value is increment by 1

Block1
window begin = -2
Iteration 0:    sum_of_squares = 1*1                              // window begin = -2
                store sum_of_squares * window_factor in MMS[0]

Iteration 1:    sum_of_squares = 1*1 + 2*2                        // window begin = -1
                store sum_of_squares * window_factor in MMS[1]

Iteration 2:    sum_of_squares = 1*1 + 2*2 + 3*3                  // window begin =  0
                store sum_of_squares * window_factor in MMS[2]
                sum_of_squares -= 1*1

Iteration 3:    sum_of_squares = 2*2 + 3*3 + 4*4                  // window begin =  1
                store sum_of_squares * window_factor in MMS[3]
                sum_of_squares -= 2*2

Block2
Iteration 0:    sum_of_squares = 3*3 + 4*4 + 5*5                  // window begin = 2
                store sum_of_squares * window_factor in MMS[4]
                sum_of_squares -= 3*3

Iteration 1:    sum_of_squares = 4*4 + 5*5 + 6*6                 // window begin = 3
                store sum_of_squares * window_factor in MMS[5]
                sum_of_squares -= 4*4

Iteration 2:    sum_of_squares = 5*5 + 6*6 + 7*7                 // window begin = 4
                store sum_of_squares * window_factor in MMS[6]
                sum_of_squares -= 5*5

Iteration 3:    sum_of_squares  = 6*6 + 7*7 + 8*8                // window begin = 5
                store sum_of_squares * window_factor in MMS[7]
                sum_of_squares -= 6*6

For computing beginning index and length of Non Silent Region in audio data we traverse over
the entire MMS buffer and compare these values with the calculated cutoff value
    - For beginning index, traverse over MMS buffer from 0 to audio_length - 1 and compare if any value
      is greater than or equal to cutoff value. if yes, that is the beginning index
    - For length, traverse over MMS buffer from audio_length - 1 to beginning index and compare if any value
      is greater than or equal to cutoff value. if yes, that is the ending index of Non Silent Region. From this
      data compute length with the formulae, length = ending index - beginning index + 1
*/

#include "rppdefs.h"
#include "rpp_cpu_common.hpp"

RppStatus non_silent_region_detection_host_tensor(Rpp32f *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  Rpp32s *srcLengthTensor,
                                                  Rpp32s *detectedIndexTensor,
                                                  Rpp32s *detectionLengthTensor,
                                                  Rpp32f cutOffDB,
                                                  Rpp32s windowLength,
                                                  Rpp32f referencePower,
                                                  Rpp32s resetInterval,
                                                  rpp::Handle& handle);