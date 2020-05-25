/*To be used by the library alone*/
#ifndef RPP_CL_COMMON_H
#define RPP_CL_COMMON_H

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <rppdefs.h>

#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

enum class RPPTensorDataType
{
    U8 = 0,
    FP32,
    FP16,
};

inline RppStatus generate_gaussian_kernel_gpu(Rpp32f stdDev, Rpp32f* kernel, Rpp32u kernelSize)
{
    Rpp32f s, sum = 0.0, multiplier;
    int bound = ((kernelSize - 1) / 2);
    Rpp32u c = 0;
    s = 1 / (2 * stdDev * stdDev);
    multiplier = (1 / M_PI) * (s);
    for (int i = -bound; i <= bound; i++)
    {
        for (int j = -bound; j <= bound; j++)
        {
            kernel[c] = multiplier * exp((-1) * (s) * (i*i + j*j));
            sum += kernel[c];
            c += 1;
        }
    }
    for (int i = 0; i < (kernelSize * kernelSize); i++)
    {
        kernel[i] /= sum;
    }

    return RPP_SUCCESS;
}
#endif //RPP_CL_COMMON_H
