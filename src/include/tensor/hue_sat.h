/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

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

#include "rpp_cpu_common.hpp"

inline void RGB_to_HSV(RpptFloatRGB *pixel, Rpp32f &hue, Rpp32f &sat, Rpp32f &val)
{
    Rpp32f rf = pixel->R, gf = pixel->G, bf = pixel->B;
    Rpp32f cmax = RPPMAX3(rf, gf, bf);
    Rpp32f cmin = RPPMIN3(rf, gf, bf);
    Rpp32f delta = cmax - cmin;

    hue = 0.0f;
    sat = 0.0f;
    val = cmax;

    if ((delta != 0) && (cmax != 0)) {
        sat = delta / cmax;
        if (cmax == rf)
        {
            hue = (gf - bf) / delta;
        }
        else if (cmax == gf) {
            sat = delta / cmax;
            hue = (bf - rf) / delta + 2.0f;
        } else {
            sat = delta / cmax;
            hue = (rf - gf) / delta + 4.0f;
        }
    }
}

inline void HSV_to_RGB(Rpp32f hue, Rpp32f sat, Rpp32f val, RpptFloatRGB *pixel)
{
    Rpp32s hueIntegerPart = (Rpp32s)hue;
    Rpp32f f = hue - hueIntegerPart;
    Rpp32f p = val * (1.0f - sat);
    Rpp32f q = val * (1.0f - sat * f);
    Rpp32f t = val * (1.0f - sat * (1.0f - f));

    switch (hueIntegerPart) {
        case 0: pixel->R = val; pixel->G = t; pixel->B = p; break;
        case 1: pixel->R = q; pixel->G = val; pixel->B = p; break;
        case 2: pixel->R = p; pixel->G = val; pixel->B = t; break;
        case 3: pixel->R = p; pixel->G = q; pixel->B = val; break;
        case 4: pixel->R = t; pixel->G = p; pixel->B = val; break;
        case 5: pixel->R = val; pixel->G = p; pixel->B = q; break;
    }
}


__device__ void RGB_to_HSV_hip(float *pixelR, float *pixelG, float *pixelB, float &hue, float &sat, float &val)
{
    float cmax = fmaxf(fmaxf(*pixelR, *pixelG), *pixelB);
    float cmin = fminf(fminf(*pixelR, *pixelG), *pixelB);
    float delta = cmax - cmin;

    hue = 0.0f;
    sat = 0.0f;
    val = cmax;

    if ((delta != 0) && (cmax != 0))
    {
        sat = delta / cmax;   
        if (cmax == *pixelR)
            hue = (*pixelG - *pixelB) / delta;
        else if (cmax == *pixelG)
            hue = 2.0f + (*pixelB - *pixelR) / delta;
        else
            hue = 4.0f + (*pixelR - *pixelG) / delta;
    }
}

__device__ void HSV_to_RGB_hip(float hue, float sat, float val, float *pixelR, float *pixelG, float *pixelB)
{
    float p = val * (1.0f - sat);
    float q = val * (1.0f - (sat * (hue - floor(hue))));
    float t = val * (1.0f - (sat * (1.0f - (hue - floor(hue)))));

    switch ((int)hue)
    {
        case 0: *pixelR = val; *pixelG = t;   *pixelB = p;   break;
        case 1: *pixelR = q;   *pixelG = val; *pixelB = p;   break;
        case 2: *pixelR = p;   *pixelG = val; *pixelB = t;   break;
        case 3: *pixelR = p;   *pixelG = q;   *pixelB = val; break;
        case 4: *pixelR = t;   *pixelG = p;   *pixelB = val; break;
        case 5: *pixelR = val; *pixelG = p;   *pixelB = q;   break;
    }
}

