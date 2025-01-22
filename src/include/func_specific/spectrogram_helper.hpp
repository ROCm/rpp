#ifndef RPP_SPECTROGRAM_HELPER_H
#define RPP_SPECTROGRAM_HELPER_H

#include <rppdefs.h>

inline bool is_pow2(Rpp64s n) { return (n & (n-1)) == 0; }
inline bool can_use_real_impl(Rpp64s n) { return is_pow2(n); }
inline Rpp64s size_in_buf(Rpp64s n) { return can_use_real_impl(n) ? n : 2 * n; }
inline Rpp64s size_out_buf(Rpp64s n) { return can_use_real_impl(n) ? n + 2 : 2 * n; }

// Compute hanning window
inline void hann_window(Rpp32f *output, Rpp32s windowSize)
{
    Rpp64f a = (2.0 * M_PI) / windowSize;
    for (Rpp32s t = 0; t < windowSize; t++)
    {
        Rpp64f phase = a * (t + 0.5);
        output[t] = (0.5 * (1.0 - std::cos(phase)));
    }
}

// Compute number of spectrogram windows
inline Rpp32s get_num_windows(Rpp32s length, Rpp32s windowLength, Rpp32s windowStep, bool centerWindows)
{
    if (!centerWindows)
        length -= windowLength;
    return ((length / windowStep) + 1);
}

// Compute reflect start idx to pad
inline RPP_HOST_DEVICE Rpp32s get_idx_reflect(Rpp32s loc, Rpp32s minLoc, Rpp32s maxLoc)
{
    if (maxLoc - minLoc < 2)
        return maxLoc - 1;
    for (;;)
    {
        if (loc < minLoc)
            loc = 2 * minLoc - loc;
        else if (loc >= maxLoc)
            loc = 2 * maxLoc - 2 - loc;
        else
            break;
    }
    return loc;
}

#endif