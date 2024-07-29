#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__device__ __forceinline__ void compute_mel(float *srcPtr, int melBin, float *weightsDown, int *intervals, int fftStride, int fftShift, float normFactor, float &dstVal)
{
    dstVal = 0;
    //start and end FFT bin indices for the current mel bin
    int fftbin = intervals[melBin];
    int fftBinEnd = intervals[melBin + 1];

    float *srcPtrTemp = srcPtr + fftbin * fftStride + fftShift;
    // Process the first interval of FFT bins, applying the weights up
    for (; fftbin < fftBinEnd; fftbin++, srcPtrTemp += fftStride) 
    {
        auto weightUp = float(1) - weightsDown[fftbin];
        weightUp *= normFactor;
        dstVal += *srcPtrTemp * weightUp;
    }

    fftBinEnd = intervals[melBin + 2];    // Update the end FFT bin index for the next interval
    srcPtrTemp = srcPtr + fftbin * fftStride + fftShift;

    // Process the second interval of FFT bins, applying the weights down
    for (; fftbin < fftBinEnd; fftbin++, srcPtrTemp += fftStride) 
    {
        auto weightDown = weightsDown[fftbin];
        weightDown *= normFactor;
        dstVal += *srcPtrTemp * weightDown;
    }
}

__global__ void mel_filter_bank_tensor(float *srcPtr,
                                       uint2 srcStridesNH,
                                       float *dstPtr,
                                       uint2 dstStridesNH,
                                       int *srcDimsTensor,
                                       int numFilter,
                                       float sampleRate,
                                       bool normalize,
                                       float *normFactors,
                                       float *weightsDown,
                                       int *intervals)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= srcDimsTensor[id_z * 2 + 1] || id_y >= numFilter)
        return;

    uint dstIdx = id_z * dstStridesNH.x + id_y * dstStridesNH.y + id_x;
    uint srcIdx = id_z * srcStridesNH.x;

    float normFactor = (normalize) ? normFactors[id_y] : 1;
    compute_mel(srcPtr + srcIdx, id_y, weightsDown, intervals, srcStridesNH.y, id_x, normFactor, dstPtr[dstIdx]);
}

RppStatus hip_exec_mel_filter_bank_tensor(Rpp32f *srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          Rpp32f *dstPtr,
                                          RpptDescPtr dstDescPtr,
                                          Rpp32s* srcDimsTensor,
                                          Rpp32f maxFreqVal,
                                          Rpp32f minFreqVal,
                                          RpptMelScaleFormula melFormula,
                                          Rpp32s numFilter,
                                          Rpp32f sampleRate,
                                          bool normalize,
                                          rpp::Handle& handle)
{
    BaseMelScale *melScalePtr;
    switch (melFormula)
    {
        case RpptMelScaleFormula::HTK:
            melScalePtr = new HtkMelScale;
            break;
        case RpptMelScaleFormula::SLANEY:
        default:
            melScalePtr = new SlaneyMelScale();
            break;
    }

    Rpp32f maxFreq = sampleRate / 2;
    Rpp32f minFreq = minFreqVal;

    Rpp64f melLow = melScalePtr->hz_to_mel(minFreq);
    Rpp64f melHigh = melScalePtr->hz_to_mel(maxFreq);
    Rpp64f melStep = (melHigh - melLow) / (numFilter + 1);

    Rpp32f *scratchMem = handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem;
    Rpp32f *normFactors = scratchMem;
    Rpp32f *weightsDown = scratchMem + numFilter;
    Rpp32s *intervals = reinterpret_cast<Rpp32s *>(weightsDown + srcDescPtr->h);

    Rpp32s nfft = (srcDescPtr->h - 1) * 2;
    Rpp32s numBins = nfft / 2 + 1;
    Rpp64f hzStep = static_cast<Rpp64f>(sampleRate) / nfft;
    Rpp64f invHzStep = 1.0 / hzStep;

    Rpp32s fftBinStart = std::ceil(minFreq * invHzStep);
    Rpp32s fftBinEnd = std::ceil(maxFreq * invHzStep);
    fftBinEnd = std::min(fftBinEnd, numBins);

    std::fill(normFactors, normFactors + numFilter, 1.0f);
    memset(weightsDown, 0, sizeof(srcDescPtr->h * sizeof(Rpp32f)));
    std::fill(intervals, intervals + numFilter + 2, -1);

    Rpp32s fftBin = fftBinStart;
    Rpp64f mel0 = melLow, mel1 = melLow + melStep;
    Rpp64f fIter = fftBin * hzStep;

    intervals[0] = fftBinStart;
    intervals[numFilter + 1] = fftBinEnd;

    for (int interval = 1, index = 0; index < numFilter + 1; interval++, index++, mel0 = mel1, mel1 += melStep)
    {
        Rpp64f f0 = melScalePtr->mel_to_hz(mel0);
        Rpp64f f1 = melScalePtr->mel_to_hz(index == numFilter ? melHigh : mel1);
        Rpp64f slope = 1.0 / (f1 - f0);
        intervals[interval] = std::ceil(f1 / hzStep);

        if (normalize && index < numFilter)
        {
            Rpp64f f2 = melScalePtr->mel_to_hz(mel1 + melStep);
            normFactors[index] = 2.0 / (f2 - f0);
        }

        for (; fftBin < fftBinEnd && fIter < f1; fftBin++, fIter = fftBin * hzStep) {
            weightsDown[fftBin] = (f1 - fIter) * slope;
        }
    }

    Rpp32s globalThreads_x = dstDescPtr->w;
    Rpp32s globalThreads_y = dstDescPtr->h;
    Rpp32s globalThreads_z = dstDescPtr->n;
    hipLaunchKernelGGL(mel_filter_bank_tensor,
                       dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), LOCAL_THREADS_Z),
                       dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, globalThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                       dstPtr,
                       make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                       srcDimsTensor,
                       numFilter,
                       sampleRate,
                       normalize,
                       normFactors,
                       weightsDown,
                       intervals);

    delete melScalePtr;
    return RPP_SUCCESS;
}
