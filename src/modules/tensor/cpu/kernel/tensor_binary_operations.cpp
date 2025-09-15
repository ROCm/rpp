#include "host_tensor_executors.hpp"
#include "broadcast.hpp"
#include "rpp_cpu_simd_math.hpp"

template<typename T>
inline void add_op(T *dst, T *src1, T *src2) { *dst = *src1 + *src2; }

template<typename T>
inline void subtract_op(T *dst, T *src1, T *src2) { *dst = *src1 - *src2; }

template<typename T>
inline void multiply_op(T *dst, T *src1, T *src2) { *dst = *src1 * *src2; }

template<typename T1, typename T2>
inline void divide_op(T1 *dst, T2 *src1, T2 *src2)
{
    if constexpr (std::is_same_v<T1, T2>)
        *dst = *src1 / *src2;
    else if constexpr (std::is_same_v<T1, Rpp32f>)
        *dst = static_cast<Rpp32f>(*src1) / static_cast<Rpp32f>(*src2);
}

inline void simd_add_ps(__m256 &a, __m256 &b) { a = _mm256_add_ps(a, b); }
inline void simd_subtract_ps(__m256 &a, __m256 &b) { a = _mm256_sub_ps(a, b); }
inline void simd_multiply_ps(__m256 &a, __m256 &b) { a = _mm256_mul_ps(a, b); }
inline void simd_divide_ps(__m256 &a, __m256 &b) { a = _mm256_div_ps(a, b); }

template<typename T>
inline void simd_add_si256(__m256i &a, __m256i &b)
{
    if constexpr (std::is_same<T, Rpp8u>::value || std::is_same<T, Rpp8s>::value)
        a = _mm256_add_epi8(a, b);
    else if constexpr (std::is_same<T, Rpp16s>::value || std::is_same<T, Rpp16u>::value)
        a = _mm256_add_epi16(a, b);
    else if constexpr (std::is_same<T, Rpp32s>::value || std::is_same<T, Rpp32u>::value)
        a = _mm256_add_epi32(a, b);
}

template<typename T>
inline void simd_subtract_si256(__m256i &a, __m256i &b)
{
    if constexpr (std::is_same<T, Rpp8u>::value || std::is_same<T, Rpp8s>::value)
        a = _mm256_sub_epi8(a, b);
    else if constexpr (std::is_same<T, Rpp16s>::value || std::is_same<T, Rpp16u>::value)
        a = _mm256_sub_epi16(a, b);
    else if constexpr (std::is_same<T, Rpp32s>::value || std::is_same<T, Rpp32u>::value)
        a = _mm256_sub_epi32(a, b);
}

template<typename T>
inline void simd_multiply_si256(__m256i &a, __m256i &b)
{
    if constexpr (std::is_same<T, Rpp16u>::value || std::is_same<T, Rpp16s>::value)
        a = _mm256_mullo_epi16(a, b);
    else if constexpr (std::is_same<T, Rpp32s>::value || std::is_same<T, Rpp32u>::value)
        a = _mm256_mullo_epi32(a, b);
}

template<typename T>
inline void simd_divide_si256(__m256 *out, __m256i &a, __m256i &b)
{
    if constexpr (std::is_same<T, Rpp8u>::value)
    {
        printf("SIMD Divide Called 1\n");
        __m128i a_half1 = _mm256_castsi256_si128(a);
        __m128i a_half2 = _mm256_extracti128_si256(a, 1);
        __m128i b_half1 = _mm256_castsi256_si128(b);
        __m128i b_half2 = _mm256_extracti128_si256(b, 1);
        out[0] = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(a_half1)), _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b_half1)));
        out[1] = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(a_half1, 8))), _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(b_half1, 8))));
        out[2] = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(a_half2)), _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b_half2)));
        out[3] = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(a_half2, 8))), _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(b_half2, 8))));
    }
    else if constexpr (std::is_same<T, Rpp8s>::value)
    {
        printf("SIMD Divide Called 2\n");
        __m128i a_half1 = _mm256_castsi256_si128(a);
        __m128i a_half2 = _mm256_extracti128_si256(a, 1);
        __m128i b_half1 = _mm256_castsi256_si128(b);
        __m128i b_half2 = _mm256_extracti128_si256(b, 1);
        out[0] = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(a_half1)), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b_half1)));
        out[1] = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(a_half1, 8))), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b_half1, 8))));
        out[2] = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(a_half2)), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b_half2)));
        out[3] = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(a_half2, 8))), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b_half2, 8))));
    }
    else if constexpr (std::is_same<T, Rpp16u>::value)
    {
        printf("SIMD Divide Called 3\n");
        __m128i a_half1 = _mm256_castsi256_si128(a);
        __m128i a_half2 = _mm256_extracti128_si256(a, 1);
        __m128i b_half1 = _mm256_castsi256_si128(b);
        __m128i b_half2 = _mm256_extracti128_si256(b, 1);
        out[0] = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(a_half1)), _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(b_half1)));
        out[1] = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(a_half2)), _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(b_half2)));
    }
    else if constexpr (std::is_same<T, Rpp16s>::value)
    {
        printf("SIMD Divide Called 4\n");
        __m128i a_half1 = _mm256_castsi256_si128(a);
        __m128i a_half2 = _mm256_extracti128_si256(a, 1);
        __m128i b_half1 = _mm256_castsi256_si128(b);
        __m128i b_half2 = _mm256_extracti128_si256(b, 1);
        out[0] = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(a_half1)), _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(b_half1)));
        out[1] = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(a_half2)), _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(b_half2)));
    }
    else if constexpr (std::is_same<T, Rpp32s>::value)
    {
        printf("SIMD Divide Called 4\n");
        out[0] = _mm256_div_ps(_mm256_cvtepi32_ps(a), _mm256_cvtepi32_ps(b));
    }
}

template<typename T>
inline void store_ps_function(__m256 *out, Rpp32f *dst)
{
    if constexpr (std::is_same<T, Rpp8u>::value || std::is_same<T, Rpp8s>::value)
    {
        printf("Store for 8 bit\n");
        _mm256_storeu_ps(dst, out[0]);
        _mm256_storeu_ps(dst + 8, out[1]);
        _mm256_storeu_ps(dst + 16, out[2]);
        _mm256_storeu_ps(dst + 24, out[3]);
    }
    else if constexpr (std::is_same<T, Rpp16u>::value || std::is_same<T, Rpp16s>::value)
    {
        printf("Store for 16 bit\n");
        _mm256_storeu_ps(dst, out[0]);
        _mm256_storeu_ps(dst + 8, out[1]);
    }
    else if constexpr (std::is_same<T, Rpp32s>::value)
    {
        printf("Store for 32 bit\n");
        _mm256_storeu_ps(dst, out[0]);
    }
}

// Helper functions for broadcasting for different datatypes (8 bit, 16 bit and 32 bit)
inline __m256i simd_set1_val(Rpp8u &val) { return _mm256_set1_epi8(val); }
inline __m256i simd_set1_val(Rpp8s &val) { return _mm256_set1_epi8(val); }
inline __m256i simd_set1_val(Rpp16u &val) { return _mm256_set1_epi16(val); }
inline __m256i simd_set1_val(Rpp16s &val) { return _mm256_set1_epi16(val); }
inline __m256i simd_set1_val(Rpp32u &val) { return _mm256_set1_epi32(val); }
inline __m256i simd_set1_val(Rpp32s &val) { return _mm256_set1_epi32(val); }

template<typename T1, typename T2, typename Operation>
inline void tensor_binary_arithmetic_op_recursive(T1 *src1, T1 *src2, Rpp32u *src1Strides, Rpp32u *src2Strides, T2 *dst, Rpp32u *dstStrides, Rpp32u *dstShape, Rpp32u nDim, Operation op)
{
    if (!nDim)
        op(dst, src1, src2);
    else
    {
        for (int i = 0; i < *dstShape; i++)
        {
            tensor_binary_arithmetic_op_recursive(src1, src2, src1Strides + 1, src2Strides + 1, dst, dstStrides + 1, dstShape + 1, nDim - 1, op);
            dst += *(dstStrides + 1);
            src1 += *(src1Strides + 1);
            src2 += *(src2Strides + 1);
        }
    }
}

template<typename Operation, typename SIMDOperation>
RppStatus tensor_binary_op_f32_f32_host_tensor(Rpp32f *srcPtr1,
                                               Rpp32f *srcPtr2,
                                               RpptGenericDescPtr srcPtr1GenericDescPtr,
                                               RpptGenericDescPtr srcPtr2GenericDescPtr,
                                               Rpp32f *dstPtr,
                                               RpptGenericDescPtr dstGenericDescPtr,
                                               Operation op,
                                               SIMDOperation simd_op,
                                               RpptBroadcastMode broadcastMode,
                                               Rpp32u *srcPtr1roiTensor,
                                               Rpp32u *srcPtr2roiTensor,
                                               rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u src1NDim = srcPtr1GenericDescPtr->numDims - 1;  // Omitting batchSize here to get tensor dimension
    Rpp32u src2NDim = srcPtr2GenericDescPtr->numDims - 1;  // Omitting batchSize here to get tensor dimension
    Rpp32u dstDim = src1NDim > src2NDim ? src1NDim : src2NDim; // Destination dimension set to maximum of the input dimensions
    Rpp32u minDim = src1NDim < src2NDim ? src1NDim : src2NDim; // Minimum of input dimensions

    // Overall dimension compatibility check for the entire batch
    for(int test = 0; test < minDim; test++)
        if(srcPtr1GenericDescPtr->dims[src1NDim - test] != srcPtr2GenericDescPtr->dims[src2NDim - test])
            if((srcPtr1GenericDescPtr->dims[src1NDim - test] != 1) && (srcPtr2GenericDescPtr->dims[src2NDim - test] != 1))
            {
                printf("Incompatible dimensions for the batch\n");
                return RPP_SUCCESS;
            }

    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        Rpp32u *src1roi = srcPtr1roiTensor + batchCount * src1NDim * 2;
        Rpp32u *src1Begin = src1roi;
        Rpp32u *src1Dims = src1Begin + src1NDim;

        Rpp32u *src2roi = srcPtr2roiTensor + batchCount * src2NDim * 2;
        Rpp32u *src2Begin = src2roi;
        Rpp32u *src2Dims = src2Begin + src2NDim;

        Rpp32u *src1Strides = srcPtr1GenericDescPtr->strides;
        Rpp32u *src2Strides = srcPtr2GenericDescPtr->strides;
        Rpp32u *dstStrides = dstGenericDescPtr->strides;

        Rpp32u *length, *src1length, *src2length, *src1BcastStrides, *src2BcastStrides, *dstBcastStrides;

        if(broadcastMode == RPP_BROADCAST_ENABLE)
        {
            // Dimensions and Strides based on individual sample ROIs, used for broadcasting purposes
            Rpp32u src1BroadcastDims[RPPT_MAX_DIMS_SAMPLE], src2BroadcastDims[RPPT_MAX_DIMS_SAMPLE], dstBroadcastDims[RPPT_MAX_DIMS_SAMPLE];
            Rpp32u src1BroadcastStrides[RPPT_MAX_DIMS_SAMPLE], src2BroadcastStrides[RPPT_MAX_DIMS_SAMPLE], dstBroadcastStrides[RPPT_MAX_DIMS_SAMPLE];

            bool incompatibleDims = false;

            // Copy ROI limits and Strides to individual sample strides and dims until minDim
            for(int i = 0; i < minDim; i++)
            {
                Rpp32u curIndex = RPPT_MAX_DIMS_SAMPLE - i - 1;
                src1BroadcastDims[curIndex] = src1Dims[src1NDim - i - 1];
                src2BroadcastDims[curIndex] = src2Dims[src2NDim - i - 1];
                src1BroadcastStrides[curIndex] = src1Strides[src1NDim - i];
                src2BroadcastStrides[curIndex] = src2Strides[src2NDim - i];
                dstBroadcastStrides[curIndex] = dstStrides[dstDim - i];
                // Check compatibility of dimension i.e check for equal shape or one of the input dims to be 1
                if((src1BroadcastDims[curIndex] != src2BroadcastDims[curIndex]) && (src1BroadcastDims[curIndex] != 1) && (src2BroadcastDims[curIndex] != 1))
                    incompatibleDims = true;
                dstBroadcastDims[curIndex] = std::max(src1BroadcastDims[curIndex], src2BroadcastDims[curIndex]);
            }

            // Dimension compatibility failure case
            if(incompatibleDims == true)
                printf("Incompatible dimensions for operation for sample %d inside batch\n", batchCount);

            // Handle cases of mismatching num dims
            if(src1NDim < src2NDim)
                for(int i = minDim; i < dstDim; i++)
                {
                    Rpp32u curIndex = RPPT_MAX_DIMS_SAMPLE - i - 1;
                    src1BroadcastDims[curIndex] = 1;
                    src2BroadcastDims[curIndex] = src2Dims[src2NDim - i];
                    dstBroadcastDims[curIndex] = src2Dims[src2NDim - i];
                    src1BroadcastStrides[curIndex] = 0;
                    src2BroadcastStrides[curIndex] = src2Strides[src2NDim - i];
                    dstBroadcastStrides[curIndex] = dstStrides[dstDim - i];
                }
            else if(src1NDim > src2NDim)
                for(int i = minDim; i < dstDim; i++)
                {
                    Rpp32u curIndex = RPPT_MAX_DIMS_SAMPLE - i - 1;
                    src2BroadcastDims[curIndex] = 1;
                    src1BroadcastDims[curIndex] = src1Dims[src1NDim - i];
                    dstBroadcastDims[curIndex] = src1Dims[src1NDim - i];
                    src1BroadcastStrides[curIndex] = src1Strides[src1NDim - i];
                    src2BroadcastStrides[curIndex] = 0;
                    dstBroadcastStrides[curIndex] = dstStrides[dstDim - i];
                }

            // Source strides for sample set to zero if corresponding axis shape = 1 for broadcasting purposes
            // Setting stride to zero will allow for repetition of values operated required for broadcasting
            for(int i = 0; i < minDim; i++)
            {
                if((src1BroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i] != dstBroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i]) && (src1BroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i] == 1))
                {
                    src1BroadcastStrides[RPPT_MAX_DIMS_SAMPLE - 1 - i] = 0;
                }
                if((src2BroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i] != dstBroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i]) && (src2BroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i] == 1))
                {
                    src2BroadcastStrides[RPPT_MAX_DIMS_SAMPLE - 1 - i] = 0;
                }
            }

            Rpp32u testOffset = RPPT_MAX_DIMS_SAMPLE - dstDim;

            // Shift dims and strides by offset to process only valid values
            length = dstBroadcastDims + testOffset;
            src1length = src1BroadcastDims + testOffset;
            src2length = src2BroadcastDims + testOffset;

            src1BcastStrides = src1BroadcastStrides + testOffset;
            src2BcastStrides = src2BroadcastStrides + testOffset;
            dstBcastStrides =  dstBroadcastStrides + testOffset;
        }
        else
        {
            // Both length for dest and srclength set to src1Dims because in non broadcast case, source and destination dimensions are expected to be the same
            length = src1Dims;
            src1length = src1Dims;
            src2length = src2Dims;

            src1BcastStrides = srcPtr1GenericDescPtr->strides + 1;
            src2BcastStrides = srcPtr2GenericDescPtr->strides + 1;
            dstBcastStrides =  dstGenericDescPtr->strides + 1;
        }

        Rpp32f *srcPtrTemp1 = srcPtr1 + batchCount * srcPtr1GenericDescPtr->strides[0];
        Rpp32f *srcPtrTemp2 = srcPtr2 + batchCount * srcPtr2GenericDescPtr->strides[0];

        for(int i = 0; i < src1NDim; i++)
            srcPtrTemp1 += src1Begin[i] * srcPtr1GenericDescPtr->strides[i + 1];

        for(int i = 0; i < src2NDim; i++)
            srcPtrTemp2 += src2Begin[i] * srcPtr2GenericDescPtr->strides[i + 1];

        Rpp32f *dstPtrTemp = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        Rpp32u vectorIncrement = 16;

        // For nDim = 1, 2, 3 cases handled when the lowest axis shape = 1 and != 1 for efficient processing
        if (dstDim == 1)
        {
            Rpp32u alignedLength = length[0] & ~15;
            Rpp32u src1shape = src1length[0];
            Rpp32u src2shape = src2length[0];
            Rpp32u vectorLoopCount = 0;
            if (src1shape == 1)
            {
#if __AVX2__
                __m256 p1 = _mm256_set1_ps(srcPtrTemp1[0]);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p2[2];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrTemp2, p2);    // simd loads
                    simd_op(p2[0], p1);
                    simd_op(p2[1], p1);
                    rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, p2);    // simd stores
                    srcPtrTemp2 += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                 for (; vectorLoopCount < length[0]; vectorLoopCount++)
                 {
                     op(dstPtrTemp, srcPtrTemp1, srcPtrTemp2);
                     srcPtrTemp2++;
                     dstPtrTemp++;
                 }
            }
            else if (src2shape == 1)
            {
#if __AVX2__
                __m256 p2 = _mm256_set1_ps(srcPtrTemp2[0]);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p1[2];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrTemp1, p1);    // simd loads
                    simd_op(p1[0], p2);
                    simd_op(p1[1], p2);
                    rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, p1);    // simd stores
                    srcPtrTemp1 += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                 for (; vectorLoopCount < length[0]; vectorLoopCount++)
                 {
                     op(dstPtrTemp, srcPtrTemp1, srcPtrTemp2);
                     srcPtrTemp1++;
                     dstPtrTemp++;
                 }
            }
            else
            {
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p1[2], p2[2];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrTemp1, p1);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrTemp2, p2);    // simd loads
                    simd_op(p1[0], p2[0]);
                    simd_op(p1[1], p2[1]);
                    rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, p1);    // simd stores
                    srcPtrTemp1 += vectorIncrement;
                    srcPtrTemp2 += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < length[0]; vectorLoopCount++)
                {
                    op(dstPtrTemp, srcPtrTemp1, srcPtrTemp2);
                    srcPtrTemp1++;
                    srcPtrTemp2++;
                    dstPtrTemp++;
                }
            }
        }
        else if (dstDim == 2)
        {
            Rpp32u alignedLength = length[1] & ~15;
            Rpp32u src1shape = src1length[1];
            Rpp32u src2shape = src2length[1];
            if(src1shape == 1)
            {
                for (int i = 0; i < length[0]; i++)
                {
                    Rpp32f *srcPtrElem1 = srcPtrTemp1;
                    Rpp32f *srcPtrElem2 = srcPtrTemp2;
                    Rpp32f *dstPtrElem = dstPtrTemp;

                    int vectorLoopCount = 0;
#if __AVX2__
                    __m256 p1 = _mm256_set1_ps(srcPtrElem1[0]);
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p2[2];
                        rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrElem2, p2);    // simd loads
                        simd_op(p2[0], p1);
                        simd_op(p2[1], p1);
                        rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrElem, p2);    // simd stores
                        srcPtrElem2 += vectorIncrement;
                        dstPtrElem += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[1]; vectorLoopCount++)
                    {
                        op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                        srcPtrElem2++;
                        dstPtrElem++;
                    }
                    srcPtrTemp1 += src1BcastStrides[0];
                    srcPtrTemp2 += src2BcastStrides[0];
                    dstPtrTemp += dstBcastStrides[0];
                }
            }
            else if (src2shape == 1)
            {
                for (int i = 0; i < length[0]; i++)
                {
                    Rpp32f *srcPtrElem1 = srcPtrTemp1;
                    Rpp32f *srcPtrElem2 = srcPtrTemp2;
                    Rpp32f *dstPtrElem = dstPtrTemp;

                    int vectorLoopCount = 0;
#if __AVX2__
                    __m256 p2 = _mm256_set1_ps(srcPtrElem2[0]);
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p1[2];
                        rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrElem1, p1);    // simd loads
                        simd_op(p1[0], p2);
                        simd_op(p1[1], p2);
                        rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrElem, p1);    // simd stores
                        srcPtrElem1 += vectorIncrement;
                        dstPtrElem += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[1]; vectorLoopCount++)
                    {
                        op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                        srcPtrElem1++;
                        dstPtrElem++;
                    }
                    srcPtrTemp1 += src1BcastStrides[0];
                    srcPtrTemp2 += src2BcastStrides[0];
                    dstPtrTemp += dstBcastStrides[0];
                }
            }
            else
            {
                for (int i = 0; i < length[0]; i++)
                {
                    Rpp32f *srcPtrElem1 = srcPtrTemp1;
                    Rpp32f *srcPtrElem2 = srcPtrTemp2;
                    Rpp32f *dstPtrElem = dstPtrTemp;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p1[2], p2[2];
                        rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrElem1, p1);    // simd loads
                        rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrElem2, p2);    // simd loads
                        simd_op(p1[0], p2[0]);
                        simd_op(p1[1], p2[1]);
                        rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrElem, p1);    // simd stores
                        srcPtrElem1 += vectorIncrement;
                        srcPtrElem2 += vectorIncrement;
                        dstPtrElem += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[1]; vectorLoopCount++)
                    {
                        op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                        srcPtrElem1++;
                        srcPtrElem2++;
                        dstPtrElem++;
                    }
                    srcPtrTemp1 += src1BcastStrides[0];
                    srcPtrTemp2 += src2BcastStrides[0];
                    dstPtrTemp += dstBcastStrides[0];
                }
            }
        }
        else if (dstDim == 3)
        {
            Rpp32u alignedLength = length[2] & ~15;
            Rpp32u src1shape = src1length[2];
            Rpp32u src2shape = src2length[2];
            if(src1shape == 1)
            {
                for (int i = 0; i < length[0]; i++)
                {
                    Rpp32f *srcPtrOuter1 = srcPtrTemp1;
                    Rpp32f *srcPtrOuter2 = srcPtrTemp2;
                    Rpp32f *dstPtrOuter = dstPtrTemp;

                    for (int j = 0; j < length[1]; j++)
                    {
                        Rpp32f *srcPtrElem1 = srcPtrOuter1;
                        Rpp32f *srcPtrElem2 = srcPtrOuter2;
                        Rpp32f *dstPtrElem = dstPtrOuter;

                        int vectorLoopCount = 0;
#if __AVX2__
                        __m256 p1 = _mm256_set1_ps(srcPtrElem1[0]);
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                        {
                            __m256 p2[2];
                            rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrElem2, p2);    // simd loads
                            simd_op(p2[0], p1);
                            simd_op(p2[0], p1);
                            rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrElem, p2);    // simd stores
                            srcPtrElem2 += vectorIncrement;
                            dstPtrElem += vectorIncrement;
                        }
#endif
                        for (; vectorLoopCount < length[2]; vectorLoopCount++)
                        {
                            op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                            srcPtrElem2++;
                            dstPtrElem++;
                        }

                        srcPtrOuter1 += src1BcastStrides[1];
                        srcPtrOuter2 += src2BcastStrides[1];
                        dstPtrOuter += dstBcastStrides[1];
                    }

                    srcPtrTemp1 += src1BcastStrides[0];
                    srcPtrTemp2 += src2BcastStrides[0];
                    dstPtrTemp += dstBcastStrides[0];
                }
            }
            else if (src2shape == 1)
            {
                for (int i = 0; i < length[0]; i++)
                {
                    Rpp32f *srcPtrOuter1 = srcPtrTemp1;
                    Rpp32f *srcPtrOuter2 = srcPtrTemp2;
                    Rpp32f *dstPtrOuter = dstPtrTemp;

                    for (int j = 0; j < length[1]; j++)
                    {
                        Rpp32f *srcPtrElem1 = srcPtrOuter1;
                        Rpp32f *srcPtrElem2 = srcPtrOuter2;
                        Rpp32f *dstPtrElem = dstPtrOuter;

                        int vectorLoopCount = 0;
#if __AVX2__
                        __m256 p2 =  _mm256_set1_ps(srcPtrElem2[0]);
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                        {
                            __m256 p1[2];
                            rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrElem1, p1);    // simd loads
                            simd_op(p1[0], p2);
                            simd_op(p1[1], p2);
                            rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrElem, p1);    // simd stores
                            srcPtrElem1 += vectorIncrement;
                            dstPtrElem += vectorIncrement;
                        }
#endif
                        for (; vectorLoopCount < length[2]; vectorLoopCount++)
                        {
                            op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                            srcPtrElem1++;
                            dstPtrElem++;
                        }

                        srcPtrOuter1 += src1BcastStrides[1];
                        srcPtrOuter2 += src2BcastStrides[1];
                        dstPtrOuter += dstBcastStrides[1];
                    }

                    srcPtrTemp1 += src1BcastStrides[0];
                    srcPtrTemp2 += src2BcastStrides[0];
                    dstPtrTemp += dstBcastStrides[0];
                }
            }
            else
            {
                for (int i = 0; i < length[0]; i++)
                {
                    Rpp32f *srcPtrOuter1 = srcPtrTemp1;
                    Rpp32f *srcPtrOuter2 = srcPtrTemp2;
                    Rpp32f *dstPtrOuter = dstPtrTemp;

                    for (int j = 0; j < length[1]; j++)
                    {
                        Rpp32f *srcPtrElem1 = srcPtrOuter1;
                        Rpp32f *srcPtrElem2 = srcPtrOuter2;
                        Rpp32f *dstPtrElem = dstPtrOuter;

                        int vectorLoopCount = 0;
#if __AVX2__
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                        {
                            __m256 p1[2], p2[2];
                            rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrElem1, p1);    // simd loads
                            rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrElem2, p2);    // simd loads
                            simd_op(p1[0], p2[0]);
                            simd_op(p1[1], p2[1]);
                            rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrElem, p1);    // simd stores
                            srcPtrElem1 += vectorIncrement;
                            srcPtrElem2 += vectorIncrement;
                            dstPtrElem += vectorIncrement;
                        }
#endif
                        for (; vectorLoopCount < length[2]; vectorLoopCount++)
                        {
                            op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                            srcPtrElem1++;
                            srcPtrElem2++;
                            dstPtrElem++;
                        }

                        srcPtrOuter1 += src1BcastStrides[1];
                        srcPtrOuter2 += src2BcastStrides[1];
                        dstPtrOuter += dstBcastStrides[1];
                    }

                    srcPtrTemp1 += src1BcastStrides[0];
                    srcPtrTemp2 += src2BcastStrides[0];
                    dstPtrTemp += dstBcastStrides[0];
                }
            }
        }
        else
            tensor_binary_arithmetic_op_recursive(srcPtrTemp1, srcPtrTemp2, src1BcastStrides, src2BcastStrides, dstPtrTemp, dstBcastStrides, length, dstDim, op);
    }

    return RPP_SUCCESS;
}

RppStatus tensor_binary_op_dispatch_f32_f32_host_tensor(Rpp32f *srcPtr1,
                                                        Rpp32f *srcPtr2,
                                                        RpptGenericDescPtr srcPtr1GenericDescPtr,
                                                        RpptGenericDescPtr srcPtr2GenericDescPtr,
                                                        Rpp32f *dstPtr,
                                                        RpptGenericDescPtr dstGenericDescPtr,
                                                        RpptOp tensorOp,
                                                        RpptBroadcastMode broadcastMode,
                                                        Rpp32u *srcPtr1roiTensor,
                                                        Rpp32u *srcPtr2roiTensor,
                                                        rpp::Handle& handle)
{
    switch(tensorOp) {
        case RPP_TENSOR_OP_ADD:
            tensor_binary_op_f32_f32_host_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, add_op<Rpp32f>, simd_add_ps, broadcastMode, srcPtr1roiTensor, srcPtr2roiTensor, handle);
            break;
        case RPP_TENSOR_OP_SUBTRACT:
            tensor_binary_op_f32_f32_host_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, subtract_op<Rpp32f>, simd_subtract_ps, broadcastMode, srcPtr1roiTensor, srcPtr2roiTensor, handle);
            break;
        case RPP_TENSOR_OP_MULTIPLY:
            tensor_binary_op_f32_f32_host_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, multiply_op<Rpp32f>, simd_multiply_ps, broadcastMode, srcPtr1roiTensor, srcPtr2roiTensor, handle);
            break;
        case RPP_TENSOR_OP_DIVIDE:
            tensor_binary_op_f32_f32_host_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, divide_op<Rpp32f, Rpp32f>, simd_divide_ps, broadcastMode, srcPtr1roiTensor, srcPtr2roiTensor, handle);
            break;
    }

    return RPP_SUCCESS;
}

template<typename Operation, typename SIMDOperation>
RppStatus tensor_binary_op_f16_f16_host_tensor(Rpp16f *srcPtr1,
                                               Rpp16f *srcPtr2,
                                               RpptGenericDescPtr srcPtr1GenericDescPtr,
                                               RpptGenericDescPtr srcPtr2GenericDescPtr,
                                               Rpp16f *dstPtr,
                                               RpptGenericDescPtr dstGenericDescPtr,
                                               Operation op,
                                               SIMDOperation simd_op,
                                               RpptBroadcastMode broadcastMode,
                                               Rpp32u *srcPtr1roiTensor,
                                               Rpp32u *srcPtr2roiTensor,
                                               rpp::Handle& handle)
{

    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u src1NDim = srcPtr1GenericDescPtr->numDims - 1;  // Omitting batchSize here to get tensor dimension
    Rpp32u src2NDim = srcPtr2GenericDescPtr->numDims - 1;  // Omitting batchSize here to get tensor dimension
    Rpp32u dstDim = src1NDim > src2NDim ? src1NDim : src2NDim; // Destination dimension set to maximum of the input dimensions
    Rpp32u minDim = src1NDim < src2NDim ? src1NDim : src2NDim; // Minimum of input dimensions

    // Overall dimension compatibility check for the entire batch
    for(int test = 0; test < minDim; test++)
        if(srcPtr1GenericDescPtr->dims[src1NDim - test] != srcPtr2GenericDescPtr->dims[src2NDim - test])
            if((srcPtr1GenericDescPtr->dims[src1NDim - test] != 1) && (srcPtr2GenericDescPtr->dims[src2NDim - test] != 1))
            {
                printf("Incompatible dimensions for the batch\n");
                return RPP_SUCCESS;
            }

    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        Rpp32u *src1roi = srcPtr1roiTensor + batchCount * src1NDim * 2;
        Rpp32u *src1Begin = src1roi;
        Rpp32u *src1Dims = src1Begin + src1NDim;

        Rpp32u *src2roi = srcPtr2roiTensor + batchCount * src2NDim * 2;
        Rpp32u *src2Begin = src2roi;
        Rpp32u *src2Dims = src2Begin + src2NDim;

        Rpp32u *src1Strides = srcPtr1GenericDescPtr->strides;
        Rpp32u *src2Strides = srcPtr2GenericDescPtr->strides;
        Rpp32u *dstStrides = dstGenericDescPtr->strides;

        Rpp32u *length, *src1length, *src2length, *src1BcastStrides, *src2BcastStrides, *dstBcastStrides;

        if(broadcastMode == RPP_BROADCAST_ENABLE)
        {
            // Dimensions and Strides based on individual sample ROIs, used for broadcasting purposes
            Rpp32u src1BroadcastDims[RPPT_MAX_DIMS_SAMPLE], src2BroadcastDims[RPPT_MAX_DIMS_SAMPLE], dstBroadcastDims[RPPT_MAX_DIMS_SAMPLE];
            Rpp32u src1BroadcastStrides[RPPT_MAX_DIMS_SAMPLE], src2BroadcastStrides[RPPT_MAX_DIMS_SAMPLE], dstBroadcastStrides[RPPT_MAX_DIMS_SAMPLE];

            bool incompatibleDims = false;

            // Copy ROI limits and Strides to individual sample strides and dims until minDim
            for(int i = 0; i < minDim; i++)
            {
                Rpp32u curIndex = RPPT_MAX_DIMS_SAMPLE - i - 1;
                src1BroadcastDims[curIndex] = src1Dims[src1NDim - i - 1];
                src2BroadcastDims[curIndex] = src2Dims[src2NDim - i - 1];
                src1BroadcastStrides[curIndex] = src1Strides[src1NDim - i];
                src2BroadcastStrides[curIndex] = src2Strides[src2NDim - i];
                dstBroadcastStrides[curIndex] = dstStrides[dstDim - i];
                // Check compatibility of dimension i.e check for equal shape or one of the input dims to be 1
                if((src1BroadcastDims[curIndex] != src2BroadcastDims[curIndex]) && (src1BroadcastDims[curIndex] != 1) && (src2BroadcastDims[curIndex] != 1))
                    incompatibleDims = true;
                dstBroadcastDims[curIndex] = std::max(src1BroadcastDims[curIndex], src2BroadcastDims[curIndex]);
            }

            // Dimension compatibility failure case
            if(incompatibleDims == true)
                printf("Incompatible dimensions for operation for sample %d inside batch\n", batchCount);

            // Handle cases of mismatching num dims
            if(src1NDim < src2NDim)
                for(int i = minDim; i < dstDim; i++)
                {
                    Rpp32u curIndex = RPPT_MAX_DIMS_SAMPLE - i - 1;
                    src1BroadcastDims[curIndex] = 1;
                    src2BroadcastDims[curIndex] = src2Dims[src2NDim - i];
                    dstBroadcastDims[curIndex] = src2Dims[src2NDim - i];
                    src1BroadcastStrides[curIndex] = 0;
                    src2BroadcastStrides[curIndex] = src2Strides[src2NDim - i];
                    dstBroadcastStrides[curIndex] = dstStrides[dstDim - i];
                }
            else if(src1NDim > src2NDim)
                for(int i = minDim; i < dstDim; i++)
                {
                    Rpp32u curIndex = RPPT_MAX_DIMS_SAMPLE - i - 1;
                    src2BroadcastDims[curIndex] = 1;
                    src1BroadcastDims[curIndex] = src1Dims[src1NDim - i];
                    dstBroadcastDims[curIndex] = src1Dims[src1NDim - i];
                    src1BroadcastStrides[curIndex] = src1Strides[src1NDim - i];
                    src2BroadcastStrides[curIndex] = 0;
                    dstBroadcastStrides[curIndex] = dstStrides[dstDim - i];
                }

            // Source strides for sample set to zero if corresponding axis shape = 1 for broadcasting purposes
            // Setting stride to zero will allow for repetition of values operated required for broadcasting
            for(int i = 0; i < minDim; i++)
            {
                if((src1BroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i] != dstBroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i]) && (src1BroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i] == 1))
                {
                    src1BroadcastStrides[RPPT_MAX_DIMS_SAMPLE - 1 - i] = 0;
                }
                if((src2BroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i] != dstBroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i]) && (src2BroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i] == 1))
                {
                    src2BroadcastStrides[RPPT_MAX_DIMS_SAMPLE - 1 - i] = 0;
                }
            }

            Rpp32u testOffset = RPPT_MAX_DIMS_SAMPLE - dstDim;

            // Shift dims and strides by offset to process only valid values
            length = dstBroadcastDims + testOffset;
            src1length = src1BroadcastDims + testOffset;
            src2length = src2BroadcastDims + testOffset;

            src1BcastStrides = src1BroadcastStrides + testOffset;
            src2BcastStrides = src2BroadcastStrides + testOffset;
            dstBcastStrides =  dstBroadcastStrides + testOffset;
        }
        else
        {
            // Both length for dest and srclength set to src1Dims because in non broadcast case, source and destination dimensions are expected to be the same
            length = src1Dims;
            src1length = src1Dims;
            src2length = src2Dims;

            src1BcastStrides = srcPtr1GenericDescPtr->strides + 1;
            src2BcastStrides = srcPtr2GenericDescPtr->strides + 1;
            dstBcastStrides =  dstGenericDescPtr->strides + 1;
        }

        Rpp16f *srcPtrTemp1 = srcPtr1 + batchCount * srcPtr1GenericDescPtr->strides[0];
        Rpp16f *srcPtrTemp2 = srcPtr2 + batchCount * srcPtr2GenericDescPtr->strides[0];

        for(int i = 0; i < src1NDim; i++)
            srcPtrTemp1 += src1Begin[i] * srcPtr1GenericDescPtr->strides[i + 1];

        for(int i = 0; i < src2NDim; i++)
            srcPtrTemp2 += src2Begin[i] * srcPtr2GenericDescPtr->strides[i + 1];

        Rpp16f *dstPtrTemp = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        Rpp32u vectorIncrement = 16;

        // For nDim = 1, 2, 3 cases handled when the lowest axis shape = 1 and != 1 for efficient processing
        if (dstDim == 1)
        {
            Rpp32u alignedLength = length[0] & ~15;
            Rpp32u src1shape = src1length[0];
            Rpp32u src2shape = src2length[0];
            Rpp32u vectorLoopCount = 0;
            if (src1shape == 1)
            {
#if __AVX2__
                __m256 p1 = _mm256_cvtph_ps(_mm_set1_epi16(reinterpret_cast<Rpp16s*>(srcPtrTemp1)[0]));
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p2[2];
                    rpp_simd_load(rpp_load16_f16_to_f32_avx, srcPtrTemp2, p2);    // simd loads
                    simd_op(p2[0], p1);
                    simd_op(p2[1], p1);
                    rpp_simd_store(rpp_store16_f32_to_f16_avx, dstPtrTemp, p2);    // simd stores
                    srcPtrTemp2 += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                 for (; vectorLoopCount < length[0]; vectorLoopCount++)
                 {
                     op(dstPtrTemp, srcPtrTemp1, srcPtrTemp2);
                     srcPtrTemp2++;
                     dstPtrTemp++;
                 }
            }
            else if (src2shape == 1)
            {
#if __AVX2__
                __m256 p2 = _mm256_cvtph_ps(_mm_set1_epi16(reinterpret_cast<Rpp16s*>(srcPtrTemp2)[0]));
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p1[2];
                    rpp_simd_load(rpp_load16_f16_to_f32_avx, srcPtrTemp1, p1);    // simd loads
                    simd_op(p1[0], p2);
                    simd_op(p1[1], p2);
                    rpp_simd_store(rpp_store16_f32_to_f16_avx, dstPtrTemp, p1);    // simd stores
                    srcPtrTemp1 += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                 for (; vectorLoopCount < length[0]; vectorLoopCount++)
                 {
                     op(dstPtrTemp, srcPtrTemp1, srcPtrTemp2);
                     srcPtrTemp1++;
                     dstPtrTemp++;
                 }
            }
            else
            {
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p1[2], p2[2];
                    rpp_simd_load(rpp_load16_f16_to_f32_avx, srcPtrTemp1, p1);    // simd loads
                    rpp_simd_load(rpp_load16_f16_to_f32_avx, srcPtrTemp2, p2);    // simd loads
                    simd_op(p1[0], p2[0]);
                    simd_op(p1[1], p2[1]);
                    rpp_simd_store(rpp_store16_f32_to_f16_avx, dstPtrTemp, p1);    // simd stores
                    srcPtrTemp1 += vectorIncrement;
                    srcPtrTemp2 += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < length[0]; vectorLoopCount++)
                {
                    op(dstPtrTemp, srcPtrTemp1, srcPtrTemp2);
                    srcPtrTemp1++;
                    srcPtrTemp2++;
                    dstPtrTemp++;
                }
            }
        }
        else if (dstDim == 2)
        {
            Rpp32u alignedLength = length[1] & ~15;
            Rpp32u src1shape = src1length[1];
            Rpp32u src2shape = src2length[1];
            if(src1shape == 1)
            {
                for (int i = 0; i < length[0]; i++)
                {
                    Rpp16f *srcPtrElem1 = srcPtrTemp1;
                    Rpp16f *srcPtrElem2 = srcPtrTemp2;
                    Rpp16f *dstPtrElem = dstPtrTemp;

                    int vectorLoopCount = 0;
#if __AVX2__
                    __m256 p1 = _mm256_cvtph_ps(_mm_set1_epi16(reinterpret_cast<Rpp16s*>(srcPtrElem1)[0]));
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p2[2];
                        rpp_simd_load(rpp_load16_f16_to_f32_avx, srcPtrElem2, p2);    // simd loads
                        simd_op(p2[0], p1);
                        simd_op(p2[1], p1);
                        rpp_simd_store(rpp_store16_f32_to_f16_avx, dstPtrElem, p2);    // simd stores
                        srcPtrElem2 += vectorIncrement;
                        dstPtrElem += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[1]; vectorLoopCount++)
                    {
                        op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                        srcPtrElem2++;
                        dstPtrElem++;
                    }
                    srcPtrTemp1 += src1BcastStrides[0];
                    srcPtrTemp2 += src2BcastStrides[0];
                    dstPtrTemp += dstBcastStrides[0];
                }
            }
            else if (src2shape == 1)
            {
                for (int i = 0; i < length[0]; i++)
                {
                    Rpp16f *srcPtrElem1 = srcPtrTemp1;
                    Rpp16f *srcPtrElem2 = srcPtrTemp2;
                    Rpp16f *dstPtrElem = dstPtrTemp;

                    int vectorLoopCount = 0;
#if __AVX2__
                    __m256 p2 = _mm256_cvtph_ps(_mm_set1_epi16(reinterpret_cast<Rpp16s*>(srcPtrElem2)[0]));
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p1[2];
                        rpp_simd_load(rpp_load16_f16_to_f32_avx, srcPtrElem1, p1);    // simd loads
                        simd_op(p1[0], p2);
                        simd_op(p1[1], p2);
                        rpp_simd_store(rpp_store16_f32_to_f16_avx, dstPtrElem, p1);    // simd stores
                        srcPtrElem1 += vectorIncrement;
                        dstPtrElem += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[1]; vectorLoopCount++)
                    {
                        op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                        srcPtrElem1++;
                        dstPtrElem++;
                    }
                    srcPtrTemp1 += src1BcastStrides[0];
                    srcPtrTemp2 += src2BcastStrides[0];
                    dstPtrTemp += dstBcastStrides[0];
                }
            }
            else
            {
                for (int i = 0; i < length[0]; i++)
                {
                    Rpp16f *srcPtrElem1 = srcPtrTemp1;
                    Rpp16f *srcPtrElem2 = srcPtrTemp2;
                    Rpp16f *dstPtrElem = dstPtrTemp;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p1[2], p2[2];
                        rpp_simd_load(rpp_load16_f16_to_f32_avx, srcPtrElem1, p1);    // simd loads
                        rpp_simd_load(rpp_load16_f16_to_f32_avx, srcPtrElem2, p2);    // simd loads
                        simd_op(p1[0], p2[0]);
                        simd_op(p1[1], p2[1]);
                        rpp_simd_store(rpp_store16_f32_to_f16_avx, dstPtrElem, p1);    // simd stores
                        srcPtrElem1 += vectorIncrement;
                        srcPtrElem2 += vectorIncrement;
                        dstPtrElem += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[1]; vectorLoopCount++)
                    {
                        op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                        srcPtrElem1++;
                        srcPtrElem2++;
                        dstPtrElem++;
                    }
                    srcPtrTemp1 += src1BcastStrides[0];
                    srcPtrTemp2 += src2BcastStrides[0];
                    dstPtrTemp += dstBcastStrides[0];
                }
            }
        }
        else if (dstDim == 3)
        {
            Rpp32u alignedLength = length[2] & ~15;
            Rpp32u src1shape = src1length[2];
            Rpp32u src2shape = src2length[2];
            if(src1shape == 1)
            {
                for (int i = 0; i < length[0]; i++)
                {
                    Rpp16f *srcPtrOuter1 = srcPtrTemp1;
                    Rpp16f *srcPtrOuter2 = srcPtrTemp2;
                    Rpp16f *dstPtrOuter = dstPtrTemp;

                    for (int j = 0; j < length[1]; j++)
                    {
                        Rpp16f *srcPtrElem1 = srcPtrOuter1;
                        Rpp16f *srcPtrElem2 = srcPtrOuter2;
                        Rpp16f *dstPtrElem = dstPtrOuter;

                        int vectorLoopCount = 0;
#if __AVX2__
                        __m256 p1 = _mm256_cvtph_ps(_mm_set1_epi16(reinterpret_cast<Rpp16s*>(srcPtrElem1)[0]));
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                        {
                            __m256 p2[2];
                            rpp_simd_load(rpp_load16_f16_to_f32_avx, srcPtrElem2, p2);    // simd loads
                            simd_op(p2[0], p1);
                            simd_op(p2[0], p1);
                            rpp_simd_store(rpp_store16_f32_to_f16_avx, dstPtrElem, p2);    // simd stores
                            srcPtrElem2 += vectorIncrement;
                            dstPtrElem += vectorIncrement;
                        }
#endif
                        for (; vectorLoopCount < length[2]; vectorLoopCount++)
                        {
                            op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                            srcPtrElem2++;
                            dstPtrElem++;
                        }

                        srcPtrOuter1 += src1BcastStrides[1];
                        srcPtrOuter2 += src2BcastStrides[1];
                        dstPtrOuter += dstBcastStrides[1];
                    }

                    srcPtrTemp1 += src1BcastStrides[0];
                    srcPtrTemp2 += src2BcastStrides[0];
                    dstPtrTemp += dstBcastStrides[0];
                }
            }
            else if (src2shape == 1)
            {
                for (int i = 0; i < length[0]; i++)
                {
                    Rpp16f *srcPtrOuter1 = srcPtrTemp1;
                    Rpp16f *srcPtrOuter2 = srcPtrTemp2;
                    Rpp16f *dstPtrOuter = dstPtrTemp;

                    for (int j = 0; j < length[1]; j++)
                    {
                        Rpp16f *srcPtrElem1 = srcPtrOuter1;
                        Rpp16f *srcPtrElem2 = srcPtrOuter2;
                        Rpp16f *dstPtrElem = dstPtrOuter;

                        int vectorLoopCount = 0;
#if __AVX2__
                        __m256 p2 = _mm256_cvtph_ps(_mm_set1_epi16(reinterpret_cast<Rpp16s*>(srcPtrElem2)[0]));
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                        {
                            __m256 p1[2];
                            rpp_simd_load(rpp_load16_f16_to_f32_avx, srcPtrElem1, p1);    // simd loads
                            simd_op(p1[0], p2);
                            simd_op(p1[1], p2);
                            rpp_simd_store(rpp_store16_f32_to_f16_avx, dstPtrElem, p1);    // simd stores
                            srcPtrElem1 += vectorIncrement;
                            dstPtrElem += vectorIncrement;
                        }
#endif
                        for (; vectorLoopCount < length[2]; vectorLoopCount++)
                        {
                            op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                            srcPtrElem1++;
                            dstPtrElem++;
                        }

                        srcPtrOuter1 += src1BcastStrides[1];
                        srcPtrOuter2 += src2BcastStrides[1];
                        dstPtrOuter += dstBcastStrides[1];
                    }

                    srcPtrTemp1 += src1BcastStrides[0];
                    srcPtrTemp2 += src2BcastStrides[0];
                    dstPtrTemp += dstBcastStrides[0];
                }
            }
            else
            {
                for (int i = 0; i < length[0]; i++)
                {
                    Rpp16f *srcPtrOuter1 = srcPtrTemp1;
                    Rpp16f *srcPtrOuter2 = srcPtrTemp2;
                    Rpp16f *dstPtrOuter = dstPtrTemp;

                    for (int j = 0; j < length[1]; j++)
                    {
                        Rpp16f *srcPtrElem1 = srcPtrOuter1;
                        Rpp16f *srcPtrElem2 = srcPtrOuter2;
                        Rpp16f *dstPtrElem = dstPtrOuter;

                        int vectorLoopCount = 0;
#if __AVX2__
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                        {
                            __m256 p1[2], p2[2];
                            rpp_simd_load(rpp_load16_f16_to_f32_avx, srcPtrElem1, p1);    // simd loads
                            rpp_simd_load(rpp_load16_f16_to_f32_avx, srcPtrElem2, p2);    // simd loads
                            simd_op(p1[0], p2[0]);
                            simd_op(p1[1], p2[1]);
                            rpp_simd_store(rpp_store16_f32_to_f16_avx, dstPtrElem, p1);    // simd stores
                            srcPtrElem1 += vectorIncrement;
                            srcPtrElem2 += vectorIncrement;
                            dstPtrElem += vectorIncrement;
                        }
#endif
                        for (; vectorLoopCount < length[2]; vectorLoopCount++)
                        {
                            op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                            srcPtrElem1++;
                            srcPtrElem2++;
                            dstPtrElem++;
                        }

                        srcPtrOuter1 += src1BcastStrides[1];
                        srcPtrOuter2 += src2BcastStrides[1];
                        dstPtrOuter += dstBcastStrides[1];
                    }

                    srcPtrTemp1 += src1BcastStrides[0];
                    srcPtrTemp2 += src2BcastStrides[0];
                    dstPtrTemp += dstBcastStrides[0];
                }
            }
        }
        else
            tensor_binary_arithmetic_op_recursive(srcPtrTemp1, srcPtrTemp2, src1BcastStrides, src2BcastStrides, dstPtrTemp, dstBcastStrides, length, dstDim, op);
    }

    return RPP_SUCCESS;
}

RppStatus tensor_binary_op_dispatch_f16_f16_host_tensor(Rpp16f *srcPtr1,
                                                        Rpp16f *srcPtr2,
                                                        RpptGenericDescPtr srcPtr1GenericDescPtr,
                                                        RpptGenericDescPtr srcPtr2GenericDescPtr,
                                                        Rpp16f *dstPtr,
                                                        RpptGenericDescPtr dstGenericDescPtr,
                                                        RpptOp tensorOp,
                                                        RpptBroadcastMode broadcastMode,
                                                        Rpp32u *srcPtr1roiTensor,
                                                        Rpp32u *srcPtr2roiTensor,
                                                        rpp::Handle& handle)
{
    switch(tensorOp) {
        case RPP_TENSOR_OP_ADD:
            tensor_binary_op_f16_f16_host_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, add_op<Rpp16f>, simd_add_ps, broadcastMode, srcPtr1roiTensor, srcPtr2roiTensor, handle);
            break;
        case RPP_TENSOR_OP_SUBTRACT:
            tensor_binary_op_f16_f16_host_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, subtract_op<Rpp16f>, simd_subtract_ps, broadcastMode, srcPtr1roiTensor, srcPtr2roiTensor, handle);
            break;
        case RPP_TENSOR_OP_MULTIPLY:
            tensor_binary_op_f16_f16_host_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, multiply_op<Rpp16f>, simd_multiply_ps, broadcastMode, srcPtr1roiTensor, srcPtr2roiTensor, handle);
            break;
        case RPP_TENSOR_OP_DIVIDE:
            tensor_binary_op_f16_f16_host_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, divide_op<Rpp16f, Rpp16f>, simd_divide_ps, broadcastMode, srcPtr1roiTensor, srcPtr2roiTensor, handle);
            break;
    }

    return RPP_SUCCESS;
}

template<typename T, typename Operation, typename SIMDOperation>
RppStatus tensor_binary_op_int_host_tensor(T *srcPtr1,
                                           T *srcPtr2,
                                           RpptGenericDescPtr srcPtr1GenericDescPtr,
                                           RpptGenericDescPtr srcPtr2GenericDescPtr,
                                           T *dstPtr,
                                           RpptGenericDescPtr dstGenericDescPtr,
                                           Operation op,
                                           SIMDOperation simd_op,
                                           RpptBroadcastMode broadcastMode,
                                           Rpp32u vectorIncrement,
                                           Rpp32u *srcPtr1roiTensor,
                                           Rpp32u *srcPtr2roiTensor,
                                           rpp::Handle& handle)
{

    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u src1NDim = srcPtr1GenericDescPtr->numDims - 1;  // Omitting batchSize here to get tensor dimension
    Rpp32u src2NDim = srcPtr2GenericDescPtr->numDims - 1;  // Omitting batchSize here to get tensor dimension
    Rpp32u dstDim = src1NDim > src2NDim ? src1NDim : src2NDim; // Destination dimension set to maximum of the input dimensions
    Rpp32u minDim = src1NDim < src2NDim ? src1NDim : src2NDim; // Minimum of input dimensions

    // Overall dimension compatibility check for the entire batch
    for(int test = 0; test < minDim; test++)
        if(srcPtr1GenericDescPtr->dims[src1NDim - test] != srcPtr2GenericDescPtr->dims[src2NDim - test])
            if((srcPtr1GenericDescPtr->dims[src1NDim - test] != 1) && (srcPtr2GenericDescPtr->dims[src2NDim - test] != 1))
            {
                printf("Incompatible dimensions for the batch\n");
                return RPP_SUCCESS;
            }

    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        Rpp32u *src1roi = srcPtr1roiTensor + batchCount * src1NDim * 2;
        Rpp32u *src1Begin = src1roi;
        Rpp32u *src1Dims = src1Begin + src1NDim;

        Rpp32u *src2roi = srcPtr2roiTensor + batchCount * src2NDim * 2;
        Rpp32u *src2Begin = src2roi;
        Rpp32u *src2Dims = src2Begin + src2NDim;

        Rpp32u *src1Strides = srcPtr1GenericDescPtr->strides;
        Rpp32u *src2Strides = srcPtr2GenericDescPtr->strides;
        Rpp32u *dstStrides = dstGenericDescPtr->strides;

        Rpp32u *length, *src1length, *src2length, *src1BcastStrides, *src2BcastStrides, *dstBcastStrides;

        if(broadcastMode == RPP_BROADCAST_ENABLE)
        {
            // Dimensions and Strides based on individual sample ROIs, used for broadcasting purposes
            Rpp32u src1BroadcastDims[RPPT_MAX_DIMS_SAMPLE], src2BroadcastDims[RPPT_MAX_DIMS_SAMPLE], dstBroadcastDims[RPPT_MAX_DIMS_SAMPLE];
            Rpp32u src1BroadcastStrides[RPPT_MAX_DIMS_SAMPLE], src2BroadcastStrides[RPPT_MAX_DIMS_SAMPLE], dstBroadcastStrides[RPPT_MAX_DIMS_SAMPLE];

            bool incompatibleDims = false;

            // Copy ROI limits and Strides to individual sample strides and dims until minDim
            for(int i = 0; i < minDim; i++)
            {
                Rpp32u curIndex = RPPT_MAX_DIMS_SAMPLE - i - 1;
                src1BroadcastDims[curIndex] = src1Dims[src1NDim - i - 1];
                src2BroadcastDims[curIndex] = src2Dims[src2NDim - i - 1];
                src1BroadcastStrides[curIndex] = src1Strides[src1NDim - i];
                src2BroadcastStrides[curIndex] = src2Strides[src2NDim - i];
                dstBroadcastStrides[curIndex] = dstStrides[dstDim - i];
                // Check compatibility of dimension i.e check for equal shape or one of the input dims to be 1
                if((src1BroadcastDims[curIndex] != src2BroadcastDims[curIndex]) && (src1BroadcastDims[curIndex] != 1) && (src2BroadcastDims[curIndex] != 1))
                    incompatibleDims = true;
                dstBroadcastDims[curIndex] = std::max(src1BroadcastDims[curIndex], src2BroadcastDims[curIndex]);
            }

            // Dimension compatibility failure case
            if(incompatibleDims == true)
                printf("Incompatible dimensions for operation for sample %d inside batch\n", batchCount);

            // Handle cases of mismatching num dims
            if(src1NDim < src2NDim)
                for(int i = minDim; i < dstDim; i++)
                {
                    Rpp32u curIndex = RPPT_MAX_DIMS_SAMPLE - i - 1;
                    src1BroadcastDims[curIndex] = 1;
                    src2BroadcastDims[curIndex] = src2Dims[src2NDim - i];
                    dstBroadcastDims[curIndex] = src2Dims[src2NDim - i];
                    src1BroadcastStrides[curIndex] = 0;
                    src2BroadcastStrides[curIndex] = src2Strides[src2NDim - i];
                    dstBroadcastStrides[curIndex] = dstStrides[dstDim - i];
                }
            else if(src1NDim > src2NDim)
                for(int i = minDim; i < dstDim; i++)
                {
                    Rpp32u curIndex = RPPT_MAX_DIMS_SAMPLE - i - 1;
                    src2BroadcastDims[curIndex] = 1;
                    src1BroadcastDims[curIndex] = src1Dims[src1NDim - i];
                    dstBroadcastDims[curIndex] = src1Dims[src1NDim - i];
                    src1BroadcastStrides[curIndex] = src1Strides[src1NDim - i];
                    src2BroadcastStrides[curIndex] = 0;
                    dstBroadcastStrides[curIndex] = dstStrides[dstDim - i];
                }

            // Source strides for sample set to zero if corresponding axis shape = 1 for broadcasting purposes
            // Setting stride to zero will allow for repetition of values operated required for broadcasting
            for(int i = 0; i < minDim; i++)
            {
                if((src1BroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i] != dstBroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i]) && (src1BroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i] == 1))
                {
                    src1BroadcastStrides[RPPT_MAX_DIMS_SAMPLE - 1 - i] = 0;
                }
                if((src2BroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i] != dstBroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i]) && (src2BroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i] == 1))
                {
                    src2BroadcastStrides[RPPT_MAX_DIMS_SAMPLE - 1 - i] = 0;
                }
            }

            Rpp32u testOffset = RPPT_MAX_DIMS_SAMPLE - dstDim;

            // Shift dims and strides by offset to process only valid values
            length = dstBroadcastDims + testOffset;
            src1length = src1BroadcastDims + testOffset;
            src2length = src2BroadcastDims + testOffset;

            src1BcastStrides = src1BroadcastStrides + testOffset;
            src2BcastStrides = src2BroadcastStrides + testOffset;
            dstBcastStrides =  dstBroadcastStrides + testOffset;
        }
        else
        {
            // Both length for dest and srclength set to src1Dims because in non broadcast case, source and destination dimensions are expected to be the same
            length = src1Dims;
            src1length = src1Dims;
            src2length = src2Dims;

            src1BcastStrides = srcPtr1GenericDescPtr->strides + 1;
            src2BcastStrides = srcPtr2GenericDescPtr->strides + 1;
            dstBcastStrides =  dstGenericDescPtr->strides + 1;
        }

        T *srcPtrTemp1 = srcPtr1 + batchCount * srcPtr1GenericDescPtr->strides[0];
        T *srcPtrTemp2 = srcPtr2 + batchCount * srcPtr2GenericDescPtr->strides[0];

        for(int i = 0; i < src1NDim; i++)
            srcPtrTemp1 += src1Begin[i] * srcPtr1GenericDescPtr->strides[i + 1];

        for(int i = 0; i < src2NDim; i++)
            srcPtrTemp2 += src2Begin[i] * srcPtr2GenericDescPtr->strides[i + 1];

        T *dstPtrTemp = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        Rpp32u alignMask = vectorIncrement - 1;

        // For nDim = 1, 2, 3 cases handled when the lowest axis shape = 1 and != 1 for efficient processing
        if (dstDim == 1)
        {
            Rpp32u alignedLength = length[0] & ~alignMask;
            Rpp32u src1shape = src1length[0];
            Rpp32u src2shape = src2length[0];
            Rpp32u vectorLoopCount = 0;
            if (src1shape == 1)
            {
#if __AVX2__
                __m256i p1 = simd_set1_val(srcPtrTemp1[0]);    // simd broadcast
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256i p2 = _mm256_loadu_si256((const __m256i *)srcPtrTemp2);    // simd load
                    simd_op(p2, p1);    // simd op
                    _mm256_storeu_si256((__m256i *)dstPtrTemp, p2);    // simd store
                    srcPtrTemp2 += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                 for (; vectorLoopCount < length[0]; vectorLoopCount++)
                 {
                     op(dstPtrTemp, srcPtrTemp1, srcPtrTemp2);
                     srcPtrTemp2++;
                     dstPtrTemp++;
                 }
            }
            else if (src2shape == 1)
            {
#if __AVX2__
                __m256i p2 = simd_set1_val(srcPtrTemp2[0]);    // simd broadcast
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256i p1 = _mm256_loadu_si256((const __m256i *)srcPtrTemp1);    // simd load
                    simd_op(p1, p2);    // simd op
                    _mm256_storeu_si256((__m256i *)dstPtrTemp, p1);    // simd store
                    srcPtrTemp1 += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                 for (; vectorLoopCount < length[0]; vectorLoopCount++)
                 {
                     op(dstPtrTemp, srcPtrTemp1, srcPtrTemp2);
                     srcPtrTemp1++;
                     dstPtrTemp++;
                 }
            }
            else
            {
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256i p1 = _mm256_loadu_si256((const __m256i *)srcPtrTemp1);    // simd load
                    __m256i p2 = _mm256_loadu_si256((const __m256i *)srcPtrTemp2);    // simd load
                    simd_op(p1, p2);    // simd op
                    _mm256_storeu_si256((__m256i *)dstPtrTemp, p1);    // simd store
                    srcPtrTemp1 += vectorIncrement;
                    srcPtrTemp2 += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < length[0]; vectorLoopCount++)
                {
                    op(dstPtrTemp, srcPtrTemp1, srcPtrTemp2);
                    srcPtrTemp1++;
                    srcPtrTemp2++;
                    dstPtrTemp++;
                }
            }
        }
        else if (dstDim == 2)
        {
            Rpp32u alignedLength = length[1] & ~alignMask;
            Rpp32u src1shape = src1length[1];
            Rpp32u src2shape = src2length[1];
            if(src1shape == 1)
            {
                for (int i = 0; i < length[0]; i++)
                {
                    T *srcPtrElem1 = srcPtrTemp1;
                    T *srcPtrElem2 = srcPtrTemp2;
                    T *dstPtrElem = dstPtrTemp;

                    int vectorLoopCount = 0;
#if __AVX2__
                    __m256i p1 = simd_set1_val(srcPtrElem1[0]);    // simd broadcast
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256i p2 = _mm256_loadu_si256((const __m256i *)srcPtrElem2);  // simd load
                        simd_op(p2, p1);    // simd op
                        _mm256_storeu_si256((__m256i *)dstPtrElem, p2);    // simd store
                        srcPtrElem2 += vectorIncrement;
                        dstPtrElem += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[1]; vectorLoopCount++)
                    {
                        op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                        srcPtrElem2++;
                        dstPtrElem++;
                    }
                    srcPtrTemp1 += src1BcastStrides[0];
                    srcPtrTemp2 += src2BcastStrides[0];
                    dstPtrTemp += dstBcastStrides[0];
                }
            }
            else if (src2shape == 1)
            {
                for (int i = 0; i < length[0]; i++)
                {
                    T *srcPtrElem1 = srcPtrTemp1;
                    T *srcPtrElem2 = srcPtrTemp2;
                    T *dstPtrElem = dstPtrTemp;

                    int vectorLoopCount = 0;
#if __AVX2__
                    __m256i p2 = simd_set1_val(srcPtrElem2[0]);    // simd broadcast
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256i p1 = _mm256_loadu_si256((const __m256i *)srcPtrElem1);    // simd load
                        simd_op(p1, p2);    // simd op
                        _mm256_storeu_si256((__m256i *)dstPtrElem, p1);    // simd store
                        srcPtrElem1 += vectorIncrement;
                        dstPtrElem += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[1]; vectorLoopCount++)
                    {
                        op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                        srcPtrElem1++;
                        dstPtrElem++;
                    }
                    srcPtrTemp1 += src1BcastStrides[0];
                    srcPtrTemp2 += src2BcastStrides[0];
                    dstPtrTemp += dstBcastStrides[0];
                }
            }
            else
            {
                for (int i = 0; i < length[0]; i++)
                {
                    T *srcPtrElem1 = srcPtrTemp1;
                    T *srcPtrElem2 = srcPtrTemp2;
                    T *dstPtrElem = dstPtrTemp;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256i p1 = _mm256_loadu_si256((const __m256i *)srcPtrElem1);    // simd load
                        __m256i p2 = _mm256_loadu_si256((const __m256i *)srcPtrElem2);    // simd load
                        simd_op(p1, p2);    // simd op
                        _mm256_storeu_si256((__m256i *)dstPtrElem, p1);    // simd store
                        srcPtrElem1 += vectorIncrement;
                        srcPtrElem2 += vectorIncrement;
                        dstPtrElem += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[1]; vectorLoopCount++)
                    {
                        op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                        srcPtrElem1++;
                        srcPtrElem2++;
                        dstPtrElem++;
                    }
                    srcPtrTemp1 += src1BcastStrides[0];
                    srcPtrTemp2 += src2BcastStrides[0];
                    dstPtrTemp += dstBcastStrides[0];
                }
            }
        }
        else if (dstDim == 3)
        {
            Rpp32u alignedLength = length[2] & ~alignMask;
            Rpp32u src1shape = src1length[2];
            Rpp32u src2shape = src2length[2];
            if(src1shape == 1)
            {
                for (int i = 0; i < length[0]; i++)
                {
                    T *srcPtrOuter1 = srcPtrTemp1;
                    T *srcPtrOuter2 = srcPtrTemp2;
                    T *dstPtrOuter = dstPtrTemp;

                    for (int j = 0; j < length[1]; j++)
                    {
                        T *srcPtrElem1 = srcPtrOuter1;
                        T *srcPtrElem2 = srcPtrOuter2;
                        T *dstPtrElem = dstPtrOuter;

                        int vectorLoopCount = 0;
#if __AVX2__
                        __m256i p1 = simd_set1_val(srcPtrElem1[0]);    // simd broadcast
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                        {
                            __m256i p2 = _mm256_loadu_si256((const __m256i *)srcPtrElem2);    // simd load
                            simd_op(p2, p1);    // simd op
                            _mm256_storeu_si256((__m256i *)dstPtrElem, p2);    // simd store
                            srcPtrElem2 += vectorIncrement;
                            dstPtrElem += vectorIncrement;
                        }
#endif
                        for (; vectorLoopCount < length[2]; vectorLoopCount++)
                        {
                            op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                            srcPtrElem2++;
                            dstPtrElem++;
                        }

                        srcPtrOuter1 += src1BcastStrides[1];
                        srcPtrOuter2 += src2BcastStrides[1];
                        dstPtrOuter += dstBcastStrides[1];
                    }

                    srcPtrTemp1 += src1BcastStrides[0];
                    srcPtrTemp2 += src2BcastStrides[0];
                    dstPtrTemp += dstBcastStrides[0];
                }
            }
            else if (src2shape == 1)
            {
                for (int i = 0; i < length[0]; i++)
                {
                    T *srcPtrOuter1 = srcPtrTemp1;
                    T *srcPtrOuter2 = srcPtrTemp2;
                    T *dstPtrOuter = dstPtrTemp;

                    for (int j = 0; j < length[1]; j++)
                    {
                        T *srcPtrElem1 = srcPtrOuter1;
                        T *srcPtrElem2 = srcPtrOuter2;
                        T *dstPtrElem = dstPtrOuter;

                        int vectorLoopCount = 0;
#if __AVX2__
                        __m256i p2 = simd_set1_val(srcPtrElem2[0]);    // simd broadcast
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                        {
                            __m256i p1 = _mm256_loadu_si256((const __m256i *)srcPtrElem1);    // simd load
                            simd_op(p1, p2);    // simd op
                            _mm256_storeu_si256((__m256i *)dstPtrElem, p1);    // simd store
                            srcPtrElem1 += vectorIncrement;
                            dstPtrElem += vectorIncrement;
                        }
#endif
                        for (; vectorLoopCount < length[2]; vectorLoopCount++)
                        {
                            op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                            srcPtrElem1++;
                            dstPtrElem++;
                        }

                        srcPtrOuter1 += src1BcastStrides[1];
                        srcPtrOuter2 += src2BcastStrides[1];
                        dstPtrOuter += dstBcastStrides[1];
                    }

                    srcPtrTemp1 += src1BcastStrides[0];
                    srcPtrTemp2 += src2BcastStrides[0];
                    dstPtrTemp += dstBcastStrides[0];
                }
            }
            else
            {
                for (int i = 0; i < length[0]; i++)
                {
                    T *srcPtrOuter1 = srcPtrTemp1;
                    T *srcPtrOuter2 = srcPtrTemp2;
                    T *dstPtrOuter = dstPtrTemp;

                    for (int j = 0; j < length[1]; j++)
                    {
                        T *srcPtrElem1 = srcPtrOuter1;
                        T *srcPtrElem2 = srcPtrOuter2;
                        T *dstPtrElem = dstPtrOuter;

                        int vectorLoopCount = 0;
#if __AVX2__
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                        {
                            __m256i p1 = _mm256_loadu_si256((const __m256i *)srcPtrElem1);    // simd load
                            __m256i p2 = _mm256_loadu_si256((const __m256i *)srcPtrElem2);    // simd load
                            simd_op(p1, p2);    // simd op
                            _mm256_storeu_si256((__m256i *)dstPtrElem, p1);    // simd store
                            srcPtrElem1 += vectorIncrement;
                            srcPtrElem2 += vectorIncrement;
                            dstPtrElem += vectorIncrement;
                        }
#endif
                        for (; vectorLoopCount < length[2]; vectorLoopCount++)
                        {
                            op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                            srcPtrElem1++;
                            srcPtrElem2++;
                            dstPtrElem++;
                        }

                        srcPtrOuter1 += src1BcastStrides[1];
                        srcPtrOuter2 += src2BcastStrides[1];
                        dstPtrOuter += dstBcastStrides[1];
                    }

                    srcPtrTemp1 += src1BcastStrides[0];
                    srcPtrTemp2 += src2BcastStrides[0];
                    dstPtrTemp += dstBcastStrides[0];
                }
            }
        }
        else
            tensor_binary_arithmetic_op_recursive(srcPtrTemp1, srcPtrTemp2, src1BcastStrides, src2BcastStrides, dstPtrTemp, dstBcastStrides, length, dstDim, op);
    }

    return RPP_SUCCESS;
}

template<typename T, typename Operation, typename SIMDOperation>
RppStatus tensor_binary_divide_host_tensor(T *srcPtr1,
                                           T *srcPtr2,
                                           RpptGenericDescPtr srcPtr1GenericDescPtr,
                                           RpptGenericDescPtr srcPtr2GenericDescPtr,
                                           Rpp32f *dstPtr,
                                           RpptGenericDescPtr dstGenericDescPtr,
                                           Operation op,
                                           SIMDOperation simd_op,
                                           RpptBroadcastMode broadcastMode,
                                           Rpp32u vectorIncrement,
                                           Rpp32u *srcPtr1roiTensor,
                                           Rpp32u *srcPtr2roiTensor,
                                           rpp::Handle& handle)
{

    printf("tensor_binary_divide_host_tensor invoked\n");

    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u src1NDim = srcPtr1GenericDescPtr->numDims - 1;  // Omitting batchSize here to get tensor dimension
    Rpp32u src2NDim = srcPtr2GenericDescPtr->numDims - 1;  // Omitting batchSize here to get tensor dimension
    Rpp32u dstDim = src1NDim > src2NDim ? src1NDim : src2NDim; // Destination dimension set to maximum of the input dimensions
    Rpp32u minDim = src1NDim < src2NDim ? src1NDim : src2NDim; // Minimum of input dimensions

    // Overall dimension compatibility check for the entire batch
    for(int test = 0; test < minDim; test++)
        if(srcPtr1GenericDescPtr->dims[src1NDim - test] != srcPtr2GenericDescPtr->dims[src2NDim - test])
            if((srcPtr1GenericDescPtr->dims[src1NDim - test] != 1) && (srcPtr2GenericDescPtr->dims[src2NDim - test] != 1))
            {
                printf("Incompatible dimensions for the batch\n");
                return RPP_SUCCESS;
            }

    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        Rpp32u *src1roi = srcPtr1roiTensor + batchCount * src1NDim * 2;
        Rpp32u *src1Begin = src1roi;
        Rpp32u *src1Dims = src1Begin + src1NDim;

        Rpp32u *src2roi = srcPtr2roiTensor + batchCount * src2NDim * 2;
        Rpp32u *src2Begin = src2roi;
        Rpp32u *src2Dims = src2Begin + src2NDim;

        Rpp32u *src1Strides = srcPtr1GenericDescPtr->strides;
        Rpp32u *src2Strides = srcPtr2GenericDescPtr->strides;
        Rpp32u *dstStrides = dstGenericDescPtr->strides;

        Rpp32u *length, *src1length, *src2length, *src1BcastStrides, *src2BcastStrides, *dstBcastStrides;

        if(broadcastMode == RPP_BROADCAST_ENABLE)
        {
            // Dimensions and Strides based on individual sample ROIs, used for broadcasting purposes
            Rpp32u src1BroadcastDims[RPPT_MAX_DIMS_SAMPLE], src2BroadcastDims[RPPT_MAX_DIMS_SAMPLE], dstBroadcastDims[RPPT_MAX_DIMS_SAMPLE];
            Rpp32u src1BroadcastStrides[RPPT_MAX_DIMS_SAMPLE], src2BroadcastStrides[RPPT_MAX_DIMS_SAMPLE], dstBroadcastStrides[RPPT_MAX_DIMS_SAMPLE];

            bool incompatibleDims = false;

            // Copy ROI limits and Strides to individual sample strides and dims until minDim
            for(int i = 0; i < minDim; i++)
            {
                Rpp32u curIndex = RPPT_MAX_DIMS_SAMPLE - i - 1;
                src1BroadcastDims[curIndex] = src1Dims[src1NDim - i - 1];
                src2BroadcastDims[curIndex] = src2Dims[src2NDim - i - 1];
                src1BroadcastStrides[curIndex] = src1Strides[src1NDim - i];
                src2BroadcastStrides[curIndex] = src2Strides[src2NDim - i];
                dstBroadcastStrides[curIndex] = dstStrides[dstDim - i];
                // Check compatibility of dimension i.e check for equal shape or one of the input dims to be 1
                if((src1BroadcastDims[curIndex] != src2BroadcastDims[curIndex]) && (src1BroadcastDims[curIndex] != 1) && (src2BroadcastDims[curIndex] != 1))
                    incompatibleDims = true;
                dstBroadcastDims[curIndex] = std::max(src1BroadcastDims[curIndex], src2BroadcastDims[curIndex]);
            }

            // Dimension compatibility failure case
            if(incompatibleDims == true)
                printf("Incompatible dimensions for operation for sample %d inside batch\n", batchCount);

            // Handle cases of mismatching num dims
            if(src1NDim < src2NDim)
                for(int i = minDim; i < dstDim; i++)
                {
                    Rpp32u curIndex = RPPT_MAX_DIMS_SAMPLE - i - 1;
                    src1BroadcastDims[curIndex] = 1;
                    src2BroadcastDims[curIndex] = src2Dims[src2NDim - i];
                    dstBroadcastDims[curIndex] = src2Dims[src2NDim - i];
                    src1BroadcastStrides[curIndex] = 0;
                    src2BroadcastStrides[curIndex] = src2Strides[src2NDim - i];
                    dstBroadcastStrides[curIndex] = dstStrides[dstDim - i];
                }
            else if(src1NDim > src2NDim)
                for(int i = minDim; i < dstDim; i++)
                {
                    Rpp32u curIndex = RPPT_MAX_DIMS_SAMPLE - i - 1;
                    src2BroadcastDims[curIndex] = 1;
                    src1BroadcastDims[curIndex] = src1Dims[src1NDim - i];
                    dstBroadcastDims[curIndex] = src1Dims[src1NDim - i];
                    src1BroadcastStrides[curIndex] = src1Strides[src1NDim - i];
                    src2BroadcastStrides[curIndex] = 0;
                    dstBroadcastStrides[curIndex] = dstStrides[dstDim - i];
                }

            // Source strides for sample set to zero if corresponding axis shape = 1 for broadcasting purposes
            // Setting stride to zero will allow for repetition of values operated required for broadcasting
            for(int i = 0; i < minDim; i++)
            {
                if((src1BroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i] != dstBroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i]) && (src1BroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i] == 1))
                {
                    src1BroadcastStrides[RPPT_MAX_DIMS_SAMPLE - 1 - i] = 0;
                }
                if((src2BroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i] != dstBroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i]) && (src2BroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i] == 1))
                {
                    src2BroadcastStrides[RPPT_MAX_DIMS_SAMPLE - 1 - i] = 0;
                }
            }

            Rpp32u testOffset = RPPT_MAX_DIMS_SAMPLE - dstDim;

            // Shift dims and strides by offset to process only valid values
            length = dstBroadcastDims + testOffset;
            src1length = src1BroadcastDims + testOffset;
            src2length = src2BroadcastDims + testOffset;

            src1BcastStrides = src1BroadcastStrides + testOffset;
            src2BcastStrides = src2BroadcastStrides + testOffset;
            dstBcastStrides =  dstBroadcastStrides + testOffset;
        }
        else
        {
            // Both length for dest and srclength set to src1Dims because in non broadcast case, source and destination dimensions are expected to be the same
            length = src1Dims;
            src1length = src1Dims;
            src2length = src2Dims;

            src1BcastStrides = srcPtr1GenericDescPtr->strides + 1;
            src2BcastStrides = srcPtr2GenericDescPtr->strides + 1;
            dstBcastStrides =  dstGenericDescPtr->strides + 1;
        }

        T *srcPtrTemp1 = srcPtr1 + batchCount * srcPtr1GenericDescPtr->strides[0];
        T *srcPtrTemp2 = srcPtr2 + batchCount * srcPtr2GenericDescPtr->strides[0];

        for(int i = 0; i < src1NDim; i++)
            srcPtrTemp1 += src1Begin[i] * srcPtr1GenericDescPtr->strides[i + 1];

        for(int i = 0; i < src2NDim; i++)
            srcPtrTemp2 += src2Begin[i] * srcPtr2GenericDescPtr->strides[i + 1];

        Rpp32f *dstPtrTemp = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        Rpp32u alignMask = vectorIncrement - 1;

        // For nDim = 1, 2, 3 cases handled when the lowest axis shape = 1 and != 1 for efficient processing
        if (dstDim == 1)
        {
            Rpp32u alignedLength = length[0] & ~alignMask;
            Rpp32u src1shape = src1length[0];
            Rpp32u src2shape = src2length[0];
            Rpp32u vectorLoopCount = 0;
            if (src1shape == 1)
            {
#if __AV__
                __m256i p1 = simd_set1_val(srcPtrTemp1[0]);    // simd broadcast
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256i p2 = _mm256_loadu_si256((const __m256i *)srcPtrTemp2);    // simd load
                    simd_op(p2, p1);    // simd op
                    _mm256_storeu_si256((__m256i *)dstPtrTemp, p2);    // simd store
                    srcPtrTemp2 += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                 for (; vectorLoopCount < length[0]; vectorLoopCount++)
                 {
                     op(dstPtrTemp, srcPtrTemp1, srcPtrTemp2);
                     srcPtrTemp2++;
                     dstPtrTemp++;
                 }
            }
            else if (src2shape == 1)
            {
#if __AV__
                __m256i p2 = simd_set1_val(srcPtrTemp2[0]);    // simd broadcast
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256i p1 = _mm256_loadu_si256((const __m256i *)srcPtrTemp1);    // simd load
                    simd_op(p1, p2);    // simd op
                    _mm256_storeu_si256((__m256i *)dstPtrTemp, p1);    // simd store
                    srcPtrTemp1 += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                 for (; vectorLoopCount < length[0]; vectorLoopCount++)
                 {
                     op(dstPtrTemp, srcPtrTemp1, srcPtrTemp2);
                     srcPtrTemp1++;
                     dstPtrTemp++;
                 }
            }
            else
            {
#if __AVX2__
                __m256 pout[4];
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256i p1 = _mm256_loadu_si256((const __m256i *)srcPtrTemp1);    // simd load
                    __m256i p2 = _mm256_loadu_si256((const __m256i *)srcPtrTemp2);    // simd load
                    simd_op(pout, p1, p2);    // simd op
                    store_ps_function<T>(pout, dstPtrTemp);    // simd store
                    srcPtrTemp1 += vectorIncrement;
                    srcPtrTemp2 += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < length[0]; vectorLoopCount++)
                {
                    op(dstPtrTemp, srcPtrTemp1, srcPtrTemp2);
                    srcPtrTemp1++;
                    srcPtrTemp2++;
                    dstPtrTemp++;
                }
            }
        }
        else if (dstDim == 2)
        {
            Rpp32u alignedLength = length[1] & ~alignMask;
            Rpp32u src1shape = src1length[1];
            Rpp32u src2shape = src2length[1];
            if(src1shape == 1)
            {
                for (int i = 0; i < length[0]; i++)
                {
                    T *srcPtrElem1 = srcPtrTemp1;
                    T *srcPtrElem2 = srcPtrTemp2;
                    Rpp32f *dstPtrElem = dstPtrTemp;

                    int vectorLoopCount = 0;
#if __AV__
                    __m256i p1 = simd_set1_val(srcPtrElem1[0]);    // simd broadcast
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256i p2 = _mm256_loadu_si256((const __m256i *)srcPtrElem2);  // simd load
                        simd_op(p2, p1);    // simd op
                        _mm256_storeu_si256((__m256i *)dstPtrElem, p2);    // simd store
                        srcPtrElem2 += vectorIncrement;
                        dstPtrElem += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[1]; vectorLoopCount++)
                    {
                        op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                        srcPtrElem2++;
                        dstPtrElem++;
                    }
                    srcPtrTemp1 += src1BcastStrides[0];
                    srcPtrTemp2 += src2BcastStrides[0];
                    dstPtrTemp += dstBcastStrides[0];
                }
            }
            else if (src2shape == 1)
            {
                for (int i = 0; i < length[0]; i++)
                {
                    T *srcPtrElem1 = srcPtrTemp1;
                    T *srcPtrElem2 = srcPtrTemp2;
                    Rpp32f *dstPtrElem = dstPtrTemp;

                    int vectorLoopCount = 0;
#if __AV__
                    __m256i p2 = simd_set1_val(srcPtrElem2[0]);    // simd broadcast
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256i p1 = _mm256_loadu_si256((const __m256i *)srcPtrElem1);    // simd load
                        simd_op(p1, p2);    // simd op
                        _mm256_storeu_si256((__m256i *)dstPtrElem, p1);    // simd store
                        srcPtrElem1 += vectorIncrement;
                        dstPtrElem += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[1]; vectorLoopCount++)
                    {
                        op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                        srcPtrElem1++;
                        dstPtrElem++;
                    }
                    srcPtrTemp1 += src1BcastStrides[0];
                    srcPtrTemp2 += src2BcastStrides[0];
                    dstPtrTemp += dstBcastStrides[0];
                }
            }
            else
            {
                for (int i = 0; i < length[0]; i++)
                {
                    T *srcPtrElem1 = srcPtrTemp1;
                    T *srcPtrElem2 = srcPtrTemp2;
                    Rpp32f *dstPtrElem = dstPtrTemp;

                    int vectorLoopCount = 0;
#if __AVX2__
                    __m256 pout[4];
                    printf("alignedLength is %d\n", alignedLength);
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256i p1 = _mm256_loadu_si256((const __m256i *)srcPtrElem1);    // simd load
                        __m256i p2 = _mm256_loadu_si256((const __m256i *)srcPtrElem2);    // simd load
                        simd_op(pout, p1, p2);    // simd op
                        store_ps_function<T>(pout, dstPtrElem);    // simd store
                        srcPtrElem1 += vectorIncrement;
                        srcPtrElem2 += vectorIncrement;
                        dstPtrElem += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[1]; vectorLoopCount++)
                    {
                        op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                        srcPtrElem1++;
                        srcPtrElem2++;
                        dstPtrElem++;
                    }
                    srcPtrTemp1 += src1BcastStrides[0];
                    srcPtrTemp2 += src2BcastStrides[0];
                    dstPtrTemp += dstBcastStrides[0];
                }
            }
        }
        else if (dstDim == 3)
        {
            Rpp32u alignedLength = length[2] & ~alignMask;
            Rpp32u src1shape = src1length[2];
            Rpp32u src2shape = src2length[2];
            if(src1shape == 1)
            {
                for (int i = 0; i < length[0]; i++)
                {
                    T *srcPtrOuter1 = srcPtrTemp1;
                    T *srcPtrOuter2 = srcPtrTemp2;
                    Rpp32f *dstPtrOuter = dstPtrTemp;

                    for (int j = 0; j < length[1]; j++)
                    {
                        T *srcPtrElem1 = srcPtrOuter1;
                        T *srcPtrElem2 = srcPtrOuter2;
                        Rpp32f *dstPtrElem = dstPtrOuter;

                        int vectorLoopCount = 0;
#if __AV__
                        __m256i p1 = simd_set1_val(srcPtrElem1[0]);    // simd broadcast
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                        {
                            __m256i p2 = _mm256_loadu_si256((const __m256i *)srcPtrElem2);    // simd load
                            simd_op(p2, p1);    // simd op
                            _mm256_storeu_si256((__m256i *)dstPtrElem, p2);    // simd store
                            srcPtrElem2 += vectorIncrement;
                            dstPtrElem += vectorIncrement;
                        }
#endif
                        for (; vectorLoopCount < length[2]; vectorLoopCount++)
                        {
                            op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                            srcPtrElem2++;
                            dstPtrElem++;
                        }

                        srcPtrOuter1 += src1BcastStrides[1];
                        srcPtrOuter2 += src2BcastStrides[1];
                        dstPtrOuter += dstBcastStrides[1];
                    }

                    srcPtrTemp1 += src1BcastStrides[0];
                    srcPtrTemp2 += src2BcastStrides[0];
                    dstPtrTemp += dstBcastStrides[0];
                }
            }
            else if (src2shape == 1)
            {
                for (int i = 0; i < length[0]; i++)
                {
                    T *srcPtrOuter1 = srcPtrTemp1;
                    T *srcPtrOuter2 = srcPtrTemp2;
                    Rpp32f *dstPtrOuter = dstPtrTemp;

                    for (int j = 0; j < length[1]; j++)
                    {
                        T *srcPtrElem1 = srcPtrOuter1;
                        T *srcPtrElem2 = srcPtrOuter2;
                        Rpp32f *dstPtrElem = dstPtrOuter;

                        int vectorLoopCount = 0;
#if __AV__
                        __m256i p2 = simd_set1_val(srcPtrElem2[0]);    // simd broadcast
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                        {
                            __m256i p1 = _mm256_loadu_si256((const __m256i *)srcPtrElem1);    // simd load
                            simd_op(p1, p2);    // simd op
                            _mm256_storeu_si256((__m256i *)dstPtrElem, p1);    // simd store
                            srcPtrElem1 += vectorIncrement;
                            dstPtrElem += vectorIncrement;
                        }
#endif
                        for (; vectorLoopCount < length[2]; vectorLoopCount++)
                        {
                            op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                            srcPtrElem1++;
                            dstPtrElem++;
                        }

                        srcPtrOuter1 += src1BcastStrides[1];
                        srcPtrOuter2 += src2BcastStrides[1];
                        dstPtrOuter += dstBcastStrides[1];
                    }

                    srcPtrTemp1 += src1BcastStrides[0];
                    srcPtrTemp2 += src2BcastStrides[0];
                    dstPtrTemp += dstBcastStrides[0];
                }
            }
            else
            {
                for (int i = 0; i < length[0]; i++)
                {
                    T *srcPtrOuter1 = srcPtrTemp1;
                    T *srcPtrOuter2 = srcPtrTemp2;
                    Rpp32f *dstPtrOuter = dstPtrTemp;

                    for (int j = 0; j < length[1]; j++)
                    {
                        T *srcPtrElem1 = srcPtrOuter1;
                        T *srcPtrElem2 = srcPtrOuter2;
                        Rpp32f *dstPtrElem = dstPtrOuter;

                        int vectorLoopCount = 0;
#if __AVX2__
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                        {
                            __m256i p1 = _mm256_loadu_si256((const __m256i *)srcPtrElem1);    // simd load
                            __m256i p2 = _mm256_loadu_si256((const __m256i *)srcPtrElem2);    // simd load
                            simd_op(pout, p1, p2);    // simd op
                            store_ps_function<T>(pout, dstPtrElem);    // simd store
                            srcPtrElem1 += vectorIncrement;
                            srcPtrElem2 += vectorIncrement;
                            dstPtrElem += vectorIncrement;
                        }
#endif
                        for (; vectorLoopCount < length[2]; vectorLoopCount++)
                        {
                            op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                            srcPtrElem1++;
                            srcPtrElem2++;
                            dstPtrElem++;
                        }

                        srcPtrOuter1 += src1BcastStrides[1];
                        srcPtrOuter2 += src2BcastStrides[1];
                        dstPtrOuter += dstBcastStrides[1];
                    }

                    srcPtrTemp1 += src1BcastStrides[0];
                    srcPtrTemp2 += src2BcastStrides[0];
                    dstPtrTemp += dstBcastStrides[0];
                }
            }
        }
        else
            tensor_binary_arithmetic_op_recursive(srcPtrTemp1, srcPtrTemp2, src1BcastStrides, src2BcastStrides, dstPtrTemp, dstBcastStrides, length, dstDim, op);
    }

    return RPP_SUCCESS;
}

// Dispatcher function that dispatches the calls to the appropriate templated function based on the datatype and operation
template<typename T1, typename T2>
RppStatus tensor_binary_bitwise_op_dispatch_int_host_tensor(T1 *srcPtr1,
                                                            T1 *srcPtr2,
                                                            RpptGenericDescPtr srcPtr1GenericDescPtr,
                                                            RpptGenericDescPtr srcPtr2GenericDescPtr,
                                                            T2 *dstPtr,
                                                            RpptGenericDescPtr dstGenericDescPtr,
                                                            RpptOp tensorOp,
                                                            RpptBroadcastMode broadcastMode,
                                                            Rpp32u *srcPtr1roiTensor,
                                                            Rpp32u *srcPtr2roiTensor,
                                                            rpp::Handle& handle)
{
    int vectorIncrement = 32; // Vector Increment for U8/I8 datatype
    if((srcPtr1GenericDescPtr->dataType == RpptDataType::U16) || (srcPtr1GenericDescPtr->dataType == RpptDataType::I16))
        vectorIncrement = 16; // Vector Increment for U16/I16 datatype
    else if((srcPtr1GenericDescPtr->dataType == RpptDataType::U32) || (srcPtr1GenericDescPtr->dataType == RpptDataType::I32))
        vectorIncrement = 8; // Vector Increment for U32/I32 datatype

    if(((tensorOp == RPP_TENSOR_OP_DIVIDE) && (srcPtr1GenericDescPtr->dataType == RpptDataType::U32))  || ((tensorOp == RPP_TENSOR_OP_MULTIPLY) && ((srcPtr1GenericDescPtr->dataType == RpptDataType::U8) || (srcPtr1GenericDescPtr->dataType == RpptDataType::I8))))
        vectorIncrement = 0;

    printf("Vector Increment is %d\n", vectorIncrement);

    printf("Inside overall dispatch function\n");

    if constexpr (std::is_same_v<T1, T2>)
    {
        switch(tensorOp) {
            case RPP_TENSOR_OP_ADD:
                tensor_binary_op_int_host_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, add_op<T1>, simd_add_si256<T1>, broadcastMode, vectorIncrement, srcPtr1roiTensor, srcPtr2roiTensor, handle);
                break;
            case RPP_TENSOR_OP_SUBTRACT:
                tensor_binary_op_int_host_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, subtract_op<T1>, simd_subtract_si256<T1>, broadcastMode, vectorIncrement, srcPtr1roiTensor, srcPtr2roiTensor, handle);
                break;
            case RPP_TENSOR_OP_MULTIPLY:
                tensor_binary_op_int_host_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, multiply_op<T1>, simd_multiply_si256<T1>, broadcastMode, vectorIncrement, srcPtr1roiTensor, srcPtr2roiTensor, handle);
                break;
            default :
                printf("Operation not supported\n");
                break;
        }
    }
    if constexpr (std::is_same_v<T2, Rpp32f>)
    {
        switch(tensorOp) {
            case RPP_TENSOR_OP_DIVIDE:
                tensor_binary_divide_host_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, divide_op<T2, T1>, simd_divide_si256<T1>, broadcastMode, vectorIncrement, srcPtr1roiTensor, srcPtr2roiTensor, handle);
                break;
            default :
                printf("Operation not supported\n");
                break;
        }
    }

    return RPP_SUCCESS;
}

template RppStatus tensor_binary_bitwise_op_dispatch_int_host_tensor<Rpp8u, Rpp8u>(Rpp8u*,
                                                                                   Rpp8u*,
                                                                                   RpptGenericDescPtr,
                                                                                   RpptGenericDescPtr,
                                                                                   Rpp8u*,
                                                                                   RpptGenericDescPtr,
                                                                                   RpptOp,
                                                                                   RpptBroadcastMode,
                                                                                   Rpp32u*,
                                                                                   Rpp32u*,
                                                                                   rpp::Handle&);

template RppStatus tensor_binary_bitwise_op_dispatch_int_host_tensor<Rpp8u, Rpp32f>(Rpp8u*,
                                                                                    Rpp8u*,
                                                                                    RpptGenericDescPtr,
                                                                                    RpptGenericDescPtr,
                                                                                    Rpp32f*,
                                                                                    RpptGenericDescPtr,
                                                                                    RpptOp,
                                                                                    RpptBroadcastMode,
                                                                                    Rpp32u*,
                                                                                    Rpp32u*,
                                                                                    rpp::Handle&);

template RppStatus tensor_binary_bitwise_op_dispatch_int_host_tensor<Rpp8s, Rpp8s>(Rpp8s*,
                                                                                   Rpp8s*,
                                                                                   RpptGenericDescPtr,
                                                                                   RpptGenericDescPtr,
                                                                                   Rpp8s*,
                                                                                   RpptGenericDescPtr,
                                                                                   RpptOp,
                                                                                   RpptBroadcastMode,
                                                                                   Rpp32u*,
                                                                                   Rpp32u*,
                                                                                   rpp::Handle&);

template RppStatus tensor_binary_bitwise_op_dispatch_int_host_tensor<Rpp8s, Rpp32f>(Rpp8s*,
                                                                                    Rpp8s*,
                                                                                    RpptGenericDescPtr,
                                                                                    RpptGenericDescPtr,
                                                                                    Rpp32f*,
                                                                                    RpptGenericDescPtr,
                                                                                    RpptOp,
                                                                                    RpptBroadcastMode,
                                                                                    Rpp32u*,
                                                                                    Rpp32u*,
                                                                                    rpp::Handle&);

template RppStatus tensor_binary_bitwise_op_dispatch_int_host_tensor<Rpp16u, Rpp16u>(Rpp16u*,
                                                                                     Rpp16u*,
                                                                                     RpptGenericDescPtr,
                                                                                     RpptGenericDescPtr,
                                                                                     Rpp16u*,
                                                                                     RpptGenericDescPtr,
                                                                                     RpptOp,
                                                                                     RpptBroadcastMode,
                                                                                     Rpp32u*,
                                                                                     Rpp32u*,
                                                                                     rpp::Handle&);

template RppStatus tensor_binary_bitwise_op_dispatch_int_host_tensor<Rpp16u, Rpp32f>(Rpp16u*,
                                                                                     Rpp16u*,
                                                                                     RpptGenericDescPtr,
                                                                                     RpptGenericDescPtr,
                                                                                     Rpp32f*,
                                                                                     RpptGenericDescPtr,
                                                                                     RpptOp,
                                                                                     RpptBroadcastMode,
                                                                                     Rpp32u*,
                                                                                     Rpp32u*,
                                                                                     rpp::Handle&);

template RppStatus tensor_binary_bitwise_op_dispatch_int_host_tensor<Rpp16s, Rpp16s>(Rpp16s*,
                                                                                     Rpp16s*,
                                                                                     RpptGenericDescPtr,
                                                                                     RpptGenericDescPtr,
                                                                                     Rpp16s*,
                                                                                     RpptGenericDescPtr,
                                                                                     RpptOp,
                                                                                     RpptBroadcastMode,
                                                                                     Rpp32u*,
                                                                                     Rpp32u*,
                                                                                     rpp::Handle&);

template RppStatus tensor_binary_bitwise_op_dispatch_int_host_tensor<Rpp16s, Rpp32f>(Rpp16s*,
                                                                                     Rpp16s*,
                                                                                     RpptGenericDescPtr,
                                                                                     RpptGenericDescPtr,
                                                                                     Rpp32f*,
                                                                                     RpptGenericDescPtr,
                                                                                     RpptOp,
                                                                                     RpptBroadcastMode,
                                                                                     Rpp32u*,
                                                                                     Rpp32u*,
                                                                                     rpp::Handle&);

template RppStatus tensor_binary_bitwise_op_dispatch_int_host_tensor<Rpp32u, Rpp32u>(Rpp32u*,
                                                                                     Rpp32u*,
                                                                                     RpptGenericDescPtr,
                                                                                     RpptGenericDescPtr,
                                                                                     Rpp32u*,
                                                                                     RpptGenericDescPtr,
                                                                                     RpptOp,
                                                                                     RpptBroadcastMode,
                                                                                     Rpp32u*,
                                                                                     Rpp32u*,
                                                                                     rpp::Handle&);

template RppStatus tensor_binary_bitwise_op_dispatch_int_host_tensor<Rpp32u, Rpp32f>(Rpp32u*,
                                                                                     Rpp32u*,
                                                                                     RpptGenericDescPtr,
                                                                                     RpptGenericDescPtr,
                                                                                     Rpp32f*,
                                                                                     RpptGenericDescPtr,
                                                                                     RpptOp,
                                                                                     RpptBroadcastMode,
                                                                                     Rpp32u*,
                                                                                     Rpp32u*,
                                                                                     rpp::Handle&);

template RppStatus tensor_binary_bitwise_op_dispatch_int_host_tensor<Rpp32s, Rpp32s>(Rpp32s*,
                                                                                     Rpp32s*,
                                                                                     RpptGenericDescPtr,
                                                                                     RpptGenericDescPtr,
                                                                                     Rpp32s*,
                                                                                     RpptGenericDescPtr,
                                                                                     RpptOp,
                                                                                     RpptBroadcastMode,
                                                                                     Rpp32u*,
                                                                                     Rpp32u*,
                                                                                     rpp::Handle&);

template RppStatus tensor_binary_bitwise_op_dispatch_int_host_tensor<Rpp32s, Rpp32f>(Rpp32s*,
                                                                                     Rpp32s*,
                                                                                     RpptGenericDescPtr,
                                                                                     RpptGenericDescPtr,
                                                                                     Rpp32f*,
                                                                                     RpptGenericDescPtr,
                                                                                     RpptOp,
                                                                                     RpptBroadcastMode,
                                                                                     Rpp32u*,
                                                                                     Rpp32u*,
                                                                                     rpp::Handle&);