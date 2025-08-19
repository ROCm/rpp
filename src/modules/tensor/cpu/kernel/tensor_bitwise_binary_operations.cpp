#include "host_tensor_executors.hpp"
#include "broadcast.hpp"
#include "rpp_cpu_simd_math.hpp"

template<typename T>
inline void and_op(T *dst, T *src1, T *src2) { *dst = *src1 & *src2; }

template<typename T>
inline void or_op(T *dst, T *src1, T *src2) { *dst = *src1 | *src2; }

template<typename T>
inline void xor_op(T *dst, T *src1, T *src2) { *dst = *src1 ^ *src2; }

inline void simd_and_si256(__m256i &a, __m256i &b) { a = _mm256_and_si256(a, b); }
inline void simd_or_si256(__m256i &a, __m256i &b) { a = _mm256_or_si256(a, b); }
inline void simd_xor_si256(__m256i &a, __m256i &b) { a = _mm256_xor_si256(a, b); }

inline __m256i simd_set1_val(Rpp8u &val) { return _mm256_set1_epi8(val); }
inline __m256i simd_set1_val(Rpp16u &val) { return _mm256_set1_epi16(val); }
inline __m256i simd_set1_val(Rpp32u &val) { return _mm256_set1_epi32(val); }

template<typename T, typename Operation>
inline void tensor_binary_op_recursive(T *src1, T *src2, Rpp32u *src1Strides, Rpp32u *src2Strides, T *dst, Rpp32u *dstStrides, Rpp32u *dstShape, Rpp32u nDim, Operation op)
{
    if (!nDim)
        op(dst, src1, src2);
    else
    {
        for (int i = 0; i < *dstShape; i++)
        {
            tensor_binary_op_recursive(src1, src2, src1Strides + 1, src2Strides + 1, dst, dstStrides + 1, dstShape + 1, nDim - 1, op);
            dst += *(dstStrides);
            src1 += *(src1Strides);
            src2 += *(src2Strides);
        }
    }
}

template<typename T, typename Operation, typename SIMDOperation>
RppStatus tensor_binary_bitwise_op_host_tensor(T *srcPtr1,
                                               T *srcPtr2,
                                               RpptGenericDescPtr srcPtr1GenericDescPtr,
                                               RpptGenericDescPtr srcPtr2GenericDescPtr,
                                               T *dstPtr,
                                               RpptGenericDescPtr dstGenericDescPtr,
                                               Operation op,
                                               SIMDOperation simd_op,
                                               Rpp32u vectorIncrement,
                                               Rpp32u *srcPtr1roiTensor,
                                               Rpp32u *srcPtr2roiTensor,
                                               rpp::Handle& handle) {

    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u src1NDim = srcPtr1GenericDescPtr->numDims - 1;
    Rpp32u src2NDim = srcPtr2GenericDescPtr->numDims - 1;
    Rpp32u dstDim = src1NDim > src2NDim ? src1NDim : src2NDim;
    Rpp32u minDim = src1NDim < src2NDim ? src1NDim : src2NDim;

    for(int test = 0; test < minDim; test++) {
        if(srcPtr1GenericDescPtr->dims[src1NDim - test] != srcPtr2GenericDescPtr->dims[src2NDim - test])
            if((srcPtr1GenericDescPtr->dims[src1NDim - test] != 1) && (srcPtr2GenericDescPtr->dims[src2NDim - test] != 1)) {
                printf("Incompatible dimensions for the batch\n");
                return RPP_SUCCESS;
        }
    }

    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        Rpp32u *src1roi = srcPtr1roiTensor + batchCount * src1NDim * 2;
        Rpp32u *src1begin = src1roi;
        Rpp32u *src1dims = src1begin + src1NDim;

        Rpp32u *src2roi = srcPtr2roiTensor + batchCount * src2NDim * 2;
        Rpp32u *src2begin = src2roi;
        Rpp32u *src2dims = src2begin + src2NDim;

        Rpp32u *src1Strides = srcPtr1GenericDescPtr->strides;
        Rpp32u *src2Strides = srcPtr2GenericDescPtr->strides;
        Rpp32u *dstStrides = dstGenericDescPtr->strides;

        // These are the dimensions that are based on individual ROIs, and strides are separate for each sample in the batch
        Rpp32u src1BroadcastDims[RPPT_MAX_DIMS], src2BroadcastDims[RPPT_MAX_DIMS], dstBroadcastDims[RPPT_MAX_DIMS];
        Rpp32u src1BroadcastStrides[RPPT_MAX_DIMS], src2BroadcastStrides[RPPT_MAX_DIMS], dstBroadcastStrides[RPPT_MAX_DIMS];

        bool incompatibleDims = false;

        for(int i = 0; i < minDim; i++) {
            Rpp32u curIndex = RPPT_MAX_DIMS - i - 1;
            src1BroadcastDims[curIndex] = src1dims[src1NDim - i - 1];
            src2BroadcastDims[curIndex] = src2dims[src2NDim - i - 1];
            src1BroadcastStrides[curIndex] = src1Strides[src1NDim - i];
            src2BroadcastStrides[curIndex] = src2Strides[src2NDim - i];
            dstBroadcastStrides[curIndex] = dstStrides[dstDim - i];
            if((src1BroadcastDims[curIndex] != src2BroadcastDims[curIndex]) && (src1BroadcastDims[curIndex] != 1) && (src2BroadcastDims[curIndex] != 1))
                incompatibleDims = true;
            dstBroadcastDims[curIndex] = src1BroadcastDims[curIndex] > src2BroadcastDims[curIndex] ? src1BroadcastDims[curIndex] : src2BroadcastDims[curIndex];
        }
        if(incompatibleDims == true) {
            printf("Incompatible dimensions for operation for sample %d inside batch\n", batchCount);
        }
        if(src1NDim < src2NDim) {
            for(int i = minDim; i < dstDim; i++){
                Rpp32u curIndex = RPPT_MAX_DIMS - i - 1;
                src1BroadcastDims[curIndex] = 1;
                src2BroadcastDims[curIndex] = src2dims[src2NDim - i];
                dstBroadcastDims[curIndex] = src2dims[src2NDim - i];
                src1BroadcastStrides[curIndex] = 0;
                src2BroadcastStrides[curIndex] = src2Strides[src2NDim - i];
                dstBroadcastStrides[curIndex] = dstStrides[dstDim - i];
            }
        }
        else if(src1NDim > src2NDim) {
            for(int i = minDim; i < dstDim; i++){
                Rpp32u curIndex = RPPT_MAX_DIMS - i - 1;
                src2BroadcastDims[curIndex] = 1;
                src1BroadcastDims[curIndex] = src1dims[src1NDim - i];
                dstBroadcastDims[curIndex] = src1dims[src1NDim - i];
                src1BroadcastStrides[curIndex] = src1Strides[src1NDim - i];
                src2BroadcastStrides[curIndex] = 0;
                dstBroadcastStrides[curIndex] = dstStrides[dstDim - i];
            }
        }
        for(int i = 0; i < minDim; i++) {
            if((src1BroadcastDims[RPPT_MAX_DIMS - 1 - i] != dstBroadcastDims[RPPT_MAX_DIMS - 1 - i]) && (src1BroadcastDims[RPPT_MAX_DIMS - 1 - i] == 1)) {
                src1BroadcastStrides[RPPT_MAX_DIMS - 1 - i] = 0;
            }
            if((src2BroadcastDims[RPPT_MAX_DIMS - 1 - i] != dstBroadcastDims[RPPT_MAX_DIMS - 1 - i]) && (src2BroadcastDims[RPPT_MAX_DIMS - 1 - i] == 1)) {
                src2BroadcastStrides[RPPT_MAX_DIMS - 1 - i] = 0;
            }
        }

        T *srcPtrTemp1 = srcPtr1 + batchCount * srcPtr1GenericDescPtr->strides[0];
        T *srcPtrTemp2 = srcPtr2 + batchCount * srcPtr2GenericDescPtr->strides[0];

        for(int i = 0; i < src1NDim; i++)
            srcPtrTemp1 += src1begin[i] * srcPtr1GenericDescPtr->strides[i + 1];

        for(int i = 0; i < src2NDim; i++)
            srcPtrTemp2 += src2begin[i] * srcPtr2GenericDescPtr->strides[i + 1];

        T *dstPtrTemp = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        Rpp32u testOffset = RPPT_MAX_DIMS - dstDim;

        Rpp32u *length = dstBroadcastDims + testOffset;
        Rpp32u *src1length = src1BroadcastDims + testOffset;
        Rpp32u *src2length = src2BroadcastDims + testOffset;

        Rpp32u *src1BcastStrides = src1BroadcastStrides + testOffset;
        Rpp32u *src2BcastStrides = src2BroadcastStrides + testOffset;
        Rpp32u *dstBcastStrides =  dstBroadcastStrides + testOffset;

        Rpp32u alignMask = vectorIncrement - 1;

        if (dstDim == 1)
        {
            Rpp32u alignedLength = length[0] & ~alignMask;
            Rpp32u src1shape = src1length[0];
            Rpp32u src2shape = src2length[0];
            Rpp32u vectorLoopCount = 0;
            if (src1shape == 1)
            {
#if __AVX2__
                __m256i p1 = simd_set1_val(srcPtrTemp1[0]);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256i p2 = _mm256_loadu_si256((const __m256i *)srcPtrTemp2);
                    simd_op(p2, p1);
                    _mm256_storeu_si256((__m256i *)dstPtrTemp, p2);    // simd stores
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
                __m256i p2 = simd_set1_val(srcPtrTemp2[0]);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256i p1 = _mm256_loadu_si256((const __m256i *)srcPtrTemp1);
                    simd_op(p1, p2);
                    _mm256_storeu_si256((__m256i *)dstPtrTemp, p1);    // simd stores
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
                    __m256i p1 = _mm256_loadu_si256((const __m256i *)srcPtrTemp1);    // simd loads
                    __m256i p2 = _mm256_loadu_si256((const __m256i *)srcPtrTemp2);    // simd loads
                    simd_op(p1, p2);
                    _mm256_storeu_si256((__m256i *)dstPtrTemp, p1);    // simd stores
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
                    __m256i p1 = simd_set1_val(srcPtrElem1[0]);
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256i p2 = _mm256_loadu_si256((const __m256i *)srcPtrElem2);  // simd loads
                        simd_op(p2, p1);
                        _mm256_storeu_si256((__m256i *)dstPtrElem, p2);    // simd stores
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
                    __m256i p2 = simd_set1_val(srcPtrElem2[0]);
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256i p1 = _mm256_loadu_si256((const __m256i *)srcPtrElem1);    // simd loads
                        simd_op(p1, p2);
                        _mm256_storeu_si256((__m256i *)dstPtrElem, p1);    // simd stores
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
                        __m256i p1 = _mm256_loadu_si256((const __m256i *)srcPtrElem1);    // simd loads
                        __m256i p2 = _mm256_loadu_si256((const __m256i *)srcPtrElem2);    // simd loads
                        simd_op(p1, p2);
                        _mm256_storeu_si256((__m256i *)dstPtrElem, p1);    // simd stores
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
                        __m256i p1 = simd_set1_val(srcPtrElem1[0]);
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                        {
                            __m256i p2 = _mm256_loadu_si256((const __m256i *)srcPtrElem2);    // simd loads
                            simd_op(p2, p1);
                            _mm256_storeu_si256((__m256i *)dstPtrElem, p2);    // simd stores
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
                        __m256i p2 = simd_set1_val(srcPtrElem2[0]);
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                        {
                            __m256i p1 = _mm256_loadu_si256((const __m256i *)srcPtrElem1);    // simd loads
                            simd_op(p1, p2);
                            _mm256_storeu_si256((__m256i *)dstPtrElem, p1);    // simd stores
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
                            __m256i p1 = _mm256_loadu_si256((const __m256i *)srcPtrElem1);    // simd loads
                            __m256i p2 = _mm256_loadu_si256((const __m256i *)srcPtrElem2);    // simd loads
                            simd_op(p1, p2);
                            _mm256_storeu_si256((__m256i *)dstPtrElem, p1);    // simd stores
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
        else {
            tensor_binary_op_recursive(srcPtrTemp1, srcPtrTemp2, src1BcastStrides, src2BcastStrides, dstPtrTemp, dstBcastStrides, length, dstDim, op);
        }
    }

    return RPP_SUCCESS;
}

template<typename T>
RppStatus tensor_binary_bitwise_op_dispatch_host_tensor(T *srcPtr1,
                                                        T *srcPtr2,
                                                        RpptGenericDescPtr srcPtr1GenericDescPtr,
                                                        RpptGenericDescPtr srcPtr2GenericDescPtr,
                                                        T *dstPtr,
                                                        RpptGenericDescPtr dstGenericDescPtr,
                                                        RpptBitwiseOp tensorOp,
                                                        Rpp32u *srcPtr1roiTensor,
                                                        Rpp32u *srcPtr2roiTensor,
                                                        rpp::Handle& handle) {
    int vectorIncrement = 32;
    if((srcPtr1GenericDescPtr->dataType == RpptDataType::U16) || (srcPtr1GenericDescPtr->dataType == RpptDataType::I16))
        vectorIncrement = 16;
    else if((srcPtr1GenericDescPtr->dataType == RpptDataType::U32) || (srcPtr1GenericDescPtr->dataType == RpptDataType::I32))
        vectorIncrement = 8;

    switch(tensorOp) {
        case RPP_TENSOR_OP_AND:
            tensor_binary_bitwise_op_host_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, and_op<T>, simd_and_si256, vectorIncrement, srcPtr1roiTensor, srcPtr2roiTensor, handle);
            break;
        case RPP_TENSOR_OP_OR:
            tensor_binary_bitwise_op_host_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, or_op<T>, simd_or_si256, vectorIncrement, srcPtr1roiTensor, srcPtr2roiTensor, handle);
            break;
        case RPP_TENSOR_OP_XOR:
            tensor_binary_bitwise_op_host_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, xor_op<T>, simd_xor_si256, vectorIncrement, srcPtr1roiTensor, srcPtr2roiTensor, handle);
            break;
    }

    return RPP_SUCCESS;
}

template RppStatus tensor_binary_bitwise_op_dispatch_host_tensor<Rpp8u>(Rpp8u*,
                                                                        Rpp8u*,
                                                                        RpptGenericDescPtr,
                                                                        RpptGenericDescPtr,
                                                                        Rpp8u*,
                                                                        RpptGenericDescPtr,
                                                                        RpptBitwiseOp,
                                                                        Rpp32u*,
                                                                        Rpp32u*,
                                                                        rpp::Handle&);

template RppStatus tensor_binary_bitwise_op_dispatch_host_tensor<Rpp16u>(Rpp16u*,
                                                                         Rpp16u*,
                                                                         RpptGenericDescPtr,
                                                                         RpptGenericDescPtr,
                                                                         Rpp16u*,
                                                                         RpptGenericDescPtr,
                                                                         RpptBitwiseOp,
                                                                         Rpp32u*,
                                                                         Rpp32u*,
                                                                         rpp::Handle&);

template RppStatus tensor_binary_bitwise_op_dispatch_host_tensor<Rpp32u>(Rpp32u*,
                                                                         Rpp32u*,
                                                                         RpptGenericDescPtr,
                                                                         RpptGenericDescPtr,
                                                                         Rpp32u*,
                                                                         RpptGenericDescPtr,
                                                                         RpptBitwiseOp,
                                                                         Rpp32u*,
                                                                         Rpp32u*,
                                                                         rpp::Handle&);
