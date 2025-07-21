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
            dst += *(dstStrides + 1);
            src1 += *(src1Strides + 1);
            src2 += *(src2Strides + 1);
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

    checkEqualBatchSize(srcPtr1GenericDescPtr, srcPtr2GenericDescPtr);
    BroadcastDstShape(srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstGenericDescPtr);
    RpptGenericDesc src1BroadcastDesc, src2BroadcastDesc, dstBroadcastDesc;
    RpptGenericDescPtr src1BroadcastDescPtr, src2BroadcastDescPtr, dstBroadcastDescPtr;
    src1BroadcastDescPtr = &src1BroadcastDesc;
    src2BroadcastDescPtr = &src2BroadcastDesc;
    dstBroadcastDescPtr = &dstBroadcastDesc;
    src1BroadcastDesc = *srcPtr1GenericDescPtr;
    src2BroadcastDesc = *srcPtr2GenericDescPtr;
    dstBroadcastDesc = *dstGenericDescPtr;
    GroupShapes(src1BroadcastDescPtr, src2BroadcastDescPtr, dstBroadcastDescPtr);
    StridesForBroadcasting(src1BroadcastDescPtr, dstBroadcastDescPtr);
    StridesForBroadcasting(src2BroadcastDescPtr, dstBroadcastDescPtr);

    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u src1NDim = srcPtr1GenericDescPtr->numDims - 1;
    Rpp32u src2NDim = srcPtr2GenericDescPtr->numDims - 1;
    Rpp32u broadcastNDim = dstBroadcastDescPtr->numDims - 1; // Omitting batchSize here to get tensor dimension.
    Rpp32u batchSize = dstBroadcastDescPtr->dims[0];

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        Rpp32u *src1roi = srcPtr1roiTensor + batchCount * src1NDim * 2;
        Rpp32u *src1begin = src1roi;

        Rpp32u *src2roi = srcPtr2roiTensor + batchCount * src2NDim * 2;
        Rpp32u *src2begin = src2roi;

        T *srcPtrTemp1 = srcPtr1 + batchCount * srcPtr1GenericDescPtr->strides[0];
        T *srcPtrTemp2 = srcPtr2 + batchCount * srcPtr2GenericDescPtr->strides[0];

        for(int i = 0; i < src1NDim; i++)
            srcPtrTemp1 += src1begin[i] * srcPtr1GenericDescPtr->strides[i + 1];

        for(int i = 0; i < src2NDim; i++)
            srcPtrTemp2 += src2begin[i] * srcPtr2GenericDescPtr->strides[i + 1];

        T *dstPtrTemp = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        Rpp32u *length = dstBroadcastDescPtr->dims + 1;
        Rpp32u *src1length = src1BroadcastDescPtr->dims + 1;
        Rpp32u *src2length = src2BroadcastDescPtr->dims + 1;

        Rpp32u alignMask = vectorIncrement - 1;

        if (broadcastNDim == 1)
        {
            Rpp32u alignedLength = length[0] & ~alignMask;
            Rpp32u src1shape = src1length[0];
            Rpp32u src2shape = src2length[0];
            Rpp32u vectorLoopCount = 0;
            if (src1shape == 1)
            {
                printf("Source 1 shape and broadcastNDim are %d %d\n", 1, 1);
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
                printf("Source 2 shape and broadcastNDim are %d %d\n", 1, 1);
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
                printf("broadcastNDim are %d\n", 1);
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
        else if (broadcastNDim == 2)
        {
            Rpp32u alignedLength = length[1] & ~alignMask;
            Rpp32u src1shape = src1length[1];
            Rpp32u src2shape = src2length[1];
            if(src1shape == 1)
            {
                printf("Source 1 shape and broadcastNDim are %d %d\n", 1, 2);
                for (int i = 0; i < length[0]; i++)
                {
                    T *srcPtrTest1 = srcPtrTemp1;
                    T *srcPtrTest2 = srcPtrTemp2;
                    T *dstPtrTest = dstPtrTemp;

                    int vectorLoopCount = 0;
#if __AVX2__
                    __m256i p1 = simd_set1_val(srcPtrTest1[0]);
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256i p2 = _mm256_loadu_si256((const __m256i *)srcPtrTest2);  // simd loads
                        simd_op(p2, p1);
                        _mm256_storeu_si256((__m256i *)dstPtrTest, p2);    // simd stores
                        srcPtrTest2 += vectorIncrement;
                        dstPtrTest += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[1]; vectorLoopCount++)
                    {
                        op(dstPtrTest, srcPtrTest1, srcPtrTest2);
                        srcPtrTest2++;
                        dstPtrTest++;
                    }
                    srcPtrTemp1 += src1BroadcastDescPtr->strides[1];
                    srcPtrTemp2 += src2BroadcastDescPtr->strides[1];
                    dstPtrTemp += dstBroadcastDescPtr->strides[1];
                }
            }
            else if (src2shape == 1)
            {
                printf("Source 2 shape and broadcastNDim are %d %d\n", 1, 2);
                for (int i = 0; i < length[0]; i++)
                {
                    T *srcPtrTest1 = srcPtrTemp1;
                    T *srcPtrTest2 = srcPtrTemp2;
                    T *dstPtrTest = dstPtrTemp;

                    int vectorLoopCount = 0;
                    __m256i p2 = simd_set1_val(srcPtrTest2[0]);
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256i p1 = _mm256_loadu_si256((const __m256i *)srcPtrTest1);    // simd loads
                        simd_op(p1, p2);
                        _mm256_storeu_si256((__m256i *)dstPtrTest, p1);    // simd stores
                        srcPtrTest1 += vectorIncrement;
                        dstPtrTest += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[1]; vectorLoopCount++)
                    {
                        op(dstPtrTest, srcPtrTest1, srcPtrTest2);
                        srcPtrTest1++;
                        dstPtrTest++;
                    }
                    srcPtrTemp1 += src1BroadcastDescPtr->strides[1];
                    srcPtrTemp2 += src2BroadcastDescPtr->strides[1];
                    dstPtrTemp += dstBroadcastDescPtr->strides[1];
                }
            }
            else
            {
                printf("broadcastNDim are %d\n", 2);
                for (int i = 0; i < length[0]; i++)
                {
                    T *srcPtrTest1 = srcPtrTemp1;
                    T *srcPtrTest2 = srcPtrTemp2;
                    T *dstPtrTest = dstPtrTemp;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256i p1 = _mm256_loadu_si256((const __m256i *)srcPtrTest1);    // simd loads
                        __m256i p2 = _mm256_loadu_si256((const __m256i *)srcPtrTest2);    // simd loads
                        simd_op(p1, p2);
                        _mm256_storeu_si256((__m256i *)dstPtrTest, p1);    // simd stores
                        srcPtrTest1 += vectorIncrement;
                        srcPtrTest2 += vectorIncrement;
                        dstPtrTest += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[1]; vectorLoopCount++)
                    {
                        op(dstPtrTest, srcPtrTest1, srcPtrTest2);
                        srcPtrTest1++;
                        srcPtrTest2++;
                        dstPtrTest++;
                    }
                    srcPtrTemp1 += src1BroadcastDescPtr->strides[1];
                    srcPtrTemp2 += src2BroadcastDescPtr->strides[1];
                    dstPtrTemp += dstBroadcastDescPtr->strides[1];
                }
            }
        }
        else if (broadcastNDim == 3)
        {
            Rpp32u alignedLength = length[2] & ~alignMask;
            Rpp32u src1shape = src1length[2];
            Rpp32u src2shape = src2length[2];
            if(src1shape == 1)
            {
                printf("Source 1 shape and broadcastNDim are %d %d\n", 1, 3);
                for (int i = 0; i < length[0]; i++)
                {
                    T *srcPtrTest1 = srcPtrTemp1;
                    T *srcPtrTest2 = srcPtrTemp2;
                    T *dstPtrTest = dstPtrTemp;

                    for (int j = 0; j < length[1]; j++)
                    {
                        T *srcPtrNew1 = srcPtrTest1;
                        T *srcPtrNew2 = srcPtrTest2;
                        T *dstPtrNew = dstPtrTest;

                        int vectorLoopCount = 0;

                        __m256i p1 = simd_set1_val(srcPtrNew1[0]);
#if __AVX2__
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                        {
                            __m256i p2 = _mm256_loadu_si256((const __m256i *)srcPtrNew2);    // simd loads
                            simd_op(p2, p1);
                            _mm256_storeu_si256((__m256i *)dstPtrNew, p2);    // simd stores
                            srcPtrNew2 += vectorIncrement;
                            dstPtrNew += vectorIncrement;
                        }
#endif
                        for (; vectorLoopCount < length[2]; vectorLoopCount++)
                        {
                            op(dstPtrNew, srcPtrNew1, srcPtrNew2);
                            srcPtrNew2++;
                            dstPtrNew++;
                        }

                        srcPtrTest1 += src1BroadcastDescPtr->strides[2];
                        srcPtrTest2 += src2BroadcastDescPtr->strides[2];
                        dstPtrTest += dstBroadcastDescPtr->strides[2];
                    }

                    srcPtrTemp1 += src1BroadcastDescPtr->strides[1];
                    srcPtrTemp2 += src2BroadcastDescPtr->strides[1];
                    dstPtrTemp += dstBroadcastDescPtr->strides[1];
                }
            }
            else if (src2shape == 1)
            {
                printf("Source 2 shape and broadcastNDim are %d %d\n", 1, 3);
                for (int i = 0; i < length[0]; i++)
                {
                    T *srcPtrTest1 = srcPtrTemp1;
                    T *srcPtrTest2 = srcPtrTemp2;
                    T *dstPtrTest = dstPtrTemp;

                    for (int j = 0; j < length[1]; j++)
                    {
                        T *srcPtrNew1 = srcPtrTest1;
                        T *srcPtrNew2 = srcPtrTest2;
                        T *dstPtrNew = dstPtrTest;

                        int vectorLoopCount = 0;
                        __m256i p2 = simd_set1_val(srcPtrNew2[0]);
#if __AVX2__
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                        {
                            __m256i p1 = _mm256_loadu_si256((const __m256i *)srcPtrNew1);    // simd loads
                            simd_op(p1, p2);
                            _mm256_storeu_si256((__m256i *)dstPtrNew, p1);    // simd stores
                            srcPtrNew1 += vectorIncrement;
                            dstPtrNew += vectorIncrement;
                        }
#endif
                        for (; vectorLoopCount < length[2]; vectorLoopCount++)
                        {
                            op(dstPtrNew, srcPtrNew1, srcPtrNew2);
                            srcPtrNew1++;
                            dstPtrNew++;
                        }

                        srcPtrTest1 += src1BroadcastDescPtr->strides[2];
                        srcPtrTest2 += src2BroadcastDescPtr->strides[2];
                        dstPtrTest += dstBroadcastDescPtr->strides[2];
                    }

                    srcPtrTemp1 += src1BroadcastDescPtr->strides[1];
                    srcPtrTemp2 += src2BroadcastDescPtr->strides[1];
                    dstPtrTemp += dstBroadcastDescPtr->strides[1];
                }
            }
            else
            {
                printf("broadcastNDim is %d\n", 3);
                for (int i = 0; i < length[0]; i++)
                {
                    T *srcPtrTest1 = srcPtrTemp1;
                    T *srcPtrTest2 = srcPtrTemp2;
                    T *dstPtrTest = dstPtrTemp;

                    for (int j = 0; j < length[1]; j++)
                    {
                        T *srcPtrNew1 = srcPtrTest1;
                        T *srcPtrNew2 = srcPtrTest2;
                        T *dstPtrNew = dstPtrTest;

                        int vectorLoopCount = 0;
#if __AVX2__
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                        {
                            __m256i p1 = _mm256_loadu_si256((const __m256i *)srcPtrNew1);    // simd loads
                            __m256i p2 = _mm256_loadu_si256((const __m256i *)srcPtrNew2);    // simd loads
                            simd_op(p1, p2);
                            _mm256_storeu_si256((__m256i *)dstPtrNew, p1);    // simd stores
                            srcPtrNew1 += vectorIncrement;
                            srcPtrNew2 += vectorIncrement;
                            dstPtrNew += vectorIncrement;
                        }
#endif
                        for (; vectorLoopCount < length[2]; vectorLoopCount++)
                        {
                            op(dstPtrNew, srcPtrNew1, srcPtrNew2);
                            srcPtrNew1++;
                            srcPtrNew2++;
                            dstPtrNew++;
                        }

                        srcPtrTest1 += src1BroadcastDescPtr->strides[2];
                        srcPtrTest2 += src2BroadcastDescPtr->strides[2];
                        dstPtrTest += dstBroadcastDescPtr->strides[2];
                    }

                    srcPtrTemp1 += src1BroadcastDescPtr->strides[1];
                    srcPtrTemp2 += src2BroadcastDescPtr->strides[1];
                    dstPtrTemp += dstBroadcastDescPtr->strides[1];
                }
            }
        }
        else {
            printf("broadcastNDim is %d\n", 4);
            tensor_binary_op_recursive(srcPtrTemp1, srcPtrTemp2, src1BroadcastDescPtr->strides, src2BroadcastDescPtr->strides, dstPtrTemp, dstBroadcastDescPtr->strides, length, broadcastNDim, op);
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
