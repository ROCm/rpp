#ifndef RPPI_ARITHMETIC_AND_LOGICAL_FUNCTIONS_H
#define RPPI_ARITHMETIC_AND_LOGICAL_FUNCTIONS_H
#include "rppdefs.h"

#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

// --------------------
// Bitwise AND
// --------------------

// Gpu function declarations

RppStatus
rppi_bitwise_AND_u8_pln1_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle );

RppStatus
rppi_bitwise_AND_u8_pln3_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle );

RppStatus
rppi_bitwise_AND_u8_pkd3_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle );

// Host function declarations.

RppStatus
rppi_bitwise_AND_u8_pln1_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_bitwise_AND_u8_pln3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_bitwise_AND_u8_pkd3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                                 RppiSize srcSize, RppPtr_t dstPtr);



// --------------------
// Bitwise NOT
// --------------------

// Gpu function declarations

RppStatus
rppi_bitwise_NOT_u8_pln1_gpu( RppPtr_t srcPtr1,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle );

RppStatus
rppi_bitwise_NOT_u8_pln3_gpu( RppPtr_t srcPtr1,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle );

RppStatus
rppi_bitwise_NOT_u8_pkd3_gpu( RppPtr_t srcPtr1,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle );

// Host function declarations.
RppStatus
rppi_bitwise_NOT_u8_pln1_host( RppPtr_t srcPtr1,
                              RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_bitwise_NOT_u8_pln3_host( RppPtr_t srcPtr1,
                              RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_bitwise_NOT_u8_pkd3_host( RppPtr_t srcPtr1,
                              RppiSize srcSize, RppPtr_t dstPtr);

// --------------------
// Exclusive OR
// --------------------

// Gpu function declarations

RppStatus
rppi_exclusive_OR_u8_pln1_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle );

RppStatus
rppi_exclusive_OR_u8_pln3_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle );

RppStatus
rppi_exclusive_OR_u8_pkd3_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle );

// Host function declarations.

RppStatus
rppi_exclusive_OR_u8_pln1_host( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_exclusive_OR_u8_pln3_host( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_exclusive_OR_u8_pkd3_host( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr);

// --------------------
// Inclusive OR
// --------------------

// Gpu function declarations

RppStatus
rppi_inclusive_OR_u8_pln1_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle );

RppStatus
rppi_inclusive_OR_u8_pln3_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle );

RppStatus
rppi_inclusive_OR_u8_pkd3_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle );

// Host function declarations.
RppStatus
rppi_inclusive_OR_u8_pln1_host( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_inclusive_OR_u8_pln3_host( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_inclusive_OR_u8_pkd3_host( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr);

// --------------------
// Addition
// --------------------

// Gpu function declarations

RppStatus
rppi_add_u8_pln1_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle );

RppStatus
rppi_add_u8_pln3_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle );

RppStatus
rppi_add_u8_pkd3_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle );

// Host function declarations.

RppStatus
rppi_add_u8_pln1_host( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_add_u8_pln3_host( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_add_u8_pkd3_host( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr);

// --------------------
// Subtraction
// --------------------

// Gpu function declarations

RppStatus
rppi_subtract_u8_pln1_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle );

RppStatus
rppi_subtract_u8_pln3_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle );

RppStatus
rppi_subtract_u8_pkd3_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle );

// Host function declarations.
RppStatus
rppi_subtract_u8_pln1_host( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_subtract_u8_pln3_host( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_subtract_u8_pkd3_host( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr);
// --------------------
// Absolute difference
// --------------------

// Gpu function declarations

RppStatus
rppi_absolute_difference_u8_pln1_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle );

RppStatus
rppi_absolute_difference_u8_pln3_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle );

RppStatus
rppi_absolute_difference_u8_pkd3_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, RppPtr_t dstPtr,
                              RppHandle_t rppHandle );

// Host function declarations.

RppStatus
rppi_absolute_difference_u8_pln1_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, 
                                      RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_absolute_difference_u8_pln3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                                      RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_absolute_difference_u8_pkd3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                                       RppiSize srcSize, RppPtr_t dstPtr);


// --------------------
// Bilateral filter
// --------------------

// Gpu function declarations

RppStatus
rppi_bilateral_filter_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize,
                                  RppPtr_t dstPtr, Rpp32u filterSize,
                                  Rpp64f sigmaI, Rpp64f sigmaS,
                                  RppHandle_t rppHandle);

RppStatus
rppi_bilateral_filter_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize,
                                  RppPtr_t dstPtr, Rpp32u filterSize,
                                  Rpp64f sigmaI, Rpp64f sigmaS,
                                  RppHandle_t rppHandle);

RppStatus
rppi_bilateral_filter_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize,
                                  RppPtr_t dstPtr, Rpp32u filterSize,
                                  Rpp64f sigmaI, Rpp64f sigmaS,
                                  RppHandle_t rppHandle);

// Host function declarations.
RppStatus
rppi_bilateral_filter_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize,
                                  RppPtr_t dstPtr, Rpp32u filterSize,
                                  Rpp64f sigmaI, Rpp64f sigmaS);

RppStatus
rppi_bilateral_filter_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize,
                                  RppPtr_t dstPtr, Rpp32u filterSize,
                                  Rpp64f sigmaI, Rpp64f sigmaS);

RppStatus
rppi_bilateral_filter_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize,
                                  RppPtr_t dstPtr, Rpp32u filterSize,
                                  Rpp64f sigmaI, Rpp64f sigmaS);
// --------------------
// Accumulation
// --------------------

// Gpu function declarations

RppStatus
rppi_accumulate_u8_pln1_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize,
                              RppHandle_t rppHandle );

RppStatus
rppi_accumulate_u8_pln3_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize,
                              RppHandle_t rppHandle );

RppStatus
rppi_accumulate_u8_pkd3_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize,
                              RppHandle_t rppHandle );

// Host function declarations.
RppStatus
rppi_accumulate_u8_pln1_host( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize);

RppStatus
rppi_accumulate_u8_pln3_host( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize);

RppStatus
rppi_accumulate_u8_pkd3_host( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize);

// --------------------
// Accumulate Weighted
// --------------------

//Parameters : alpha should be [0 <= alpha <=1]
RppStatus
rppi_accumulate_weighted_u8_pln1_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, Rpp64f alpha,
                              RppHandle_t rppHandle );

RppStatus
rppi_accumulate_weighted_u8_pln3_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, Rpp64f alpha,
                              RppHandle_t rppHandle );

RppStatus
rppi_accumulate_weighted_u8_pkd3_gpu( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, Rpp64f alpha,
                              RppHandle_t rppHandle );


// Host function declarations.
RppStatus
rppi_accumulate_weighted_u8_pln1_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                                       RppiSize srcSize, Rpp64f alpha);

RppStatus
rppi_accumulate_weighted_u8_pln3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, 
                                        RppiSize srcSize, Rpp64f alpha);

RppStatus
rppi_accumulate_weighted_u8_pkd3_host( RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                              RppiSize srcSize, Rpp64f alpha);


// --------------------
// Box Filter
// --------------------

// Gpu function declarations

RppStatus
rppi_box_filter_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                            RppHandle_t rppHandle);

RppStatus
rppi_box_filter_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                            RppHandle_t rppHandle);

RppStatus
rppi_box_filter_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                            RppHandle_t rppHandle);

#ifdef __cplusplus
}
#endif
#endif /* RPPI_ARITHMATIC_AND_LOGICAL_FUNCTIONS_H */
