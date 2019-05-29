#ifndef RPPI_ARITHMATIC_AND_LOGICAL_FUNCTIONS_H
#define RPPI_ARITHMATIC_AND_LOGICAL_FUNCTIONS_H
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

// Dummy
RppStatus
Rppi_Add_Constant_8u_pkd1(const RppPtr_t srcPtr, int rSrcStep,
                        RppPtr_t dstPtr, int rDstStep,
                  const Rpp8u rConstant, RppiSize oSizeROI, int rScaleFactor);

RppStatus
Rppi_Add_Constant_8u_pln1(const RppPtr_t srcPtr, int rSrcStep,
                        RppPtr_t dstPtr, int rDstStep,
                  const Rpp8u rConstant, RppiSize oSizeROI, int rScaleFactor);

RppStatus
Rppi_Add_Constant_8u_pkd3(const RppPtr_t srcPtr, int rSrcStep,
                        RppPtr_t dstPtr, int rDstStep,
                  const Rpp8u rConstant, RppiSize oSizeROI, int rScaleFactor);

RppStatus
Rppi_Add_Constant_8u_pln3(const RppPtr_t srcPtr, int rSrcStep,
                        RppPtr_t dstPtr, int rDstStep,
                  const Rpp8u rConstant, RppiSize oSizeROI, int rScaleFactor);
#ifdef __cplusplus
}
#endif
#endif /* RPPI_ARITHMATIC_AND_LOGICAL_FUNCTIONS_H */
