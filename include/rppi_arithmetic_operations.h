#ifndef RPPI_ARITHMETIC_OPERATIONS.H
#define RPPI_ARITHMETIC_OPERATIONS.H
 
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif


// ----------------------------------------
// Host accumulate_weighted functions  declaration 
// ----------------------------------------


RppStatus
rppi_accumulate_weighted_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,Rpp64f alpha);

RppStatus
rppi_accumulate_weighted_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,Rpp64f alpha);

RppStatus
rppi_accumulate_weighted_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,Rpp64f alpha);

// ----------------------------------------
// Host absolute_difference functions  declaration 
// ----------------------------------------


RppStatus
rppi_absolute_difference_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_absolute_difference_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_absolute_difference_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

// ----------------------------------------
// Host add functions  declaration 
// ----------------------------------------


RppStatus
rppi_add_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_add_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_add_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

// ----------------------------------------
// Host subtract functions  declaration 
// ----------------------------------------


RppStatus
rppi_subtract_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_subtract_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_subtract_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

// ----------------------------------------
// Host accumulate functions  declaration 
// ----------------------------------------


RppStatus
rppi_accumulate_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize);

RppStatus
rppi_accumulate_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize);

RppStatus
rppi_accumulate_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize);

// ----------------------------------------
// GPU accumulate_weighted functions  declaration 
// ----------------------------------------


RppStatus
rppi_accumulate_weighted_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,Rpp64f alpha, RppHandle_t rppHandle) ;

RppStatus
rppi_accumulate_weighted_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,Rpp64f alpha, RppHandle_t rppHandle) ;

RppStatus
rppi_accumulate_weighted_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,Rpp64f alpha, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU absolute_difference functions  declaration 
// ----------------------------------------


RppStatus
rppi_absolute_difference_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_absolute_difference_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_absolute_difference_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU add functions  declaration 
// ----------------------------------------


RppStatus
rppi_add_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_add_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_add_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU subtract functions  declaration 
// ----------------------------------------


RppStatus
rppi_subtract_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_subtract_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_subtract_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU accumulate functions  declaration 
// ----------------------------------------


RppStatus
rppi_accumulate_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize, RppHandle_t rppHandle) ;

RppStatus
rppi_accumulate_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize, RppHandle_t rppHandle) ;

RppStatus
rppi_accumulate_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize, RppHandle_t rppHandle) ;
 
#ifdef __cplusplus
}
#endif
#endif
