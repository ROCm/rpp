#ifndef RPPI_LOGICAL_OPERATIONS.H
#define RPPI_LOGICAL_OPERATIONS.H
 
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif


// ----------------------------------------
// Host bitwise_AND functions  declaration 
// ----------------------------------------


RppStatus
rppi_bitwise_AND_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_bitwise_AND_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_bitwise_AND_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

// ----------------------------------------
// Host bitwise_NOT functions  declaration 
// ----------------------------------------


RppStatus
rppi_bitwise_NOT_u8_pln1_host(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_bitwise_NOT_u8_pln3_host(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_bitwise_NOT_u8_pkd3_host(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t dstPtr);

// ----------------------------------------
// Host exclusive_OR functions  declaration 
// ----------------------------------------


RppStatus
rppi_exclusive_OR_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_exclusive_OR_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_exclusive_OR_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

// ----------------------------------------
// Host inclusive_OR functions  declaration 
// ----------------------------------------


RppStatus
rppi_inclusive_OR_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_inclusive_OR_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_inclusive_OR_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

// ----------------------------------------
// GPU bitwise_AND functions  declaration 
// ----------------------------------------


RppStatus
rppi_bitwise_AND_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_bitwise_AND_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_bitwise_AND_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU bitwise_NOT functions  declaration 
// ----------------------------------------


RppStatus
rppi_bitwise_NOT_u8_pln1_gpu(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_bitwise_NOT_u8_pln3_gpu(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_bitwise_NOT_u8_pkd3_gpu(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU exclusive_OR functions  declaration 
// ----------------------------------------


RppStatus
rppi_exclusive_OR_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_exclusive_OR_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_exclusive_OR_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU inclusive_OR functions  declaration 
// ----------------------------------------


RppStatus
rppi_inclusive_OR_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_inclusive_OR_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_inclusive_OR_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;
 
#ifdef __cplusplus
}
#endif
#endif
