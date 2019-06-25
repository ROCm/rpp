#ifndef RPPI_COMPUTER_VISION_H
#define RPPI_COMPUTER_VISION_H
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif


// BOX FILTER ----------------------------------------------------------
// CPU---------
RppStatus
rppi_box_filter_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_box_filter_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_box_filter_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);
//-------CPU

//GPU----------
RppStatus
rppi_box_filter_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppHandle_t rppHandle);

RppStatus
rppi_box_filter_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppHandle_t rppHandle);

RppStatus
rppi_box_filter_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppHandle_t rppHandle);
//--------GPU
//------------------------------------------------BOX FILTER

#ifdef __cplusplus
}
#endif
#endif /* RPPI_COMPUTER_VSION */