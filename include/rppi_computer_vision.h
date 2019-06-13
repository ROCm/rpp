#ifndef RPPI_COMPUTER_VISION_H
#define RPPI_COMPUTER_VISION_H
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

RppStatus
rppi_box_filter_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppHandle_t rppHandle);

RppStatus
rppi_box_filter_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppHandle_t rppHandle);

RppStatus
rppi_box_filter_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppHandle_t rppHandle);

#endif /* RPPI_COMPUTER_VSION */