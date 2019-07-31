#ifndef RPPI_MORPHOLOGICAL_TRANSFORMS
#define RPPI_MORPHOLOGICAL_TRANSFORMS
 
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif


// ----------------------------------------
// Host erode functions declaration 
// ----------------------------------------

RppStatus
rppi_erode_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32u kernelSize);

RppStatus
rppi_erode_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32u kernelSize);

RppStatus
rppi_erode_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32u kernelSize);

// ----------------------------------------
// Host dilate functions declaration 
// ----------------------------------------

RppStatus
rppi_dilate_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32u kernelSize);

RppStatus
rppi_dilate_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32u kernelSize);

RppStatus
rppi_dilate_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32u kernelSize);
 
#ifdef __cplusplus
}
#endif
#endif
