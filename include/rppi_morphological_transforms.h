#ifndef RPPI_MORPHOLOGICAL_TRANSFORMS
#define RPPI_MORPHOLOGICAL_TRANSFORMS
 
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

//--------------------------------------
//Dilate
//--------------------------------------
RppStatus
rppi_dilate_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize, RppHandle_t rppHandle);

RppStatus
rppi_dilate_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize, RppHandle_t rppHandle);

RppStatus
rppi_dilate_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize, RppHandle_t rppHandle);

//--------------------------------------
//Erode
//--------------------------------------
RppStatus
rppi_erode_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize, RppHandle_t rppHandle);

RppStatus
rppi_erode_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize, RppHandle_t rppHandle);

RppStatus
rppi_erode_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize, RppHandle_t rppHandle);
 
#ifdef __cplusplus
}
#endif
#endif