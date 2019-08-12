#ifndef RPPI_COMPUTER_VISION
#define RPPI_COMPUTER_VISION
 
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif


//--------------------------------------
//LBP
//--------------------------------------
RppStatus
rppi_local_binary_pattern_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppHandle_t rppHandle);

RppStatus
rppi_local_binary_pattern_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppHandle_t rppHandle);

RppStatus
rppi_local_binary_pattern_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppHandle_t rppHandle);


// ----------------------------------------
// GPU data_object_copy functions declaration 
// ----------------------------------------

RppStatus
rppi_data_object_copy_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_data_object_copy_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_data_object_copy_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU gaussian_image_pyramid functions declaration 
// ----------------------------------------

RppStatus
rppi_gaussian_image_pyramid_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppHandle_t rppHandle);

RppStatus
rppi_gaussian_image_pyramid_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppHandle_t rppHandle);

RppStatus
rppi_gaussian_image_pyramid_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppHandle_t rppHandle);

// ----------------------------------------
// GPU laplacian_image_pyramid functions declaration 
// ----------------------------------------

RppStatus
rppi_laplacian_image_pyramid_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppHandle_t rppHandle);

RppStatus
rppi_laplacian_image_pyramid_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppHandle_t rppHandle);

RppStatus
rppi_laplacian_image_pyramid_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppHandle_t rppHandle);

#ifdef __cplusplus
}
#endif
#endif