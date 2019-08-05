#ifndef RPPI_SUPPORT_FUNCTIONS
#define RPPI_SUPPORT_FUNCTIONS

#ifdef __cplusplus
extern "C" {
#endif
#include "rppdefs.h"

RppStatus warp_affine_output_size(RppiSize srcSize, RppiSize *dstSizePtr,
                                       float* affine);

RppStatus warp_affine_output_offset(RppiSize srcSize, RppiPoint *offset,
                                       float* affine);

RppStatus rotate_output_size(RppiSize srcSize, RppiSize *dstSizePtr,
                                  Rpp32f angleDeg);

RppStatus rotate_output_offset(RppiSize srcSize, RppiPoint *offset,
                                  Rpp32f angleDeg);

// ----------------------------------------
// GPU data_object_copy functions declaration 
// ----------------------------------------

RppStatus
rppi_data_object_copy_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_data_object_copy_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_data_object_copy_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

#ifdef __cplusplus
}
#endif

#endif /* RPPI_SUPPORT_FUNCTIONS_H */