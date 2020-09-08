#ifndef RPPI_ADVANCED_AUGMENTATIONS_H
#define RPPI_ADVANCED_AUGMENTATIONS_H

#include "rppdefs.h"
#include "rpp.h"
#ifdef __cplusplus
extern "C"
{
#endif // cpusplus

/// Non Linear Blend.
/// \param srcPtr1
/// \param srcPtr2
/// \param srcSize - Array of it
/// \param dstPtr
/// \param alpha - non linear blending parameter
/// \param outputForamtToggle - NCHW to NHWC toggling and Vice Versa
/// \param rppHandle
/// \return
RppStatus
rppi_non_linear_blend_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus
rppi_non_linear_blend_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus
rppi_non_linear_blend_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus
rppi_non_linear_blend_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);

#ifdef __cplusplus
}
#endif
#endif // RPPI_ADVANCED_AUGMENTATIONS