#ifndef RPPI_GEOMETRY_TRANSFORMS_H
#define RPPI_GEOMETRY_TRANSFORMS_H

#include "rppdefs.h"
#include "rpp.h"
#ifdef __cplusplus
extern "C"
{
#endif

    // ----------------------------------------
    // GPU lens_correction functions declaration
    // ----------------------------------------
    /* Does lens correction in the lens distorted images.
*param srcPtr [in/out] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] strength strength strength of lens correction needed which should be greater than 0
*param[in] zoom zoom extent to which zoom-out is needed which should be greater than 1
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
    RppStatus
    rppi_lens_correction_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);

    // ----------------------------------------
    // CPU lens_correction functions declaration
    // ----------------------------------------
    /* Does lens correction in the lens distorted images.
*param srcPtr [in/out] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] strength strength strength of lens correction needed which should be greater than 0
*param[in] zoom zoom extent to which zoom-out is needed which should be greater than 1
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
    RppStatus
    rppi_lens_correction_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_lens_correction_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *strength, Rpp32f *zoom, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle); // ----------------------------------------
    // GPU fisheye functions declaration
    // ----------------------------------------
    /* Add fish eye effect in the entire image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
    RppStatus
    rppi_fisheye_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);

    // ----------------------------------------
    // CPU fisheye functions declaration
    // ----------------------------------------
    /* Add fish eye effect in the entire image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
    RppStatus
    rppi_fisheye_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_fisheye_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);

    // ----------------------------------------
    // GPU flip functions declaration
    // ----------------------------------------
    /* Flips the image.
*param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] flipAxis flip axis and value should range beetween 0 and 2
*0 ---> horizontal flip
*1 ---> vertical flip
*2 ---> horizontal + vertical flip
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
    RppStatus
    rppi_flip_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);

    // ----------------------------------------
    // CPU flip functions declaration
    // ----------------------------------------
    /* Flips the image.
*param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] flipAxis flip axis and value should range beetween 0 and 2
*0 ---> horizontal flip
*1 ---> vertical flip
*2 ---> horizontal + vertical flip
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
    RppStatus
    rppi_flip_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_flip_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *flipAxis, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);

    // ----------------------------------------
    // GPU scale functions declaration
    // ----------------------------------------
    /* Scales the input image according to the percentage given by the user.
*param[in] srcPtr  input image
*param[in] srcSize dimensions of the images
*param[out] dstPtr output image where resized image is stored
*param[in] dstSize dimensions of the output images
*param[in] percentage percentage Percentage to which the input image needs to be scaled
*param[in] rppHandle  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
    RppStatus
    rppi_scale_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);

    // ----------------------------------------
    // CPU scale functions declaration
    // ----------------------------------------
    /* Scales the input image according to the percentage given by the user.
*param[in] srcPtr  input image
*param[in] srcSize dimensions of the images
*param[out] dstPtr output image where resized image is stored
*param[in] dstSize dimensions of the output images
*param[in] percentage percentage Percentage to which the input image needs to be scaled
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
    RppStatus
    rppi_scale_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *percentage, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *percentage, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_scale_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *percentage, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);

    // ----------------------------------------
    // GPU resize functions declaration
    // ----------------------------------------
    /* Resizes the input image to the destination dimension.
*param[in] srcPtr input image
*param[in] srcSize dimensions of the images
*param[out] dstPtr output image
*param[in] dstSize dimensions of the output images
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
    RppStatus
    rppi_resize_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                    RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFomatToggle,
                                    Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                    RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);

    // ----------------------------------------
    // CPU resize functions declaration
    // ----------------------------------------
    /* Resizes the input image to the destination dimension.
*param[in] srcPtr input image
*param[in] srcSize dimensions of the images
*param[out] dstPtr output image
*param[in] dstSize dimensions of the output images
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
    RppStatus
    rppi_resize_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);

    RppStatus
    rppi_resize_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);

    RppStatus
    rppi_resize_u8_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);

    RppStatus
    rppi_resize_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);

    RppStatus
    rppi_resize_u8_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);

    // ----------------------------------------
    // GPU rotate functions declaration
    // ----------------------------------------
    /* Rotates the input image according to the angle specified
*param[in] srcPtr input image
*param[in] srcSize dimensions of the input images
*param[out] dstPtr output image where rotated image is stored
*param[in] dstSize dimensions of the output images
*param[in] angleDeg angle for rotation
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
    RppStatus
    rppi_rotate_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                    RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg,
                                    Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                    RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);

    // ----------------------------------------
    // CPU rotate functions declaration
    // ----------------------------------------
    /* Rotates the input image according to the angle specified
*param[in] srcPtr input image
*param[in] srcSize dimensions of the input images
*param[out] dstPtr output image where rotated image is stored
*param[in] dstSize dimensions of the output images
*param[in] angleDeg angle for rotation
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
    RppStatus
    rppi_rotate_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle); // ----------------------------------------

    RppStatus
    rppi_rotate_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);

    RppStatus
    rppi_rotate_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);

    RppStatus
    rppi_rotate_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize,
                                     RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                     RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputForamtToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize,
                                     RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                     RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputForamtToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize,
                                     RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                     RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputForamtToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize,
                                     RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                     RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputForamtToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize,
                                     RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                     RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputForamtToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize,
                                     RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                     RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputForamtToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize,
                                    RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                    RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputForamtToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize,
                                    RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                    RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputForamtToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_rotate_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize,
                                    RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                    RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputForamtToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    // ----------------------------------------
    // GPU resize_crop functions declaration
    // ----------------------------------------
    /* Crops the image to the roi area and resizes to the destination size
*param[in] srcPtr input image
*param[in] srcSize dimensions of the images
*param[out] dstPtr output image
*param[in] dstSize dimensions of the output images
*param[in] xRoiBegin xRoiBegin value of roi
*param[in] yRoiBegin yRoiBegin value of roi
*param[in] xRoiEnd xRoiEnd value of roi
*param[in] yRoiEnd yRoiEnd value of roi
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
    RppStatus
    rppi_resize_crop_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin,
                                         Rpp32u *yRoiEnd, Rpp32u outputChannelToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd,
                                         Rpp32u outputChnnelToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                         RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd,
                                         Rpp32u outputChannelToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);

    // ----------------------------------------
    // CPU resize_crop functions declaration
    // ----------------------------------------
    /* Crops the image to the roi area and resizes to the destination size
*param[in] srcPtr input image
*param[in] srcSize dimensions of the images
*param[out] dstPtr output image
*param[in] dstSize dimensions of the output images
*param[in] xRoiBegin xRoiBegin value of roi
*param[in] yRoiBegin yRoiBegin value of roi
*param[in] xRoiEnd xRoiEnd value of roi
*param[in] yRoiEnd yRoiEnd value of roi
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
    RppStatus
    rppi_resize_crop_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32u xRoiBegin, Rpp32u xRoiEnd, Rpp32u yRoiBegin, Rpp32u yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);

    RppStatus
    rppi_resize_crop_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);

    RppStatus
    rppi_resize_crop_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);

    // ----------------------------------------
    // GPU warp_affine functions declaration
    // ----------------------------------------
    /* Rotates translates and sheers the input image according to the affine values.
*param[in] srcPtr input image
*param[in] srcSize dimensions of the images
*param[in] dstPtr output image
*param[in] affine affine transformation matrix
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
    RppStatus
    rppi_warp_affine_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);

    // ----------------------------------------
    // CPU warp_affine functions declaration
    // ----------------------------------------
    /* Rotates translates and sheers the input image according to the affine values.
*param[in] srcPtr input image
*param[in] srcSize dimensions of the images
*param[in] dstPtr output image
*param[in] affine affine transformation matrix
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error  
*/
    RppStatus
    rppi_warp_affine_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_affine_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);

    // ----------------------------------------
    // GPU warp_perspective functions declaration
    // ----------------------------------------
    /* Performs a perspective transform on an image.
*This kernel performs an perspective transform with a 3x3 Matrix M with this method of pixel coordinate translation:
*                // x0 = a x + b y + c;
*                // y0 = d x + e y + f;
*                // z0 = g x + h y + i;
*        vx_float32 mat[3][3] = {
*        {a, d, g}, // 'x' coefficients
*        {b, e, h}, // 'y' coefficients
*        {c, f, i}, // 'offsets'
*    };
*    vx_matrix matrix = vxCreateMatrix(context, VX_TYPE_FLOAT32, 3, 3);
*    vxCopyMatrix(matrix, mat, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
*param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] perspectiveMatrix(3x3)
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
    RppStatus
    rppi_warp_perspective_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);

    // ----------------------------------------
    // CPU warp_perspective functions declaration
    // ----------------------------------------
    /* Performs a perspective transform on an image.
*This kernel performs an perspective transform with a 3x3 Matrix M with this method of pixel coordinate translation:
*                // x0 = a x + b y + c;
*                // y0 = d x + e y + f;
*                // z0 = g x + h y + i;
*        vx_float32 mat[3][3] = {
*        {a, d, g}, // 'x' coefficients
*        {b, e, h}, // 'y' coefficients
*        {c, f, i}, // 'offsets'
*    };
*    vx_matrix matrix = vxCreateMatrix(context, VX_TYPE_FLOAT32, 3, 3);
*    vxCopyMatrix(matrix, mat, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
*param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] perspectiveMatrix(3x3)
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
    RppStatus
    rppi_warp_perspective_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, RppiSize *dstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_warp_perspective_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *perspectiveMatrix, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle);

    // Float based Calls
    RppStatus
    rppi_resize_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                     RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                     RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                     RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                     RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                     RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                     RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                    RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                    RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                    RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                       RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                       RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                       RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);

    RppStatus
    rppi_resize_u8_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                        RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                        RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                        RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);

    RppStatus
    rppi_resize_u8_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                        RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                        RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_u8_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize,
                                        RppiSize maxDstSize, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);

    RppStatus
    rppi_resize_crop_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd,
                                          Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd,
                                          Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd,
                                          Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd,
                                          Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd,
                                          Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd,
                                          Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd,
                                         Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd,
                                         Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
    RppStatus
    rppi_resize_crop_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd,
                                         Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFomatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);

#ifdef __cplusplus
}
#endif
#endif
