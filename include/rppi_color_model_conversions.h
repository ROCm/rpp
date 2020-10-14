#ifndef RPPI_COLOR_MODEL_CONVERSIONS_H
#define RPPI_COLOR_MODEL_CONVERSIONS_H
 
#include "rppdefs.h"
#include "rpp.h"
#ifdef __cplusplus
extern "C" {
#endif


// ----------------------------------------
// GPU vignette functions declaration 
// ----------------------------------------
/* Introduces vignette effect in the entire image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] stdDev stdDev standard deviation for the gaussian function used in the vignette (decides amount of vignette)
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
RppStatus
 rppi_vignette_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU vignette functions declaration 
// ----------------------------------------
/* Introduces vignette effect in the entire image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] stdDev stdDev standard deviation for the gaussian function used in the vignette (decides amount of vignette)
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_vignette_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_vignette_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU channel_extract functions declaration 
// ----------------------------------------
/* Extract a single channel from given RGB image and stores it in destination grey scale image. 
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr Output image
*param[in] extractChannelNumber extractChannelNumber The channel to be extracted and it could be
*0---> rppHandle R channel
*1---> G channel
*2---> B channel
*param[in]  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_channel_extract_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u extractChannelNumber ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u extractChannelNumber ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u extractChannelNumber ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU channel_extract functions declaration 
// ----------------------------------------
/* Extract a single channel from given RGB image and stores it in destination grey scale image. 
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr Output image
*param[in] extractChannelNumber extractChannelNumber The channel to be extracted and it could be
*0---> rppHandle R channel
*1---> G channel
*2---> B channel
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_channel_extract_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u extractChannelNumber ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u extractChannelNumber ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u extractChannelNumber ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_extract_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *extractChannelNumber ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU channel_combine functions declaration 
// ----------------------------------------
/* Combines 3 greyscale images to produce a single RGB image. 
*param srcPtr1 [in] srcPtr1 Input image1
*param srcPtr2 [in] srcPtr2 Input image2
*param srcPtr3 [in] srcPtr3 Input image3
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr Output image
*param[in] rppHandle  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
RppStatus
 rppi_channel_combine_u8_pln1_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pln1_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pln1_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pln1_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pln1_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pln1_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pln3_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pln3_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pln3_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pln3_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pln3_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pln3_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pkd3_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU channel_combine functions declaration 
// ----------------------------------------
/* Combines 3 greyscale images to produce a single RGB image. 
*param srcPtr1 [in] srcPtr1 Input image1
*param srcPtr2 [in] srcPtr2 Input image2
*param srcPtr3 [in] srcPtr3 Input image3
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr Output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_channel_combine_u8_pln1_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pln1_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pln1_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pln1_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pln1_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pln1_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pln1_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pln3_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pln3_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pln3_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pln3_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pln3_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pln3_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pln3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pkd3_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pkd3_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pkd3_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pkd3_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pkd3_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pkd3_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_channel_combine_u8_pkd3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppPtr_t srcPtr3 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

 // ----------------------------------------
// GPU color_temperature functions declaration 
// ----------------------------------------
/* Changes color temperature of an image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] adjustmentValue adjustmentValue adjustment value used in color temperature correction which should range between -100 - 100
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_color_temperature_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU color_temperature functions declaration 
// ----------------------------------------
/* Changes color temperature of an image.
*param[in] srcPtr  srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] adjustmentValue adjustmentValue adjustment value used in color temperature correction which should range between -100 - 100
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_color_temperature_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_temperature_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32s *adjustmentValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );



// ----------------------------------------
// GPU hueRGB functions declaration 
// ----------------------------------------
/* Computes hue value and updates it in RGB image.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] hueShift hue shift for hue calculation
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
RppStatus
 rppi_hueRGB_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU hueRGB functions declaration 
// ----------------------------------------
/* Computes hue value and updates it in RGB image.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] hueShift hue shift for hue calculation
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_hueRGB_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_hueRGB_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *hueShift ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU saturationRGB functions declaration 
// ----------------------------------------
/* Computes saturation value and updates it in RGB image.
*param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] saturationFactor saturationFactor for saturation calculation
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_saturationRGB_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU saturationRGB functions declaration 
// ----------------------------------------
/* Computes saturation value and updates it in RGB image.
*param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] saturationFactor saturationFactor for saturation calculation
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
RppStatus
 rppi_saturationRGB_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_saturationRGB_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU look_up_table functions declaration 
// ----------------------------------------
/* This function uses each pixel in an image to index into a LUT and put the indexed LUT value into the output image.
*param[in] srcPtr  srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] lutPtr lutPtr contains the input look up table values
*param[in] rppHandle  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_look_up_table_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU look_up_table functions declaration 
// ----------------------------------------
/* This function uses each pixel in an image to index into a LUT and put the indexed LUT value into the output image.
*param[in] srcPtr  srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] lutPtr lutPtr contains the input look up table values
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error  
*/

RppStatus
 rppi_look_up_table_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_look_up_table_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u* lutPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

// ----------------------------------------
// GPU Color Convert 
// ----------------------------------------
/*param[in] srcPtr  input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*retval RPP_ERROR : Error 
*/

RppStatus
 rppi_color_convert_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr , RppiColorConvertMode convert_mode, rppHandle_t rppHandle );
RppStatus
 rppi_color_convert_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr , RppiColorConvertMode convert_mode, Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_convert_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr , RppiColorConvertMode convert_mode, Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_convert_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr , RppiColorConvertMode convert_mode, Rpp32u nbatchSize ,rppHandle_t rppHandle );

RppStatus
 rppi_color_convert_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr , RppiColorConvertMode convert_mode, rppHandle_t rppHandle );
RppStatus
 rppi_color_convert_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr , RppiColorConvertMode convert_mode, Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_convert_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr , RppiColorConvertMode convert_mode, Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_convert_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr , RppiColorConvertMode convert_mode, Rpp32u nbatchSize ,rppHandle_t rppHandle );

// ----------------------------------------
// CPU Color Convert 
// ----------------------------------------
/*param[in] srcPtr  input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*retval RPP_ERROR : Error 
*/

RppStatus
 rppi_color_convert_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr , RppiColorConvertMode convert_mode, rppHandle_t rppHandle );
RppStatus
 rppi_color_convert_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr , RppiColorConvertMode convert_mode, Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_convert_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr , RppiColorConvertMode convert_mode, Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_convert_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr , RppiColorConvertMode convert_mode, Rpp32u nbatchSize ,rppHandle_t rppHandle );

RppStatus
 rppi_color_convert_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr , RppiColorConvertMode convert_mode, rppHandle_t rppHandle );
RppStatus
 rppi_color_convert_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr , RppiColorConvertMode convert_mode, Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_convert_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr , RppiColorConvertMode convert_mode, Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_convert_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr , RppiColorConvertMode convert_mode, Rpp32u nbatchSize ,rppHandle_t rppHandle );




#ifdef __cplusplus
}
#endif
#endif

