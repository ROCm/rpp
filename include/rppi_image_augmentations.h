#ifndef RPPI_IMAGE_AUGMENTATIONS_H
#define RPPI_IMAGE_AUGMENTATIONS_H
#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif




// ----------------------------------------
// GPU brightness functions declaration 
// ----------------------------------------
/* Computes brightness of an image.
*param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] alpha alpha for brightness calculation and value should be between 0 and 20
*param[in] beta beta value for brightness calculation and value should be between 0 and 255
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_brightness_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU brightness functions declaration 
// ----------------------------------------
/* Computes brightness of an image.
*param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] alpha alpha for brightness calculation and value should be between 0 and 20
*param[in] beta beta value for brightness calculation and value should be between 0 and 255
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_brightness_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_brightness_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU blend functions declaration 
// ----------------------------------------
/* Blends two source image and stores it in destination image.
*param srcPtr1 [in] srcPtr1 input image1
*param[in] srcPtr2 srcPtr2 input image2
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] alpha alpha transperancy factor of the images where alpha is for image1 and 1-alpha is for image2
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_blend_u8_pln1_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

// ----------------------------------------
// CPU blend functions declaration 
// ----------------------------------------
/* Blends two source image and stores it in destination image.
*param srcPtr1 [in] srcPtr1 input image1
*param[in] srcPtr2 srcPtr2 input image2
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] alpha alpha transperancy factor of the images where alpha is for image1 and 1-alpha is for image2
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
RppStatus
 rppi_blend_u8_pln1_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blend_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

// ----------------------------------------
// GPU gamma_correction functions declaration 
// ----------------------------------------
/* Computes gamma correction for an image.
param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] gamma gamma value used in gamma correction
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_gamma_correction_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU gamma_correction functions declaration 
// ----------------------------------------
/* Computes gamma correction for an image.
*param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] gamma gamma value used in gamma correction
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_gamma_correction_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gamma_correction_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *gamma ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU pixelate functions declaration 
// ----------------------------------------
/*pixelates the roi region of the image.
*param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] rppHandle gamma value used in gamma correction
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_pixelate_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU pixelate functions declaration 
// ----------------------------------------
/*pixelates the roi region of the image.
*pixelates the roi region of the image.
*param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] rppHandle gamma value used in gamma correction
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_pixelate_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_pixelate_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU fog functions declaration 
// ----------------------------------------
/* Introduces foggy effect in the entire image.
*param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] fogValue exposure value for modification
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_fog_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU fog functions declaration 
// ----------------------------------------
/*Introduces foggy effect in the entire image.
*param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] fogValue exposure value for modification
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_fog_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU snow functions declaration 
// ----------------------------------------
/* Introduces snowy effect in the entire image.
param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] snowValue exposure value for modification
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_snow_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU snow functions declaration 
// ----------------------------------------
/* Introduces snowy effect in the entire image.
param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] snowValue exposure value for modification
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_snow_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_snow_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *snowValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU jitter functions declaration 
// ----------------------------------------
/* Introduces jitter in the entire image.
*param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] kernelSize exposure value for modification
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_jitter_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kenelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kenelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kenelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kenelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kenelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kenelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kenelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kenelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kenelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kenelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kenelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kenelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kenelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kenelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kenelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kenelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kenelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kenelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kenelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kenelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kenelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kenelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kenelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kenelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kenelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kenelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kenelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kenelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );



// ----------------------------------------
// CPU jitter functions declaration 
// ----------------------------------------
/*Introduces jitter in the entire image.
*param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] kernelSize exposure value for modification
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_jitter_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_jitter_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU contrast functions declaration 
// ----------------------------------------
/* Computes contrast of the image using contrast stretch technique.
*param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] newMin minimum pixel value for contrast stretch
*param[in] newMax maxium pixel value for contrast stretch
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_contrast_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU contrast functions declaration 
// ----------------------------------------
/* Computes contrast of the image using contrast stretch technique.
*param[in] srcPtr  input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] newMin minimum pixel value for contrast stretch
*param[in] newMax maxium pixel value for contrast stretch
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_contrast_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u newMin ,Rpp32u newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_contrast_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *newMin ,Rpp32u *newMax ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU exposure functions declaration 
// ----------------------------------------
/* Modifies the exposure of the image using contrast stretch technique.
*param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] exposureValue exposure value for modification
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_exposure_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU exposure functions declaration 
// ----------------------------------------
/* Modifies the exposure of the image using contrast stretch technique.
*param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] exposureValue exposure value for modification
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_exposure_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exposure_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *exposureValue ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU blur functions declaration 
// ----------------------------------------
/* Uses Gaussian for blurring the image.
*param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] stdDev standard deviation value to populate gaussian kernels
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
RppStatus
 rppi_blur_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU blur functions declaration 
// ----------------------------------------
/* Uses Gaussian for blurring the image.
*param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] stdDev standard deviation value to populate gaussian kernels
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
RppStatus
 rppi_blur_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
 rppi_blur_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_blur_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------                                                                                                                                                                                                                                                                                                     
// GPU rain functions declaration 
// ----------------------------------------
/* Introduces rainy effect in the entire image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] rainPercentage rainPercentage float value to decide the amount of rainy effect to be added which should range between 0 - 1
*param[in] rainWidth rainWidth width of the rain line
*param[in] rainHeight rainHeight height of the rain line
*param[in] rain transparency float value to decide the amount of rain transparency to be added which should range between 0 - 1
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
 */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
RppStatus
 rppi_rain_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f rainPercentage ,Rpp32u rainWidth ,Rpp32u rainHeight ,Rpp32f transperancy ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f rainPercentage ,Rpp32u rainWidth ,Rpp32u rainHeight ,Rpp32f transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *rainPercentage ,Rpp32u *rainWidth ,Rpp32u *rainHeight ,Rpp32f *transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f rainPercentage ,Rpp32u rainWidth ,Rpp32u rainHeight ,Rpp32f transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *rainPercentage ,Rpp32u *rainWidth ,Rpp32u *rainHeight ,Rpp32f *transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f rainPercentage ,Rpp32u rainWidth ,Rpp32u rainHeight ,Rpp32f transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *rainPercentage ,Rpp32u *rainWidth ,Rpp32u *rainHeight ,Rpp32f *transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f rainPercentage ,Rpp32u rainWidth ,Rpp32u rainHeight ,Rpp32f transperancy ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f rainPercentage ,Rpp32u rainWidth ,Rpp32u rainHeight ,Rpp32f transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *rainPercentage ,Rpp32u *rainWidth ,Rpp32u *rainHeight ,Rpp32f *transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f rainPercentage ,Rpp32u rainWidth ,Rpp32u rainHeight ,Rpp32f transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *rainPercentage ,Rpp32u *rainWidth ,Rpp32u *rainHeight ,Rpp32f *transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f rainPercentage ,Rpp32u rainWidth ,Rpp32u rainHeight ,Rpp32f transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *rainPercentage ,Rpp32u *rainWidth ,Rpp32u *rainHeight ,Rpp32f *transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f rainPercentage ,Rpp32u rainWidth ,Rpp32u rainHeight ,Rpp32f transperancy ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f rainPercentage ,Rpp32u rainWidth ,Rpp32u rainHeight ,Rpp32f transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *rainPercentage ,Rpp32u *rainWidth ,Rpp32u *rainHeight ,Rpp32f *transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f rainPercentage ,Rpp32u rainWidth ,Rpp32u rainHeight ,Rpp32f transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *rainPercentage ,Rpp32u *rainWidth ,Rpp32u *rainHeight ,Rpp32f *transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f rainPercentage ,Rpp32u rainWidth ,Rpp32u rainHeight ,Rpp32f transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *rainPercentage ,Rpp32u *rainWidth ,Rpp32u *rainHeight ,Rpp32f *transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU rain functions declaration 
// ----------------------------------------
/* Introduces rainy effect in the entire image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] rainPercentage rainPercentage float value to decide the amount of rainy effect to be added which should range between 0 - 1
*param[in] rainWidth rainWidth width of the rain line
*param[in] rainHeight rainHeight height of the rain line
*param[in] rain transparency float value to decide the amount of rain transparency to be added which should range between 0 - 1
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_rain_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f rainPercentage ,Rpp32u rainWidth ,Rpp32u rainHeight ,Rpp32f transperancy ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f rainPercentage ,Rpp32u rainWidth ,Rpp32u rainHeight ,Rpp32f transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *rainPercentage ,Rpp32u *rainWidth ,Rpp32u *rainHeight ,Rpp32f *transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f rainPercentage ,Rpp32u rainWidth ,Rpp32u rainHeight ,Rpp32f transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *rainPercentage ,Rpp32u *rainWidth ,Rpp32u *rainHeight ,Rpp32f *transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f rainPercentage ,Rpp32u rainWidth ,Rpp32u rainHeight ,Rpp32f transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *rainPercentage ,Rpp32u *rainWidth ,Rpp32u *rainHeight ,Rpp32f *transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f rainPercentage ,Rpp32u rainWidth ,Rpp32u rainHeight ,Rpp32f transperancy ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f rainPercentage ,Rpp32u rainWidth ,Rpp32u rainHeight ,Rpp32f transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *rainPercentage ,Rpp32u *rainWidth ,Rpp32u *rainHeight ,Rpp32f *transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f rainPercentage ,Rpp32u rainWidth ,Rpp32u rainHeight ,Rpp32f transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *rainPercentage ,Rpp32u *rainWidth ,Rpp32u *rainHeight ,Rpp32f *transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f rainPercentage ,Rpp32u rainWidth ,Rpp32u rainHeight ,Rpp32f transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *rainPercentage ,Rpp32u *rainWidth ,Rpp32u *rainHeight ,Rpp32f *transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f rainPercentage ,Rpp32u rainWidth ,Rpp32u rainHeight ,Rpp32f transperancy ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f rainPercentage ,Rpp32u rainWidth ,Rpp32u rainHeight ,Rpp32f transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *rainPercentage ,Rpp32u *rainWidth ,Rpp32u *rainHeight ,Rpp32f *transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f rainPercentage ,Rpp32u rainWidth ,Rpp32u rainHeight ,Rpp32f transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *rainPercentage ,Rpp32u *rainWidth ,Rpp32u *rainHeight ,Rpp32f *transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f rainPercentage ,Rpp32u rainWidth ,Rpp32u rainHeight ,Rpp32f transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_rain_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *rainPercentage ,Rpp32u *rainWidth ,Rpp32u *rainHeight ,Rpp32f *transperancy ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU fog functions declaration 
// ----------------------------------------
/* Adds the fog effect of the image using contrast stretch technique.
*param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] fogValue exposure value for modification
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
/* RppStatus
 rppi_fog_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU fog functions declaration 
// ----------------------------------------
 Adds the fog effect of the image using contrast stretch technique.
param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] fogValue exposure value for modification
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
/* RppStatus
 rppi_fog_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fog_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *fogValue ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

// ----------------------------------------
// GPU histogram_balance functions declaration 
// ----------------------------------------
*Equalizes histogram of image.
*param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_histogram_balance_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU histogram_balance functions declaration 
// ----------------------------------------
/* Equalizes histogram of image.
*param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_histogram_balance_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_balance_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

// ----------------------------------------
// GPU noise functions declaration 
// ----------------------------------------
/* Introduces noise in the entire image using salt and pepper.
*param[in] srcPtr input image
*param[in] srcSize dimensions of the input images
*param[out] dstPtr output image
*param[in] noiseProbability float value to decide the amount of noise effect to be added which should range between 0 - 1
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
RppStatus
 rppi_noise_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU noise functions declaration 
// ----------------------------------------
/* Introduces noise in the entire image using salt and pepper.
*param[in] srcPtr input image
*param[in] srcSize dimensions of the input images
*param[out] dstPtr output image
*param[in] noiseProbability float value to decide the amount of noise effect to be added which should range between 0 - 1
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
RppStatus
 rppi_noise_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_noise_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *noiseProbability ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU random_shadow functions declaration 
// ----------------------------------------
/* Adds multiple random shadows [rectangle shaped shadows] in the image to the roi area.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] x1 value of roi
*param[in] y1 value of roi
*param[in] x2 value of roi
*param[in] y2 value of roi
*param[in]  numberOfShadows number of shadows to be added in the roi region
*param[in]  maxSizeX shadow's maximum width
*param[in]  maxSizeY shadow's maximum height
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
RppStatus
 rppi_random_shadow_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u x1 ,Rpp32u y1 ,Rpp32u x2 ,Rpp32u y2 ,Rpp32u numberOfShadows ,Rpp32u maxSizeX ,Rpp32u maxSizeY ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u x1 ,Rpp32u y1 ,Rpp32u x2 ,Rpp32u y2 ,Rpp32u numberOfShadows ,Rpp32u maxSizeX ,Rpp32u maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *x1 ,Rpp32u *y1 ,Rpp32u *x2 ,Rpp32u *y2 ,Rpp32u *numberOfShadows ,Rpp32u *maxSizeX ,Rpp32u *maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u x1 ,Rpp32u y1 ,Rpp32u x2 ,Rpp32u y2 ,Rpp32u numberOfShadows ,Rpp32u maxSizeX ,Rpp32u maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *x1 ,Rpp32u *y1 ,Rpp32u *x2 ,Rpp32u *y2 ,Rpp32u *numberOfShadows ,Rpp32u *maxSizeX ,Rpp32u *maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u x1 ,Rpp32u y1 ,Rpp32u x2 ,Rpp32u y2 ,Rpp32u numberOfShadows ,Rpp32u maxSizeX ,Rpp32u maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *x1 ,Rpp32u *y1 ,Rpp32u *x2 ,Rpp32u *y2 ,Rpp32u *numberOfShadows ,Rpp32u *maxSizeX ,Rpp32u *maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u x1 ,Rpp32u y1 ,Rpp32u x2 ,Rpp32u y2 ,Rpp32u numberOfShadows ,Rpp32u maxSizeX ,Rpp32u maxSizeY ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u x1 ,Rpp32u y1 ,Rpp32u x2 ,Rpp32u y2 ,Rpp32u numberOfShadows ,Rpp32u maxSizeX ,Rpp32u maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *x1 ,Rpp32u *y1 ,Rpp32u *x2 ,Rpp32u *y2 ,Rpp32u *numberOfShadows ,Rpp32u *maxSizeX ,Rpp32u *maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u x1 ,Rpp32u y1 ,Rpp32u x2 ,Rpp32u y2 ,Rpp32u numberOfShadows ,Rpp32u maxSizeX ,Rpp32u maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *x1 ,Rpp32u *y1 ,Rpp32u *x2 ,Rpp32u *y2 ,Rpp32u *numberOfShadows ,Rpp32u *maxSizeX ,Rpp32u *maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u x1 ,Rpp32u y1 ,Rpp32u x2 ,Rpp32u y2 ,Rpp32u numberOfShadows ,Rpp32u maxSizeX ,Rpp32u maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *x1 ,Rpp32u *y1 ,Rpp32u *x2 ,Rpp32u *y2 ,Rpp32u *numberOfShadows ,Rpp32u *maxSizeX ,Rpp32u *maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u x1 ,Rpp32u y1 ,Rpp32u x2 ,Rpp32u y2 ,Rpp32u numberOfShadows ,Rpp32u maxSizeX ,Rpp32u maxSizeY ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u x1 ,Rpp32u y1 ,Rpp32u x2 ,Rpp32u y2 ,Rpp32u numberOfShadows ,Rpp32u maxSizeX ,Rpp32u maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *x1 ,Rpp32u *y1 ,Rpp32u *x2 ,Rpp32u *y2 ,Rpp32u *numberOfShadows ,Rpp32u *maxSizeX ,Rpp32u *maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u x1 ,Rpp32u y1 ,Rpp32u x2 ,Rpp32u y2 ,Rpp32u numberOfShadows ,Rpp32u maxSizeX ,Rpp32u maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *x1 ,Rpp32u *y1 ,Rpp32u *x2 ,Rpp32u *y2 ,Rpp32u *numberOfShadows ,Rpp32u *maxSizeX ,Rpp32u *maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u x1 ,Rpp32u y1 ,Rpp32u x2 ,Rpp32u y2 ,Rpp32u numberOfShadows ,Rpp32u maxSizeX ,Rpp32u maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *x1 ,Rpp32u *y1 ,Rpp32u *x2 ,Rpp32u *y2 ,Rpp32u *numberOfShadows ,Rpp32u *maxSizeX ,Rpp32u *maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU random_shadow functions declaration 
// ----------------------------------------
/* Adds multiple random shadows [rectangle shaped shadows] in the image to the roi area.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] x1 value of roi
*param[in] y1 value of roi
*param[in] x2 value of roi
*param[in] y2 value of roi
*param[in]  numberOfShadows number of shadows to be added in the roi region
*param[in]  maxSizeX shadow's maximum width
*param[in]  maxSizeY shadow's maximum height
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
RppStatus
 rppi_random_shadow_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u x1 ,Rpp32u y1 ,Rpp32u x2 ,Rpp32u y2 ,Rpp32u numberOfShadows ,Rpp32u maxSizeX ,Rpp32u maxSizeY ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u x1 ,Rpp32u y1 ,Rpp32u x2 ,Rpp32u y2 ,Rpp32u numberOfShadows ,Rpp32u maxSizeX ,Rpp32u maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *x1 ,Rpp32u *y1 ,Rpp32u *x2 ,Rpp32u *y2 ,Rpp32u *numberOfShadows ,Rpp32u *maxSizeX ,Rpp32u *maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u x1 ,Rpp32u y1 ,Rpp32u x2 ,Rpp32u y2 ,Rpp32u numberOfShadows ,Rpp32u maxSizeX ,Rpp32u maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *x1 ,Rpp32u *y1 ,Rpp32u *x2 ,Rpp32u *y2 ,Rpp32u *numberOfShadows ,Rpp32u *maxSizeX ,Rpp32u *maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u x1 ,Rpp32u y1 ,Rpp32u x2 ,Rpp32u y2 ,Rpp32u numberOfShadows ,Rpp32u maxSizeX ,Rpp32u maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *x1 ,Rpp32u *y1 ,Rpp32u *x2 ,Rpp32u *y2 ,Rpp32u *numberOfShadows ,Rpp32u *maxSizeX ,Rpp32u *maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u x1 ,Rpp32u y1 ,Rpp32u x2 ,Rpp32u y2 ,Rpp32u numberOfShadows ,Rpp32u maxSizeX ,Rpp32u maxSizeY ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u x1 ,Rpp32u y1 ,Rpp32u x2 ,Rpp32u y2 ,Rpp32u numberOfShadows ,Rpp32u maxSizeX ,Rpp32u maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *x1 ,Rpp32u *y1 ,Rpp32u *x2 ,Rpp32u *y2 ,Rpp32u *numberOfShadows ,Rpp32u *maxSizeX ,Rpp32u *maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u x1 ,Rpp32u y1 ,Rpp32u x2 ,Rpp32u y2 ,Rpp32u numberOfShadows ,Rpp32u maxSizeX ,Rpp32u maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *x1 ,Rpp32u *y1 ,Rpp32u *x2 ,Rpp32u *y2 ,Rpp32u *numberOfShadows ,Rpp32u *maxSizeX ,Rpp32u *maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u x1 ,Rpp32u y1 ,Rpp32u x2 ,Rpp32u y2 ,Rpp32u numberOfShadows ,Rpp32u maxSizeX ,Rpp32u maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *x1 ,Rpp32u *y1 ,Rpp32u *x2 ,Rpp32u *y2 ,Rpp32u *numberOfShadows ,Rpp32u *maxSizeX ,Rpp32u *maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u x1 ,Rpp32u y1 ,Rpp32u x2 ,Rpp32u y2 ,Rpp32u numberOfShadows ,Rpp32u maxSizeX ,Rpp32u maxSizeY ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u x1 ,Rpp32u y1 ,Rpp32u x2 ,Rpp32u y2 ,Rpp32u numberOfShadows ,Rpp32u maxSizeX ,Rpp32u maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *x1 ,Rpp32u *y1 ,Rpp32u *x2 ,Rpp32u *y2 ,Rpp32u *numberOfShadows ,Rpp32u *maxSizeX ,Rpp32u *maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u x1 ,Rpp32u y1 ,Rpp32u x2 ,Rpp32u y2 ,Rpp32u numberOfShadows ,Rpp32u maxSizeX ,Rpp32u maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *x1 ,Rpp32u *y1 ,Rpp32u *x2 ,Rpp32u *y2 ,Rpp32u *numberOfShadows ,Rpp32u *maxSizeX ,Rpp32u *maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u x1 ,Rpp32u y1 ,Rpp32u x2 ,Rpp32u y2 ,Rpp32u numberOfShadows ,Rpp32u maxSizeX ,Rpp32u maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_shadow_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *x1 ,Rpp32u *y1 ,Rpp32u *x2 ,Rpp32u *y2 ,Rpp32u *numberOfShadows ,Rpp32u *maxSizeX ,Rpp32u *maxSizeY ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

// ----------------------------------------
// GPU random_crop_letterbox functions declaration 
// ----------------------------------------
/* Crops the roi region of source image adds border and stores it in destination
*param srcPtr [in] srcPtr input image
*param[in] srcSize dimensions of the images
*param[out] dstPtr output image
*param[in] xRoiBegin value of roi
*param[in] xRoiEnd value of roi
*param[in] yRoiBegin value of roi
*param[in] yRoiEnd value of roi
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
RppStatus
 rppi_random_crop_letterbox_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU random_crop_letterbox functions declaration 
// ----------------------------------------
/* Crops the roi region of source image adds border and stores it in destination
*param srcPtr [in] srcPtr input image
*param[in] srcSize dimensions of the images
*param[out] dstPtr output image
*param[in] xRoiBegin value of roi
*param[in] xRoiEnd value of roi
*param[in] yRoiBegin value of roi
*param[in] yRoiEnd value of roi
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_random_crop_letterbox_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_random_crop_letterbox_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

#ifdef __cplusplus
}
#endif
#endif
