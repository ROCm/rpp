#ifndef RPPI_COMPUTER_VISION_H
#define RPPI_COMPUTER_VISION_H
 
#include "rppdefs.h"
#include "rpp.h"
#ifdef __cplusplus
extern "C" {
#endif


// ----------------------------------------
// GPU data_object_copy functions declaration 
// ----------------------------------------
/* Does deep data object copy of the image using contrast stretch technique.
*param[in] srcPtr  input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_data_object_copy_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU data_object_copy functions declaration 
// ----------------------------------------
/* Does deep data object copy of the image using contrast stretch technique.
param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_data_object_copy_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_data_object_copy_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU local_binary_pattern functions declaration 
// ----------------------------------------
/* Extracts LBP image from an input image and stores it in the output image.
 srcPtr Local binary pattern is defined as: Each pixel (y x) srcSize generate an 8 bit value describing the local binary pattern around the pixel
 dstPtr by comparing the pixel value with its 8 neighbours (selected neighbours of the 3x3 or 5x5 window).
*param[in] rppHandle  srcPtr input image
*param[in]  srcSize dimensions of the images
*param[out] dstPtr output image
*param[in]  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
RppStatus
 rppi_local_binary_pattern_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU local_binary_pattern functions declaration 
// ----------------------------------------
/* Extracts LBP image from an input image and stores it in the output image.
 srcPtr Local binary pattern is defined as: Each pixel (y x) srcSize generate an 8 bit value describing the local binary pattern around the pixel
 dstPtr by comparing the pixel value with its 8 neighbours (selected neighbours of the 3x3 or 5x5 window).
*param[in] rppHandle  srcPtr input image
*param[in]  srcSize dimensions of the images
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_local_binary_pattern_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );




RppStatus
 rppi_local_binary_pattern_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_local_binary_pattern_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// // ----------------------------------------
// // GPU local_binary_pattern functions declaration 
// // ----------------------------------------
// /* Puts local binary pattern in the destination pointer of the image using contrast stretch technique.
// param srcPtr [in] input image
// *param[in] srcSize dimensions of the image
// *param[out] dstPtr output image
// *returns a  RppStatus enumeration. 
// *retval RPP_SUCCESS : No error succesful completion
// *retval RPP_ERROR : Error 
// */
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// // ----------------------------------------
// // CPU local_binary_pattern functions declaration 
// // ----------------------------------------
// /* Puts local binary pattern in the destination pointer of the image using contrast stretch technique.
// param srcPtr [in] input image
// *param[in] srcSize dimensions of the image
// *param[out] dstPtr output image
// *returns a  RppStatus enumeration. 
// *retval RPP_SUCCESS : No error succesful completion
// *retval RPP_ERROR : Error 
// */
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );



// // ----------------------------------------
// // GPU local_binary_pattern functions declaration 
// // ----------------------------------------
// /* Puts local binary pattern in the destination pointer of the image using contrast stretch technique.
// param srcPtr [in] input image
// *param[in] srcSize dimensions of the image
// *param[out] dstPtr output image
// *returns a  RppStatus enumeration. 
// *retval RPP_SUCCESS : No error succesful completion
// *retval RPP_ERROR : Error 
// */
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// // ----------------------------------------
// // CPU local_binary_pattern functions declaration 
// // ----------------------------------------
// /* Puts local binary pattern in the destination pointer of the image using contrast stretch technique.
// param srcPtr [in] input image
// *param[in] srcSize dimensions of the image
// *param[out] dstPtr output image
// *returns a  RppStatus enumeration. 
// *retval RPP_SUCCESS : No error succesful completion
// *retval RPP_ERROR : Error 
// */
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_local_binary_pattern_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
// ----------------------------------------
// GPU canny_edge_detector functions declaration 
// ----------------------------------------
/* Canny Edge Detector technique.
Provides a Canny edge detector kernel. 
*param[in] srcPtr  input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_canny_edge_detector_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU canny_edge_detector functions declaration 
// ----------------------------------------
/* Canny Edge Detector technique.
*param[in] srcPtr  input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_canny_edge_detector_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_canny_edge_detector_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU gaussian_image_pyramid functions declaration 
// ----------------------------------------
/* Computes a Gaussian Image Pyramid from an input image and stores it in the destination image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] stdDev kernel 
*param[in] kernelSize stdDev standard deviation value to populate gaussian kernel
*param[in] rppHandle kernelSize size of the kernel
*param[in]  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_gaussian_image_pyramid_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU gaussian_image_pyramid functions declaration 
// ----------------------------------------
/* Computes a Gaussian Image Pyramid from an input image and stores it in the destination image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] stdDev kernel 
*param[in] stdDev standard deviation value to populate gaussian kernel
*param[in] rppHandle kernelSize size of the kernel
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_gaussian_image_pyramid_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_image_pyramid_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// ----------------------------------------
// GPU laplacian_image_pyramid functions declaration 
// ----------------------------------------
/* Computes a laplacian Image Pyramid from an input image and stores it in the destination image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] stdDev kernel 
*param[in] kernelSize stdDev standard deviation value to populate gaussian kernel
*param[in] rppHandle kernelSize size of the kernel
*param[in]  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_laplacian_image_pyramid_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU laplacian_image_pyramid functions declaration 
// ----------------------------------------
/* Computes a laplacian Image Pyramid from an input image and stores it in the destination image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] stdDev kernel 
*param[in] kernelSize stdDev standard deviation value to populate gaussian kernel
*param[in] rppHandle kernelSize size of the kernel
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_laplacian_image_pyramid_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_laplacian_image_pyramid_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// ----------------------------------------
// GPU harris_corner_detector functions declaration 
// ----------------------------------------
/* Computes the Harris Corners of an image.
*param[in] srcPtr  input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_harris_corner_detector_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU harris_corner_detector functions declaration 
// ----------------------------------------
/* Computes the Harris Corners of an image.
*param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_harris_corner_detector_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_harris_corner_detector_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU fast_corner_detector functions declaration 
// ----------------------------------------
/*Computes the Fast Corners of an image.
*param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_fast_corner_detector_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU fast_corner_detector functions declaration 
// ----------------------------------------
/* Computes the Fast Corners of an image.
*param[in] srcPtr  input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_fast_corner_detector_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_fast_corner_detector_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU remap functions declaration 
// ----------------------------------------
/* 
Remap takes a remap table object vx_remap to map a set of output pixels back to source input pixels. A remap is typically defined as:
output(x,y)=input(mapx(x,y),mapy(x,y));
for every (x,y) in the destination image
However, the mapping functions are contained in the vx_remap object.
param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
param[in] rowRemapTable gamma value used in gamma correction
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_remap_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU remap functions declaration 
// ----------------------------------------
/* Remap takes a remap table object vx_remap to map a set of output pixels back to source input pixels. A remap is typically defined as:
output(x,y)=input(mapx(x,y),mapy(x,y));
for every (x,y) in the destination image
However, the mapping functions are contained in the vx_remap object.
param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
param[in] rowRemapTable gamma value used in gamma correction
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error  
*/
RppStatus
 rppi_remap_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_remap_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

RppStatus
    rpp_bool_control_flow(bool num1, bool num2, bool *output, RppOp operation, rppHandle_t rppHandle);
RppStatus
    rpp_u8_control_flow(Rpp8u num1, Rpp8u num2, Rpp8u *output, RppOp operation, rppHandle_t rppHandle);
RppStatus
    rpp_i8_control_flow(Rpp8s num1, Rpp8s num2, Rpp8s *output, RppOp operation, rppHandle_t rppHandle);
RppStatus
    rpp_f32_control_flow(Rpp32f num1, Rpp32f num2, Rpp32f *output, RppOp operation, rppHandle_t rppHandle);

#ifdef __cplusplus

}
#endif
#endif