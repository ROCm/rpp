#ifndef RPPI_LOGICAL_OPERATIONS_H
#define RPPI_LOGICAL_OPERATIONS_H
 
#include "rppdefs.h"
#include "rpp.h"
#ifdef __cplusplus
extern "C" {
#endif


// ----------------------------------------
// GPU exclusive_OR functions declaration 
// ----------------------------------------
/* Computes the bitwise OR between two images.
*param[in] srcPtr1 input image1
*param[in] srcPtr2 input image2 
*param[in] srcSize dimensions of the images
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_exclusive_OR_u8_pln1_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU exclusive_OR functions declaration 
// ----------------------------------------
/* Computes the bitwise OR between two images.
*param[in] srcPtr1 input image1
*param[in] srcPtr2 input image2 
*param[in] srcSize dimensions of the images
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_exclusive_OR_u8_pln1_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_exclusive_OR_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU inclusive_OR functions declaration 
// ----------------------------------------
/* Computes the bitwise OR between two images.
*param[in] srcPtr1 input image1
*param[in] srcPtr2 input image2 
*param[in] srcSize dimensions of the images
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
RppStatus
 rppi_inclusive_OR_u8_pln1_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU inclusive_OR functions declaration 
// ----------------------------------------
/*Computes the bitwise OR between two images.
*param[in] srcPtr1 input image1
*param[in] srcPtr2 input image2 
*param[in] srcSize dimensions of the images
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_inclusive_OR_u8_pln1_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_inclusive_OR_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU bitwise_AND functions declaration 
// ----------------------------------------
/* Computes the bitwise AND between two images.
*param srcPtr1 [in] input image
*param[in] srcPtr2 dimensions of the image
*param[out] srcSize output image
*param[in] dstPtr exposure value for modification
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_bitwise_AND_u8_pln1_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU bitwise_AND functions declaration 
// ----------------------------------------
/* Computes the bitwise AND between two images.
*param srcPtr1 [in] input image
*param[in] srcPtr2 dimensions of the image
*param[out] srcSize output image
*param[in] dstPtr exposure value for modification
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_bitwise_AND_u8_pln1_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_AND_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU bitwise_NOT functions declaration 
// ----------------------------------------
/* Computes the bitwise_NOT for the given images.
*param srcPtr1 [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_bitwise_NOT_u8_pln1_gpu(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_ROI_gpu(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchSS_gpu(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchDS_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchPS_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchSD_gpu(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchDD_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_gpu(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_ROI_gpu(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchSS_gpu(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchDS_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchPS_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchSD_gpu(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchDD_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_gpu(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_ROI_gpu(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU bitwise_NOT functions declaration 
// ----------------------------------------
/* Computes the bitwise_NOT for the given images.
param srcPtr1 [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_bitwise_NOT_u8_pln1_host(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_ROI_host(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchSS_host(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchDS_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchPS_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchSD_host(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchDD_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchPD_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_host(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_ROI_host(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchSS_host(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchDS_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchPS_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchSD_host(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchDD_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchPD_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_host(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_ROI_host(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchSS_host(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchDS_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchPS_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchSD_host(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchDD_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchPD_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bitwise_NOT_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

 
#ifdef __cplusplus
}
#endif
#endif