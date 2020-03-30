#ifndef RPPI_STATISTICAL_OPERATIONS_H
#define RPPI_STATISTICAL_OPERATIONS_H
 
#include "rppdefs.h"
#include "rpp/rpp.h"
#ifdef __cplusplus
extern "C" {
#endif


// ----------------------------------------
// GPU min functions declaration 
// ----------------------------------------
/* Computes pixel wise minimum on input images and stores the result in destination image.
*param srcPtr1 [in] srcPtr1 input image1
*param[in] srcPtr2 srcPtr2 input image2 
*param[in] srcSize srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_min_u8_pln1_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU min functions declaration 
// ----------------------------------------
/* Computes pixel wise minimum on input images and stores the result in destination image.
*param srcPtr1 [in] srcPtr1 input image1
*param[in] srcPtr2 srcPtr2 input image2 
*param[in] srcSize srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_min_u8_pln1_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_min_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU max functions declaration 
// ----------------------------------------
/* Computes pixel wise maximum on input images and stores the result in destination image.
*param[in] srcPtr1  srcPtr1 input image1
*param[in] srcPtr2 srcPtr2 input image2 
*param[in] srcSize srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] rppHandle  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
RppStatus
 rppi_max_u8_pln1_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU max functions declaration 
// ----------------------------------------
/*Computes pixel wise maximum on input images and stores the result in destination image.
*param[in] srcPtr1  srcPtr1 input image1
*param[in] srcPtr2 srcPtr2 input image2 
*param[in] srcSize srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_max_u8_pln1_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_max_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU histogram_equalization functions declaration 
// ----------------------------------------
/* Does histogram equalization on the input image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] rppHandle  rppHandle OpenCL handle	
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_histogram_equalization_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU histogram_equalization functions declaration 
// ----------------------------------------
/* Does histogram equalization on the input image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_histogram_equalization_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_histogram_equalization_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU thresholding functions declaration 
// ----------------------------------------
/*  Thresholds an input image and produces an output Boolean image.
param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_thresholding_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU thresholding functions declaration 
// ----------------------------------------
/* Thresholds an input image and produces an output Boolean image.
param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_thresholding_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_thresholding_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


/* Mean and Standard Deviation - Single Implementations for scalar calls */
//CPU
/* Computes the mean pixel value and the standard deviation of the pixels in the input image.
*param srcPtr [in] input image
*param[in] srcSize dimensions of the images
*param[out] mean mean of the pixel values in the input image
*param[out] stdDev stddev standard deviation of the pixels values in the input image 
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/

RppStatus
 rppi_mean_stddev_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,Rpp32f *mean ,Rpp32f *stddev ,rppHandle_t rppHandle );
RppStatus
 rppi_mean_stddev_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,Rpp32f *mean ,Rpp32f *stddev ,rppHandle_t rppHandle );
RppStatus
 rppi_mean_stddev_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,Rpp32f *mean ,Rpp32f *stddev ,rppHandle_t rppHandle );

//CPU
/*This function finds the minimum and maximum values and its corresponding location and stores it in the output variables.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] min min minimum pixel value in the input image
 max *param[out] max maximum pixel value in the input image
 minLoc *param[out] minLoc minimum pixel's index in the input image
 maxLoc *param[out] max maximum pixel's index in the input image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/

RppStatus
 rppi_min_max_loc_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,Rpp8u *min ,Rpp8u *max ,Rpp32u *minLoc ,Rpp32u *maxLoc ,rppHandle_t rppHandle );
RppStatus
 rppi_min_max_loc_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,Rpp8u *min ,Rpp8u *max ,Rpp32u *minLoc ,Rpp32u *maxLoc ,rppHandle_t rppHandle );
RppStatus
 rppi_min_max_loc_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,Rpp8u *min ,Rpp8u *max ,Rpp32u *minLoc ,Rpp32u *maxLoc ,rppHandle_t rppHandle );

//CPU
/* Computes the histogram of image and stores it in the histogram array of size bins.
*param srcPtr [in] srcPtr input image
*param[in] srcSize srcSize dimensions of the images
*param[out] outputHistogram outputHistogram pointer to store the histogram of the input image
*param[in] bins bins size of output histogram 
*param[in] rppHandle  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/

RppStatus
 rppi_histogram_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,Rpp32u* outputHistogram ,Rpp32u bins ,rppHandle_t rppHandle);
RppStatus
 rppi_histogram_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,Rpp32u* outputHistogram ,Rpp32u bins ,rppHandle_t rppHandle);
RppStatus
 rppi_histogram_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,Rpp32u* outputHistogram ,Rpp32u bins ,rppHandle_t rppHandle);

//GPU
/*Computes the mean pixel value and the standard deviation of the pixels in the input image.
*param srcPtr [in] input image
*param[in] srcSize dimensions of the images
*param[out] mean mean mean of the pixel values in the input image
*param[out] stdDev stddev standard deviation of the pixels values in the input image 
*param[in] rppHandle  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/

RppStatus
 rppi_mean_stddev_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,Rpp32f *mean ,Rpp32f *stddev ,rppHandle_t rppHandle );
RppStatus
 rppi_mean_stddev_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,Rpp32f *mean ,Rpp32f *stddev ,rppHandle_t rppHandle );
RppStatus
 rppi_mean_stddev_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,Rpp32f *mean ,Rpp32f *stddev ,rppHandle_t rppHandle );

//GPU
/* This function finds the minimum and maximum values and its corresponding location and stores it in the output variables.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] min min minimum pixel value in the input image
 max *param[out] max maximum pixel value in the input image
 minLoc *param[out] minLoc minimum pixel's index in the input image
 maxLoc *param[out] max maximum pixel's index in the input image
*param[in] rppHandle  rppHandle OpenCL handle 
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/

RppStatus
 rppi_min_max_loc_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,Rpp8u *min ,Rpp8u *max ,Rpp32u *minLoc ,Rpp32u *maxLoc ,rppHandle_t rppHandle );
RppStatus
 rppi_min_max_loc_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,Rpp8u *min ,Rpp8u *max ,Rpp32u *minLoc ,Rpp32u *maxLoc ,rppHandle_t rppHandle );
RppStatus
 rppi_min_max_loc_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,Rpp8u *min ,Rpp8u *max ,Rpp32u *minLoc ,Rpp32u *maxLoc ,rppHandle_t rppHandle );

//GPU
/* Computes the histogram of image and stores it in the histogram array of size bins.
*param srcPtr [in] srcPtr input image
*param[in] srcSize srcSize dimensions of the images
*param[out] outputHistogram outputHistogram pointer to store the histogram of the input image
*param[in] bins bins size of output histogram 
*param[in] rppHandle  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/

RppStatus
 rppi_histogram_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,Rpp32u* outputHistogram ,Rpp32u bins ,rppHandle_t rppHandle);
RppStatus
 rppi_histogram_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,Rpp32u* outputHistogram ,Rpp32u bins ,rppHandle_t rppHandle);
RppStatus
 rppi_histogram_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,Rpp32u* outputHistogram ,Rpp32u bins ,rppHandle_t rppHandle);


// ----------------------------------------
// GPU integral functions declaration 
// ----------------------------------------
/* Computes the integral image of the input.
param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_integral_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU integral functions declaration 
// ----------------------------------------
/*Computes the integral image of the input.
*param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_integral_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_integral_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

#ifdef __cplusplus
}
#endif
#endif