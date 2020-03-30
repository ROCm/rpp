#ifndef RPPI_MORPHOLOGICAL_TRANSFORMS_H
#define RPPI_MORPHOLOGICAL_TRANSFORMS_H
 
#include "rppdefs.h"
#include "rpp/rpp.h"
#ifdef __cplusplus
extern "C" {
#endif


// ----------------------------------------
// GPU erode functions declaration 
// ----------------------------------------
/* This function uses a N x N box around the output pixel used to determine value.
*dest(x srcPtr y) srcSize = min(xi yi) dstPtr
*x-bound kernelSize < xi < x+bound and x-bound < xi < x+bound
*bound rppHandle = (kernelsize + 1) / 2
*param [in] srcPtr input image
*param[in]  srcSize dimensions of the images
*param[out] dstPtr output image
*param[in] kernelSize dimension of the kernel
*param[in]  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
RppStatus
 rppi_erode_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU erode functions declaration 
// ----------------------------------------
/* This function uses a N x N box around the output pixel used to determine value.
*dest(x srcPtr y) srcSize = min(xi yi) dstPtr
*x-bound kernelSize < xi < x+bound and x-bound < xi < x+bound
*bound rppHandle = (kernelsize + 1) / 2
*param [in] srcPtr input image
*param[in]  srcSize dimensions of the images
*param[out] dstPtr output image
*param[in] kernelSize dimension of the kernel
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
RppStatus
 rppi_erode_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_erode_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU dilate functions declaration 
// ----------------------------------------
/* This function uses a N x N box around the output pixel used to determine value.
*dest(x srcPtr y) srcSize = max(xi yi) dstPtr
*x-bound kernelSize < xi < x+bound and x-bound < xi < x+bound
*bound rppHandle = (kernelsize + 1) / 2
*param [in] srcPtr input image
*param[in]  srcSize dimensions of the images
*param[out] dstPtr output image
*param[in] kernelSize dimension of the kernel
*param[in]  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_dilate_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU dilate functions declaration 
// ----------------------------------------
/* This function uses a N x N box around the output pixel used to determine value.
*dest(x srcPtr y) srcSize = max(xi yi) dstPtr
*x-bound kernelSize < xi < x+bound and x-bound < xi < x+bound
*bound rppHandle = (kernelsize + 1) / 2
*param [in] srcPtr input image
*param[in]  srcSize dimensions of the images
*param[out] dstPtr output image
*param[in] kernelSize dimension of the kernel
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_dilate_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_dilate_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

 
#ifdef __cplusplus
}
#endif
#endif