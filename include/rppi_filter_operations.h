#ifndef RPPI_FILTER_OPERATIONS_H
#define RPPI_FILTER_OPERATIONS_H
 
#include "rppdefs.h"
#include "rpp.h"
#ifdef __cplusplus
extern "C" {
#endif


// ----------------------------------------
// GPU gaussian_filter functions declaration 
// ----------------------------------------
/* Applies gaussian filter over every pixel in the input image and stores it in the destination image.
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
 rppi_gaussian_filter_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU gaussian_filter functions declaration 
// ----------------------------------------
/*Applies gaussian filter over every pixel in the input image and stores it in the destination image.
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
 rppi_gaussian_filter_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_gaussian_filter_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU median_filter functions declaration 
// ----------------------------------------
/* This function uses a N x N box around the output pixel used to determine value.
dest(x,y) srcSize = median(xi,yi)
x-bound kernelSize < xi < x+bound and x-bound < xi < x+bound
bound = (kernelsize + 1) / 2
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
 rppi_median_filter_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU median_filter functions declaration 
// ----------------------------------------
/* This function uses a N x N box around the output pixel used to determine value.
dest(x,y) srcSize = median(xi,yi)
x-bound kernelSize < xi < x+bound and x-bound < xi < x+bound
bound = (kernelsize + 1) / 2
*param [in] srcPtr input image
*param[in]  srcSize dimensions of the images
*param[out] dstPtr output image
*param[in] kernelSize dimension of the kernel
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_median_filter_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU nonlinear_filter functions declaration 
// ----------------------------------------
/* Computes a non-linear filter over a window of the input image. The output image dimensions should be the same as the dimensions of the input image.
The attribute VX_CONTEXT_NONLINEAR_MAX_DIMENSION enables the user to query the largest nonlinear filter supported by the implementation of vxNonLinearFilterNode. 
The implementation must support all dimensions (height or width, not necessarily the same) up to the value of this attribute. 
The lowest value that must be supported for this attribute is 9.
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] kernelSize exposure value for modification
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_nonlinear_filter_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU nonlinear_filter functions declaration 
// ----------------------------------------
/* Computes a non-linear filter over a window of the input image. The output image dimensions should be the same as the dimensions of the input image.
The attribute VX_CONTEXT_NONLINEAR_MAX_DIMENSION enables the user to query the largest nonlinear filter supported by the implementation of vxNonLinearFilterNode. 
The implementation must support all dimensions (height or width, not necessarily the same) up to the value of this attribute. 
The lowest value that must be supported for this attribute is 9.
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] kernelSize exposure value for modification
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error  
*/
RppStatus
 rppi_nonlinear_filter_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_nonlinear_filter_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU non_max_suppression functions declaration 
// ----------------------------------------
/* This function uses a N x N box around the output pixel used to determine value. 
If the centre pixel is the maximum it will be retained else it will be replaced with zero.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] kernelSize kernelSize size of the kernel
*param[in] rppHandle  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_non_max_suppression_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU non_max_suppression functions declaration 
// ----------------------------------------
/* This function uses a N x N box around the output pixel used to determine value. 
If the centre pixel is the maximum it will be retained else it will be replaced with zero.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] kernelSize kernelSize size of the kernel
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_non_max_suppression_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_non_max_suppression_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU sobel_filter functions declaration 
// ----------------------------------------
/* Implements the Sobel Image Filter Kernel.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] sobelType sobelType 
*param[in] rppHandle  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
RppStatus
 rppi_sobel_filter_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU sobel_filter functions declaration 
// ----------------------------------------
/* Implements the Sobel Image Filter Kernel.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] sobelType sobelType 
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_sobel_filter_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_sobel_filter_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU median_filter functions declaration 
// ----------------------------------------
/* Median Filter of the image using contrast stretch technique.
param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] kernelSize exposure value for modification
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
 RppStatus
 rppi_median_filter_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// ----------------------------------------
// CPU median_filter functions declaration 
// ----------------------------------------
Median Filter of the image using contrast stretch technique.
param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] kernelSize exposure value for modification
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
RppStatus
 rppi_median_filter_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_median_filter_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
*/

// ----------------------------------------
// GPU nonlinear_filter functions declaration 
// ----------------------------------------
/* Median Filter of the image using contrast stretch technique.
*param srcPtr [in] input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] kernelSize exposure value for modification
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// // ----------------------------------------
// // CPU nonlinear_filter functions declaration 
// // ----------------------------------------
// /* Median Filter of the image using contrast stretch technique.
// param srcPtr [in] input image
// *param[in] srcSize dimensions of the image
// *param[out] dstPtr output image
// *param[in] kernelSize exposure value for modification
// *returns a  RppStatus enumeration. 
// *retval RPP_SUCCESS : No error succesful completion
// *retval RPP_ERROR : Error 
// */
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_nonlinear_filter_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// // ----------------------------------------
// // GPU non_max_suppression functions declaration 
// // ----------------------------------------
// /* Non maximum supression Filter of the image using contrast stretch technique.
// param srcPtr [in] input image
// *param[in] srcSize dimensions of the image
// *param[out] dstPtr output image
// *param[in] kernelSize exposure value for modification
// *returns a  RppStatus enumeration. 
// *retval RPP_SUCCESS : No error succesful completion
// *retval RPP_ERROR : Error 
// */
// RppStatus
//  rppi_non_max_suppression_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// // ----------------------------------------
// // CPU non_max_suppression functions declaration 
// // ----------------------------------------
// /* Non maximum supression Filter of the image using contrast stretch technique.
// param srcPtr [in] input image
// *param[in] srcSize dimensions of the image
// *param[out] dstPtr output image
// *param[in] kernelSize exposure value for modification
// *returns a  RppStatus enumeration. 
// *retval RPP_SUCCESS : No error succesful completion
// *retval RPP_ERROR : Error 
// */
// RppStatus
//  rppi_non_max_suppression_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_non_max_suppression_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// // ----------------------------------------
// // GPU sobel_filter functions declaration 
// // ----------------------------------------
// /* Median Filter of the image using contrast stretch technique.
// param srcPtr [in] input image
// *param[in] srcSize dimensions of the image
// *param[out] dstPtr output image
// *returns a  RppStatus enumeration. 
// *retval RPP_SUCCESS : No error succesful completion
// *retval RPP_ERROR : Error 
// */
// RppStatus
//  rppi_sobel_filter_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// // ----------------------------------------
// // CPU sobel_filter functions declaration 
// // ----------------------------------------
// /* Median Filter of the image using contrast stretch technique.
// param srcPtr [in] input image
// *param[in] srcSize dimensions of the image
// *param[out] dstPtr output image
// *returns a  RppStatus enumeration. 
// *retval RPP_SUCCESS : No error succesful completion
// *retval RPP_ERROR : Error 
// */
// RppStatus
//  rppi_sobel_filter_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
// RppStatus
//  rppi_sobel_filter_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU box_filter functions declaration 
// ----------------------------------------
/* Applies box filtering to the input image.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] kernelSize size of filter which uses the neighbouring pixels value  for filtering.
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
RppStatus
 rppi_box_filter_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU box_filter functions declaration 
// ----------------------------------------
/* Applies box filtering to the input image.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] kernelSize size of filter which uses the neighbouring pixels value  for filtering.
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_box_filter_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_box_filter_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// GPU bilateral_filter functions declaration 
// ----------------------------------------
/* Apllies bilateral filtering to the input image.
*param[in] srcPtr1 input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] filterSize size of filter which uses the neighbouring pixels value  for filtering.
*param[in] sigmaI filter sigma value in color space and value should be between 0 and 20
*param[in] sigmaS filter sigma value in coordinate space and value should be between 0 and 20
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error
*/
RppStatus
 rppi_bilateral_filter_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );


// ----------------------------------------
// CPU bilateral_filter functions declaration 
// ----------------------------------------
/* Apllies bilateral filtering to the input image.
param[in] srcPtr1 input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] filterSize size of filter which uses the neighbouring pixels value  for filtering.
*param[in] sigmaI filter sigma value in color space and value should be between 0 and 20
*param[in] sigmaS filter sigma value in coordinate space and value should be between 0 and 20
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
 rppi_bilateral_filter_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_bilateral_filter_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

#ifdef __cplusplus
}
#endif
#endif