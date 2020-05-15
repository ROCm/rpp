#ifndef RPPI_FUSED_FUNCTIONS_H
#define RPPI_FUSED_FUNCTIONS_H
 
#include "rppdefs.h"
#include "rpp.h"
#ifdef __cplusplus
extern "C" {
#endif // cpusplus

// ----------------------------------------
// GPU color_twist functions declaration 
// ----------------------------------------

RppStatus
 rppi_color_twist_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

// ----------------------------------------
// CPU color_twist functions declaration 
// ----------------------------------------

RppStatus
 rppi_color_twist_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus
 rppi_color_twist_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

RppStatus  
rppi_color_twist_f32_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );
RppStatus  
rppi_color_twist_f32_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

// ----------------------------------------
// GPU crop_mirror_normalize functions declaration 
// ----------------------------------------

RppStatus  
rppi_crop_mirror_normalize_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u crop_pos_x ,Rpp32u crop_pos_y ,Rpp32f mean ,Rpp32f stdDev ,Rpp32u mirrorFlag ,Rpp32u outputFormatToggle ,rppHandle_t rppHandle );

RppStatus  
rppi_crop_mirror_normalize_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *crop_pos_x ,Rpp32u *crop_pos_y ,Rpp32f *mean ,Rpp32f *stdDev ,Rpp32u *mirrorFlag ,Rpp32u outputFormatToggle ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

RppStatus  
rppi_crop_mirror_normalize_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u crop_pos_x ,Rpp32u crop_pos_y ,Rpp32f mean ,Rpp32f stdDev ,Rpp32u mirrorFlag ,Rpp32u outputFormatToggle ,rppHandle_t rppHandle );

RppStatus  
rppi_crop_mirror_normalize_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize , RppiSize maxDstSize ,Rpp32u *crop_pos_x, Rpp32u *crop_pos_y,Rpp32f *mean, Rpp32f* std_dev,Rpp32u *mirrorFlag ,Rpp32u outputFormatToggle ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

RppStatus  
rppi_crop_mirror_normalize_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize , RppiSize maxDstSize ,Rpp32u *crop_pos_x, Rpp32u *crop_pos_y,Rpp32f *mean, Rpp32f* std_dev,Rpp32u *mirrorFlag ,Rpp32u outputFormatToggle ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

RppStatus  
rppi_crop_mirror_normalize_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u crop_pos_x ,Rpp32u crop_pos_y ,Rpp32f mean ,Rpp32f stdDev ,Rpp32u mirrorFlag ,Rpp32u outputFormatToggle ,rppHandle_t rppHandle );

//RppStatus  
//rppi_crop_mirror_normalize_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize , Rpp32f *mean, Rpp32f* std_dev,RppiSize maxDstSize ,Rpp32u *crop_pos_x ,Rpp32u *crop_pos_y ,Rpp32u *mirrorFlag ,Rpp32u outputFormatToggle ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

// ----------------------------------------
// CPU crop_mirror_normalize functions declaration 
// ----------------------------------------

RppStatus  
rppi_crop_mirror_normalize_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u crop_pos_x ,Rpp32u crop_pos_y ,Rpp32f mean ,Rpp32f stdDev ,Rpp32u mirrorFlag ,Rpp32u outputFormatToggle ,rppHandle_t rppHandle );

RppStatus  
rppi_crop_mirror_normalize_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *crop_pos_x ,Rpp32u *crop_pos_y ,Rpp32f *mean ,Rpp32f *stdDev ,Rpp32u *mirrorFlag ,Rpp32u outputFormatToggle ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

RppStatus  
rppi_crop_mirror_normalize_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u crop_pos_x ,Rpp32u crop_pos_y ,Rpp32f mean ,Rpp32f stdDev ,Rpp32u mirrorFlag ,Rpp32u outputFormatToggle ,rppHandle_t rppHandle );

RppStatus  
rppi_crop_mirror_normalize_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *crop_pos_x ,Rpp32u *crop_pos_y ,Rpp32f *mean ,Rpp32f *stdDev ,Rpp32u *mirrorFlag ,Rpp32u outputFormatToggle ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

RppStatus  
rppi_crop_mirror_normalize_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u crop_pos_x ,Rpp32u crop_pos_y ,Rpp32f mean ,Rpp32f stdDev ,Rpp32u mirrorFlag ,Rpp32u outputFormatToggle ,rppHandle_t rppHandle );

RppStatus  
rppi_crop_mirror_normalize_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *crop_pos_x ,Rpp32u *crop_pos_y ,Rpp32f *mean ,Rpp32f *stdDev ,Rpp32u *mirrorFlag ,Rpp32u outputFormatToggle ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

RppStatus  
rppi_crop_mirror_normalize_f32_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u crop_pos_x ,Rpp32u crop_pos_y ,Rpp32f mean ,Rpp32f stdDev ,Rpp32u mirrorFlag ,Rpp32u outputFormatToggle ,rppHandle_t rppHandle );

RppStatus  
rppi_crop_mirror_normalize_f32_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *crop_pos_x ,Rpp32u *crop_pos_y ,Rpp32f *mean ,Rpp32f *stdDev ,Rpp32u *mirrorFlag ,Rpp32u outputFormatToggle ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

RppStatus  
rppi_crop_mirror_normalize_f32_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u crop_pos_x ,Rpp32u crop_pos_y ,Rpp32f mean ,Rpp32f stdDev ,Rpp32u mirrorFlag ,Rpp32u outputFormatToggle ,rppHandle_t rppHandle );

RppStatus  
rppi_crop_mirror_normalize_f32_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *crop_pos_x ,Rpp32u *crop_pos_y ,Rpp32f *mean ,Rpp32f *stdDev ,Rpp32u *mirrorFlag ,Rpp32u outputFormatToggle ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

RppStatus  
rppi_crop_mirror_normalize_f32_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u crop_pos_x ,Rpp32u crop_pos_y ,Rpp32f mean ,Rpp32f stdDev ,Rpp32u mirrorFlag ,Rpp32u outputFormatToggle ,rppHandle_t rppHandle );

RppStatus  
rppi_crop_mirror_normalize_f32_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *crop_pos_x ,Rpp32u *crop_pos_y ,Rpp32f *mean ,Rpp32f *stdDev ,Rpp32u *mirrorFlag ,Rpp32u outputFormatToggle ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

// ----------------------------------------
// GPU crop functions declaration 
// ----------------------------------------

// CROP function used for RALI
// Expects one to give the GOOD crops, i.e can be croppable from Image
// Does not take care care of out of boundary cases
// PLN3 variation has not been given as it is not been used in RALI

RppStatus  
rppi_crop_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize , RppiSize maxDstSize ,Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u nbatchSize ,rppHandle_t rppHandle );

RppStatus  
rppi_crop_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize , RppiSize maxDstSize ,Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u nbatchSize , rppHandle_t rppHandle );

// ----------------------------------------
// CPU crop functions declaration 
// ----------------------------------------

// CROP function used for RALI
// Expects one to give the GOOD crops, i.e can be croppable from Image
// Does not take care care of out of boundary cases
// PLN3 variation has not been given as it is not been used in RALI

RppStatus  
rppi_crop_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *crop_pos_x ,Rpp32u *crop_pos_y, Rpp32u nbatchSize ,rppHandle_t rppHandle );

RppStatus  
rppi_crop_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *crop_pos_x ,Rpp32u *crop_pos_y, Rpp32u nbatchSize ,rppHandle_t rppHandle );

RppStatus  
rppi_crop_f32_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *crop_pos_x ,Rpp32u *crop_pos_y, Rpp32u nbatchSize ,rppHandle_t rppHandle );

RppStatus  
rppi_crop_f32_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *crop_pos_x ,Rpp32u *crop_pos_y ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

// ----------------------------------------
// GPU resize_crop_mirror functions declaration 
// ----------------------------------------

RppStatus  
rppi_resize_crop_mirror_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u *mirrorFlag ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

RppStatus  
rppi_resize_crop_mirror_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u *mirrorFlag ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

RppStatus  
rppi_resize_crop_mirror_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u *mirrorFlag ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

// ----------------------------------------
// CPU resize_crop_mirror functions declaration 
// ----------------------------------------

RppStatus  
rppi_resize_crop_mirror_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u *mirrorFlag ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

RppStatus  
rppi_resize_crop_mirror_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u *mirrorFlag ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

RppStatus  
rppi_resize_crop_mirror_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u *mirrorFlag ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

RppStatus  
rppi_resize_crop_mirror_f32_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u *mirrorFlag ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

RppStatus  
rppi_resize_crop_mirror_f32_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u *mirrorFlag ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

RppStatus  
rppi_resize_crop_mirror_f32_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u *mirrorFlag ,Rpp32u nbatchSize ,rppHandle_t rppHandle );

#ifdef __cplusplus
}
#endif
#endif // RPPI_FUSED_FUNCTIONS_H