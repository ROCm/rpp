#include <rppi_fused_functions.h>
#include <rppdefs.h>
#include<iostream>
#include "rppi_validate.hpp"

#ifdef HIP_COMPILE
#include <hip/rpp_hip_common.hpp>
#include "hip/hip_declarations.hpp"

#elif defined(OCL_COMPILE)
#include <cl/rpp_cl_common.hpp>
#include "cl/cl_declarations.hpp"
#endif //backend
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
using namespace std::chrono; 

#include "cpu/host_fused_functions.hpp" 

RppStatus  
rppi_color_twist_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		color_twist_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			alpha,
			beta,
			hueShift,
			saturationFactor,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			alpha,
			beta,
			hueShift,
			saturationFactor,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		color_twist_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			alpha,
			beta,
			hueShift,
			saturationFactor,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			alpha,
			beta,
			hueShift,
			saturationFactor,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		color_twist_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			alpha,
			beta,
			hueShift,
			saturationFactor,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			alpha,
			beta,
			hueShift,
			saturationFactor,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,rppHandle_t rppHandle )
{ 
	color_twist_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			alpha,
			beta,
			hueShift,
			saturationFactor,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,rppHandle_t rppHandle )
{ 
	color_twist_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			alpha,
			beta,
			hueShift,
			saturationFactor,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,rppHandle_t rppHandle )
{ 
	color_twist_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			alpha,
			beta,
			hueShift,
			saturationFactor,
			RPPI_CHN_PACKED, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f alpha ,Rpp32f beta ,Rpp32f hueShift ,Rpp32f saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_color_twist_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *alpha ,Rpp32f *beta ,Rpp32f *hueShift ,Rpp32f *saturationFactor ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}



RppStatus  
rppi_crop_mirror_normalize_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u crop_pos_x ,Rpp32u crop_pos_y , Rpp32f mean, Rpp32f std_dev, Rpp32u mirrorFlag ,Rpp32u outputFormatToggle ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		crop_mirror_normalize_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			 dstSize,
			mean,
			std_dev,
			crop_pos_x,
			crop_pos_y,
			mirrorFlag,
			outputFormatToggle,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		crop_mirror_normalize_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			mean,
			std_dev,
			crop_pos_x,
			crop_pos_y,
			mirrorFlag,
			outputFormatToggle,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND
return RPP_SUCCESS;
}

RppStatus  
rppi_crop_mirror_normalize_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,
											RppiSize maxDstSize ,Rpp32u *crop_pos_x ,Rpp32u *crop_pos_y ,
											Rpp32f *mean, Rpp32f* std_dev, Rpp32u *mirrorFlag ,Rpp32u outputFormatToggle ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (crop_pos_x, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (crop_pos_y, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (mean, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (std_dev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (mirrorFlag, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (outputFormatToggle, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		crop_mirror_normalize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		crop_mirror_normalize_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_crop_mirror_normalize_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u crop_pos_x ,Rpp32u crop_pos_y ,Rpp32f mean, Rpp32f std_dev,Rpp32u mirrorFlag ,Rpp32u outputFormatToggle ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		crop_mirror_normalize_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			 dstSize,
			mean,
			std_dev,
			crop_pos_x,
			crop_pos_y,
			mirrorFlag,
			outputFormatToggle,
			RPPI_CHN_PLANAR, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		crop_mirror_normalize_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			mean,
			std_dev,
			crop_pos_x,
			crop_pos_y,
			mirrorFlag,
			outputFormatToggle,
			RPPI_CHN_PLANAR, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_crop_mirror_normalize_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize , RppiSize maxDstSize ,Rpp32u *crop_pos_x, Rpp32u *crop_pos_y,Rpp32f *mean, Rpp32f* std_dev,Rpp32u *mirrorFlag ,Rpp32u outputFormatToggle ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (crop_pos_x, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (crop_pos_y, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (mean, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (std_dev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (mirrorFlag, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (outputFormatToggle, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		crop_mirror_normalize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		crop_mirror_normalize_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_crop_mirror_normalize_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u crop_pos_x ,Rpp32u crop_pos_y ,Rpp32f mean, Rpp32f std_dev,Rpp32u mirrorFlag ,Rpp32u outputFormatToggle ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		crop_mirror_normalize_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			 dstSize,
			mean,
			std_dev,
			crop_pos_x,
			crop_pos_y,
			mirrorFlag,	
			outputFormatToggle,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		crop_mirror_normalize_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			mean,
			std_dev,
			crop_pos_x,
			crop_pos_y,
			mirrorFlag,
			outputFormatToggle,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_crop_mirror_normalize_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize , RppiSize maxDstSize ,Rpp32u *crop_pos_x, Rpp32u *crop_pos_y,Rpp32f *mean, Rpp32f* std_dev,Rpp32u *mirrorFlag ,Rpp32u outputFormatToggle ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (crop_pos_x, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (crop_pos_y, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (mean, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (std_dev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (mirrorFlag, rpp::deref(rppHandle), paramIndex++);
	//copy_param_uint (outputFormatToggle, rpp::deref(rppHandle), paramIndex++);

	
#ifdef OCL_COMPILE
	{
		crop_mirror_normalize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		crop_mirror_normalize_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}


RppStatus  
rppi_crop_mirror_normalize_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u crop_pos_x ,Rpp32u crop_pos_y ,Rpp32f mean ,Rpp32f stdDev ,Rpp32u mirrorFlag ,Rpp32u outputFormatToggle ,rppHandle_t rppHandle )
{ 
	crop_mirror_normalize_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			crop_pos_x,
			crop_pos_y,
			mean,
			stdDev,
			mirrorFlag,
			outputFormatToggle,
			RPPI_CHN_PLANAR, 1 
			);

	return RPP_SUCCESS;
}

RppStatus  
rppi_crop_mirror_normalize_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *crop_pos_x ,Rpp32u *crop_pos_y ,Rpp32f *mean ,Rpp32f *stdDev ,Rpp32u *mirrorFlag ,Rpp32u outputFormatToggle ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	crop_mirror_normalize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		crop_pos_x,
		crop_pos_y,
		mean,
		stdDev,
		mirrorFlag,
		outputFormatToggle,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_crop_mirror_normalize_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u crop_pos_x ,Rpp32u crop_pos_y ,Rpp32f mean ,Rpp32f stdDev ,Rpp32u mirrorFlag ,Rpp32u outputFormatToggle ,rppHandle_t rppHandle )
{ 
	crop_mirror_normalize_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			crop_pos_x,
			crop_pos_y,
			mean,
			stdDev,
			mirrorFlag,
			outputFormatToggle,
			RPPI_CHN_PLANAR, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_crop_mirror_normalize_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *crop_pos_x ,Rpp32u *crop_pos_y ,Rpp32f *mean ,Rpp32f *stdDev ,Rpp32u *mirrorFlag ,Rpp32u outputFormatToggle ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	crop_mirror_normalize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		crop_pos_x,
		crop_pos_y,
		mean,
		stdDev,
		mirrorFlag,
		outputFormatToggle,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);
	return RPP_SUCCESS;
}

RppStatus  
rppi_crop_mirror_normalize_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u crop_pos_x ,Rpp32u crop_pos_y ,Rpp32f mean ,Rpp32f stdDev ,Rpp32u mirrorFlag ,Rpp32u outputFormatToggle ,rppHandle_t rppHandle )
{ 
	crop_mirror_normalize_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			crop_pos_x,
			crop_pos_y,
			mean,
			stdDev,
			mirrorFlag,
			outputFormatToggle,
			RPPI_CHN_PACKED, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_crop_mirror_normalize_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *crop_pos_x ,Rpp32u *crop_pos_y ,Rpp32f *mean ,Rpp32f *stdDev ,Rpp32u *mirrorFlag ,Rpp32u outputFormatToggle ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	crop_mirror_normalize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		crop_pos_x,
		crop_pos_y,
		mean,
		stdDev,
		mirrorFlag,
		outputFormatToggle,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_crop_mirror_normalize_f32_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u crop_pos_x ,Rpp32u crop_pos_y ,Rpp32f mean ,Rpp32f stdDev ,Rpp32u mirrorFlag ,Rpp32u outputFormatToggle ,rppHandle_t rppHandle )
{ 
	crop_mirror_normalize_f32_host(
			static_cast<Rpp32f *>(srcPtr),
			 srcSize,
			static_cast<Rpp32f *>(dstPtr),
			 dstSize,
			crop_pos_x,
			crop_pos_y,
			mean,
			stdDev,
			mirrorFlag,
			outputFormatToggle,
			RPPI_CHN_PLANAR, 1 
			);

	return RPP_SUCCESS;
}

RppStatus  
rppi_crop_mirror_normalize_f32_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *crop_pos_x ,Rpp32u *crop_pos_y ,Rpp32f *mean ,Rpp32f *stdDev ,Rpp32u *mirrorFlag ,Rpp32u outputFormatToggle ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	crop_mirror_normalize_f32_host_batch(
		static_cast<Rpp32f *>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp32f *>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		crop_pos_x,
		crop_pos_y,
		mean,
		stdDev,
		mirrorFlag,
		outputFormatToggle,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_crop_mirror_normalize_f32_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u crop_pos_x ,Rpp32u crop_pos_y ,Rpp32f mean ,Rpp32f stdDev ,Rpp32u mirrorFlag ,Rpp32u outputFormatToggle ,rppHandle_t rppHandle )
{ 
	crop_mirror_normalize_f32_host(
			static_cast<Rpp32f *>(srcPtr),
			 srcSize,
			static_cast<Rpp32f *>(dstPtr),
			 dstSize,
			crop_pos_x,
			crop_pos_y,
			mean,
			stdDev,
			mirrorFlag,
			outputFormatToggle,
			RPPI_CHN_PLANAR, 3 
			);

	return RPP_SUCCESS;
}

RppStatus  
rppi_crop_mirror_normalize_f32_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *crop_pos_x ,Rpp32u *crop_pos_y ,Rpp32f *mean ,Rpp32f *stdDev ,Rpp32u *mirrorFlag ,Rpp32u outputFormatToggle ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	crop_mirror_normalize_f32_host_batch(
		static_cast<Rpp32f *>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp32f *>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		crop_pos_x,
		crop_pos_y,
		mean,
		stdDev,
		mirrorFlag,
		outputFormatToggle,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);
	return RPP_SUCCESS;
}

RppStatus  
rppi_crop_mirror_normalize_f32_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u crop_pos_x ,Rpp32u crop_pos_y ,Rpp32f mean ,Rpp32f stdDev ,Rpp32u mirrorFlag ,Rpp32u outputFormatToggle ,rppHandle_t rppHandle )
{ 
	crop_mirror_normalize_f32_host(
			static_cast<Rpp32f *>(srcPtr),
			 srcSize,
			static_cast<Rpp32f *>(dstPtr),
			 dstSize,
			crop_pos_x,
			crop_pos_y,
			mean,
			stdDev,
			mirrorFlag,
			outputFormatToggle,
			RPPI_CHN_PACKED, 3 
			);

	return RPP_SUCCESS;
}

RppStatus  
rppi_crop_mirror_normalize_f32_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *crop_pos_x ,Rpp32u *crop_pos_y ,Rpp32f *mean ,Rpp32f *stdDev ,Rpp32u *mirrorFlag ,Rpp32u outputFormatToggle ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	crop_mirror_normalize_f32_host_batch(
		static_cast<Rpp32f *>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp32f *>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		crop_pos_x,
		crop_pos_y,
		mean,
		stdDev,
		mirrorFlag,
		outputFormatToggle,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}


// CROP function used for RALI
// Expects one to give the GOOD crops, i.e can be croppable from Image
// Does not take care care of out of boundary cases

RppStatus  
rppi_crop_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize , RppiSize maxDstSize ,Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (crop_pos_x, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (crop_pos_y, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		crop_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		crop_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_crop_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize , RppiSize maxDstSize ,Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (crop_pos_x, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (crop_pos_y, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		crop_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		crop_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_crop_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *crop_pos_x ,Rpp32u *crop_pos_y, Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	crop_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		crop_pos_x,
		crop_pos_y,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);
	return RPP_SUCCESS;
}

RppStatus  
rppi_crop_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *crop_pos_x ,Rpp32u *crop_pos_y ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	crop_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		crop_pos_x,
		crop_pos_y,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);
	return RPP_SUCCESS;
}

RppStatus  
rppi_crop_f32_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *crop_pos_x ,Rpp32u *crop_pos_y, Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	crop_f32_host_batch(
		static_cast<Rpp32f*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp32f*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		crop_pos_x,
		crop_pos_y,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);
	return RPP_SUCCESS;
}

RppStatus  
rppi_crop_f32_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *crop_pos_x ,Rpp32u *crop_pos_y ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	crop_f32_host_batch(
		static_cast<Rpp32f*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp32f*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		crop_pos_x,
		crop_pos_y,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);
	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_mirror_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u *mirrorFlag ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	resize_crop_mirror_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		xRoiBegin,
		xRoiEnd,
		yRoiBegin,
		yRoiEnd,
		mirrorFlag,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_mirror_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u *mirrorFlag ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	resize_crop_mirror_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		xRoiBegin,
		xRoiEnd,
		yRoiBegin,
		yRoiEnd,
		mirrorFlag,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_mirror_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u *mirrorFlag ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	resize_crop_mirror_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		xRoiBegin,
		xRoiEnd,
		yRoiBegin,
		yRoiEnd,
		mirrorFlag,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_mirror_f32_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u *mirrorFlag ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	resize_crop_mirror_f32_host_batch(
		static_cast<Rpp32f*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp32f*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		xRoiBegin,
		xRoiEnd,
		yRoiBegin,
		yRoiEnd,
		mirrorFlag,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_mirror_f32_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u *mirrorFlag ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	resize_crop_mirror_f32_host_batch(
		static_cast<Rpp32f*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp32f*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		xRoiBegin,
		xRoiEnd,
		yRoiBegin,
		yRoiEnd,
		mirrorFlag,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_mirror_f32_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u *mirrorFlag ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	resize_crop_mirror_f32_host_batch(
		static_cast<Rpp32f*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp32f*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		xRoiBegin,
		xRoiEnd,
		yRoiBegin,
		yRoiEnd,
		mirrorFlag,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_mirror_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u *mirrorFlag ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (xRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (xRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (mirrorFlag, rpp::deref(rppHandle), paramIndex++);

	#ifdef OCL_COMPILE
		{
			resize_crop_mirror_cl_batch(
				static_cast<cl_mem>(srcPtr),
				static_cast<cl_mem>(dstPtr),
				rpp::deref(rppHandle),
				RPPI_CHN_PACKED, 3
			);
		}
	#elif defined (HIP_COMPILE)
		{
			resize_crop_mirror_hip_batch(
				static_cast<Rpp8u*>(srcPtr),
				static_cast<Rpp8u*>(dstPtr),
				rpp::deref(rppHandle),
				RPPI_CHN_PACKED, 3
			);
		}
	#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_mirror_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u *mirrorFlag ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_mirror_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u *mirrorFlag ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{
	return RPP_SUCCESS;
}
