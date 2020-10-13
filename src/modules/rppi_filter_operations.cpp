#include <rppi_filter_operations.h>
#include <rppdefs.h>
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

#include "cpu/host_filter_operations.hpp" 

RppStatus  
rppi_gaussian_filter_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_filter_hip_batch(
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
rppi_gaussian_filter_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 
	gaussian_filter_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 
	gaussian_filter_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 
	gaussian_filter_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PACKED, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_filter_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	gaussian_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_median_filter_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_median_filter_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		median_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		median_filter_hip_batch(
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
rppi_nonlinear_filter_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_nonlinear_filter_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	median_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}
RppStatus  
rppi_non_max_suppression_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			kernelSize,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		non_max_suppression_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		non_max_suppression_hip_batch(
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
rppi_non_max_suppression_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 
	non_max_suppression_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 
	non_max_suppression_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 
	non_max_suppression_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			kernelSize,
			RPPI_CHN_PACKED, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_non_max_suppression_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	non_max_suppression_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		sobel_filter_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			sobleType,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			sobleType,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		sobel_filter_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			sobleType,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			sobleType,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		sobel_filter_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			sobleType,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			sobleType,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		sobel_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		sobel_filter_hip_batch(
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
rppi_sobel_filter_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,rppHandle_t rppHandle )
{ 
	sobel_filter_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			sobleType,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		sobleType,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		sobleType,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		sobleType,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		sobleType,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		sobleType,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		sobleType,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		sobleType,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		sobleType,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		sobleType,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,rppHandle_t rppHandle )
{ 
	sobel_filter_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			sobleType,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		sobleType,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		sobleType,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		sobleType,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		sobleType,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		sobleType,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		sobleType,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		sobleType,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		sobleType,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		sobleType,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,rppHandle_t rppHandle )
{ 
	sobel_filter_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			sobleType,
			RPPI_CHN_PACKED, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		sobleType,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		sobleType,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		sobleType,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		sobleType,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		sobleType,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		sobleType,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (sobleType, rpp::deref(rppHandle), paramIndex++);
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		sobleType,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		sobleType,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_sobel_filter_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *sobleType ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	sobel_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		sobleType,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		box_filter_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		box_filter_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		box_filter_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			kernelSize,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		box_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		box_filter_hip_batch(
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
rppi_box_filter_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 
	box_filter_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 
	box_filter_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 
	box_filter_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			kernelSize,
			RPPI_CHN_PACKED, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_box_filter_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	box_filter_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		kernelSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_bilateral_filter_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
			sigmaI,
			sigmaS,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			kernelSize,
			sigmaI,
			sigmaS,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bilateral_filter_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
			sigmaI,
			sigmaS,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			kernelSize,
			sigmaI,
			sigmaS,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bilateral_filter_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
			sigmaI,
			sigmaS,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			kernelSize,
			sigmaI,
			sigmaS,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bilateral_filter_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
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
rppi_bilateral_filter_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		bilateral_filter_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bilateral_filter_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

// RppStatus  
// rppi_bilateral_filter_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,rppHandle_t rppHandle )
// { 
// 	bilateral_filter_host(
// 			static_cast<Rpp8u *>(srcPtr),
// 			 srcSize,
// 			static_cast<Rpp8u *>(dstPtr),
// 			kernelSize,
// 			sigmaI,
// 			sigmaS,
// 			RPPI_CHN_PLANAR, 1 
// 			);


// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		kernelSize,
// 		sigmaI,
// 		sigmaS,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		kernelSize,
// 		sigmaI,
// 		sigmaS,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		kernelSize,
// 		sigmaI,
// 		sigmaS,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		kernelSize,
// 		sigmaI,
// 		sigmaS,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		kernelSize,
// 		sigmaI,
// 		sigmaS,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		kernelSize,
// 		sigmaI,
// 		sigmaS,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		kernelSize,
// 		sigmaI,
// 		sigmaS,
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		kernelSize,
// 		sigmaI,
// 		sigmaS,
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		kernelSize,
// 		sigmaI,
// 		sigmaS,
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,rppHandle_t rppHandle )
// { 
// 	bilateral_filter_host(
// 			static_cast<Rpp8u *>(srcPtr),
// 			 srcSize,
// 			static_cast<Rpp8u *>(dstPtr),
// 			kernelSize,
// 			sigmaI,
// 			sigmaS,
// 			RPPI_CHN_PLANAR, 1 
// 			);


// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		kernelSize,
// 		sigmaI,
// 		sigmaS,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		kernelSize,
// 		sigmaI,
// 		sigmaS,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		kernelSize,
// 		sigmaI,
// 		sigmaS,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		kernelSize,
// 		sigmaI,
// 		sigmaS,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		kernelSize,
// 		sigmaI,
// 		sigmaS,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		kernelSize,
// 		sigmaI,
// 		sigmaS,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		kernelSize,
// 		sigmaI,
// 		sigmaS,
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		kernelSize,
// 		sigmaI,
// 		sigmaS,
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		kernelSize,
// 		sigmaI,
// 		sigmaS,
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,rppHandle_t rppHandle )
// { 
// 	bilateral_filter_host(
// 			static_cast<Rpp8u *>(srcPtr),
// 			 srcSize,
// 			static_cast<Rpp8u *>(dstPtr),
// 			kernelSize,
// 			sigmaI,
// 			sigmaS,
// 			RPPI_CHN_PACKED, 3 
// 			);


// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		kernelSize,
// 		sigmaI,
// 		sigmaS,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		kernelSize,
// 		sigmaI,
// 		sigmaS,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		kernelSize,
// 		sigmaI,
// 		sigmaS,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		kernelSize,
// 		sigmaI,
// 		sigmaS,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		kernelSize,
// 		sigmaI,
// 		sigmaS,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		kernelSize,
// 		sigmaI,
// 		sigmaS,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u kernelSize ,Rpp64f sigmaI ,Rpp64f sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		kernelSize,
// 		sigmaI,
// 		sigmaS,
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		kernelSize,
// 		sigmaI,
// 		sigmaS,
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_bilateral_filter_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *kernelSize ,Rpp64f *sigmaI ,Rpp64f *sigmaS ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	bilateral_filter_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		kernelSize,
// 		sigmaI,
// 		sigmaS,
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// ********************************** custom convolution ***************************************

RppStatus
rppi_custom_convolution_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppPtr_t kernel,RppiSize kernelSize)
{

 	 validate_image_size(srcSize);
	 custom_convolution_host<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr), 
		srcSize,
		static_cast<Rpp8u*>(dstPtr),  
		static_cast<Rpp32f*>(kernel), 
		kernelSize,
		RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_custom_convolution_u8_pln1_batchPD_host(RppPtr_t srcPtr,RppiSize *srcSize,RppiSize maxSrcSize,RppPtr_t dstPtr,RppPtr_t kernel,RppiSize *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle)
{
 	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle)); 
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	custom_convolution_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr), 
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		static_cast<Rpp32f*>(kernel), 
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_custom_convolution_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppPtr_t kernel,RppiSize kernelSize)
{

 	 validate_image_size(srcSize);
	 custom_convolution_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			static_cast<Rpp32f*>(kernel), 
			kernelSize,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_custom_convolution_u8_pln3_batchPD_host(RppPtr_t srcPtr,RppiSize *srcSize,RppiSize maxSrcSize,RppPtr_t dstPtr,RppPtr_t kernel,RppiSize *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle)
{
 	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle)); 
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	custom_convolution_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr), 
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		static_cast<Rpp32f*>(kernel), 
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_custom_convolution_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppPtr_t kernel,RppiSize kernelSize)
{

 	 validate_image_size(srcSize);
	 custom_convolution_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr),  
			static_cast<Rpp32f*>(kernel), 
			kernelSize,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_custom_convolution_u8_pkd3_batchPD_host(RppPtr_t srcPtr,RppiSize *srcSize,RppiSize maxSrcSize,RppPtr_t dstPtr,RppPtr_t kernel,RppiSize *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle)
{
 	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle)); 
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	custom_convolution_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr), 
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		static_cast<Rpp32f*>(kernel), 
		kernelSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}