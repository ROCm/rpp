#include <rppi_geometry_transforms.h>
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

#include "cpu/host_geometry_transforms.hpp" 


RppStatus  
rppi_lens_correction_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		lens_correction_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			strength,
			zoom,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			strength,
			zoom,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		lens_correction_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			strength,
			zoom,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			strength,
			zoom,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		lens_correction_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			strength,
			zoom,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			strength,
			zoom,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		lens_correction_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		lens_correction_hip_batch(
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
rppi_lens_correction_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,rppHandle_t rppHandle )
{ 
	lens_correction_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			strength,
			zoom,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		strength,
		zoom,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		strength,
		zoom,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		strength,
		zoom,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		strength,
		zoom,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		strength,
		zoom,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		strength,
		zoom,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		strength,
		zoom,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		strength,
		zoom,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		strength,
		zoom,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,rppHandle_t rppHandle )
{ 
	lens_correction_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			strength,
			zoom,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		strength,
		zoom,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		strength,
		zoom,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		strength,
		zoom,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		strength,
		zoom,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		strength,
		zoom,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		strength,
		zoom,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		strength,
		zoom,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		strength,
		zoom,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		strength,
		zoom,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,rppHandle_t rppHandle )
{ 
	lens_correction_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			strength,
			zoom,
			RPPI_CHN_PACKED, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		strength,
		zoom,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		strength,
		zoom,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		strength,
		zoom,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		strength,
		zoom,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		strength,
		zoom,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		strength,
		zoom,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f strength ,Rpp32f zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (strength, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (zoom, rpp::deref(rppHandle), paramIndex++);
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		strength,
		zoom,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		strength,
		zoom,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_lens_correction_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *strength ,Rpp32f *zoom ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	lens_correction_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		strength,
		zoom,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}
RppStatus  
rppi_fisheye_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		fisheye_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		fisheye_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		fisheye_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		fisheye_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fisheye_hip_batch(
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
rppi_fisheye_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 
	fisheye_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 
	fisheye_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 
	fisheye_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PACKED, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fisheye_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	fisheye_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}
RppStatus  
rppi_flip_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		flip_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			flipAxis,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			flipAxis,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		flip_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			flipAxis,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			flipAxis,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		flip_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			flipAxis,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			flipAxis,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
		//std::cerr << "starting - coming till here" << std::endl;

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
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

			//std::cerr << "End - coming till here" << std::endl;


#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		flip_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		flip_hip_batch(
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
rppi_flip_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,rppHandle_t rppHandle )
{ 
	flip_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			flipAxis,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	flip_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		flipAxis,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	flip_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		flipAxis,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	flip_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		flipAxis,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	flip_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		flipAxis,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	flip_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		flipAxis,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	flip_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		flipAxis,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	flip_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		flipAxis,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	flip_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		flipAxis,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	flip_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		flipAxis,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,rppHandle_t rppHandle )
{ 
	flip_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			flipAxis,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	flip_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		flipAxis,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	flip_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		flipAxis,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	flip_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		flipAxis,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	flip_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		flipAxis,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	flip_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		flipAxis,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	flip_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		flipAxis,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	flip_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		flipAxis,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	flip_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		flipAxis,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	flip_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		flipAxis,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,rppHandle_t rppHandle )
{ 
	flip_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			flipAxis,
			RPPI_CHN_PACKED, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	flip_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		flipAxis,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	flip_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		flipAxis,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	flip_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		flipAxis,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	flip_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		flipAxis,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	flip_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		flipAxis,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	flip_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		flipAxis,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (flipAxis, rpp::deref(rppHandle), paramIndex++);
	flip_host_batch<Rpp8u>(
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
rppi_flip_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	flip_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		flipAxis,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	flip_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		flipAxis,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_flip_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *flipAxis ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	flip_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		flipAxis,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}
RppStatus  
rppi_resize_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		resize_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			 dstSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		resize_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			 dstSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		resize_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			 dstSize,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		resize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_hip_batch(
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
rppi_resize_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,rppHandle_t rppHandle )
{ 
	resize_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,rppHandle_t rppHandle )
{ 
	resize_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,rppHandle_t rppHandle )
{ 
	resize_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			RPPI_CHN_PACKED, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	resize_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		rotate_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			 dstSize,
			angleDeg,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			angleDeg,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		rotate_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			 dstSize,
			angleDeg,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			angleDeg,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		rotate_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			 dstSize,
			angleDeg,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			angleDeg,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		rotate_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		rotate_hip_batch(
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
rppi_rotate_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,rppHandle_t rppHandle )
{ 
	rotate_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			angleDeg,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		angleDeg,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		angleDeg,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		angleDeg,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		angleDeg,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		angleDeg,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		angleDeg,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		angleDeg,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		angleDeg,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		angleDeg,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,rppHandle_t rppHandle )
{ 
	rotate_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			angleDeg,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		angleDeg,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		angleDeg,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		angleDeg,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		angleDeg,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		angleDeg,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		angleDeg,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		angleDeg,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		angleDeg,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		angleDeg,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,rppHandle_t rppHandle )
{ 
	rotate_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			angleDeg,
			RPPI_CHN_PACKED, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		angleDeg,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		angleDeg,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *angleDeg ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		angleDeg,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		angleDeg,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		angleDeg,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *angleDeg ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		angleDeg,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	copy_param_float (angleDeg, rpp::deref(rppHandle), paramIndex++);
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		angleDeg,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		angleDeg,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_rotate_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *angleDeg ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	rotate_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		angleDeg,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		scale_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			 dstSize,
			percentage,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			percentage,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		scale_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			 dstSize,
			percentage,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			percentage,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		scale_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			 dstSize,
			percentage,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			percentage,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		scale_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		scale_hip_batch(
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
rppi_scale_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,rppHandle_t rppHandle )
{ 
	scale_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			percentage,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		percentage,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		percentage,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		percentage,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		percentage,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		percentage,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		percentage,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		percentage,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		percentage,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		percentage,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,rppHandle_t rppHandle )
{ 
	scale_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			percentage,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		percentage,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		percentage,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		percentage,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		percentage,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		percentage,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		percentage,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		percentage,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		percentage,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		percentage,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,rppHandle_t rppHandle )
{ 
	scale_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			percentage,
			RPPI_CHN_PACKED, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		percentage,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		percentage,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *percentage ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		percentage,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		percentage,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		percentage,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *percentage ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		percentage,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	copy_param_float (percentage, rpp::deref(rppHandle), paramIndex++);
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		percentage,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		percentage,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_scale_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *percentage ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	scale_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		percentage,
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		resize_crop_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			 dstSize,
			xRoiBegin,
			xRoiEnd,
			yRoiBegin,
			yRoiEnd,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		resize_crop_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			xRoiBegin,
			xRoiEnd,
			yRoiBegin,
			yRoiEnd,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (xRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (xRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiEnd, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		resize_crop_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_crop_hip_batch(
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
rppi_resize_crop_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (xRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (xRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiEnd, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		resize_crop_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_crop_hip_batch(
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
rppi_resize_crop_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (xRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (xRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiEnd, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		resize_crop_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_crop_hip_batch(
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
rppi_resize_crop_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (xRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (xRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiEnd, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		resize_crop_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_crop_hip_batch(
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
rppi_resize_crop_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (xRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (xRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiEnd, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		resize_crop_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_crop_hip_batch(
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
rppi_resize_crop_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (xRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (xRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiEnd, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		resize_crop_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_crop_hip_batch(
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
rppi_resize_crop_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		resize_crop_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			 dstSize,
			xRoiBegin,
			xRoiEnd,
			yRoiBegin,
			yRoiEnd,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		resize_crop_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			xRoiBegin,
			xRoiEnd,
			yRoiBegin,
			yRoiEnd,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (xRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (xRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiEnd, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		resize_crop_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_crop_hip_batch(
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
rppi_resize_crop_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (xRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (xRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiEnd, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		resize_crop_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_crop_hip_batch(
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
rppi_resize_crop_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (xRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (xRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiEnd, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		resize_crop_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_crop_hip_batch(
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
rppi_resize_crop_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (xRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (xRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiEnd, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		resize_crop_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_crop_hip_batch(
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
rppi_resize_crop_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (xRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (xRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiEnd, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		resize_crop_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_crop_hip_batch(
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
rppi_resize_crop_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (xRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (xRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiEnd, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		resize_crop_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_crop_hip_batch(
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
rppi_resize_crop_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		resize_crop_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			 dstSize,
			xRoiBegin,
			xRoiEnd,
			yRoiBegin,
			yRoiEnd,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		resize_crop_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			xRoiBegin,
			xRoiEnd,
			yRoiBegin,
			yRoiEnd,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (xRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (xRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiEnd, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		resize_crop_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_crop_hip_batch(
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
rppi_resize_crop_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (xRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (xRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiEnd, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		resize_crop_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_crop_hip_batch(
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
rppi_resize_crop_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (xRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (xRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiEnd, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		resize_crop_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_crop_hip_batch(
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
rppi_resize_crop_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (xRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (xRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiEnd, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		resize_crop_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_crop_hip_batch(
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
rppi_resize_crop_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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

#ifdef OCL_COMPILE
	{
		resize_crop_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_crop_hip_batch(
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
rppi_resize_crop_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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

#ifdef OCL_COMPILE
	{
		resize_crop_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		resize_crop_hip_batch(
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
rppi_resize_crop_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,rppHandle_t rppHandle )
{ 
	resize_crop_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			xRoiBegin,
			xRoiEnd,
			yRoiBegin,
			yRoiEnd,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_uint (xRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (xRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiEnd, rpp::deref(rppHandle), paramIndex++);
	resize_crop_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[2].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[3].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	resize_crop_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		xRoiBegin,
		xRoiEnd,
		yRoiBegin,
		yRoiEnd,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_uint (xRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (xRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiEnd, rpp::deref(rppHandle), paramIndex++);
	resize_crop_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[2].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[3].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	resize_crop_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		xRoiBegin,
		xRoiEnd,
		yRoiBegin,
		yRoiEnd,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_uint (xRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (xRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiEnd, rpp::deref(rppHandle), paramIndex++);
	resize_crop_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[2].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[3].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	resize_crop_host_batch<Rpp8u>(
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
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,rppHandle_t rppHandle )
{ 
	resize_crop_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			xRoiBegin,
			xRoiEnd,
			yRoiBegin,
			yRoiEnd,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_uint (xRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (xRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiEnd, rpp::deref(rppHandle), paramIndex++);
	resize_crop_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[2].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[3].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	resize_crop_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		xRoiBegin,
		xRoiEnd,
		yRoiBegin,
		yRoiEnd,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_uint (xRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (xRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiEnd, rpp::deref(rppHandle), paramIndex++);
	resize_crop_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[2].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[3].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	resize_crop_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		xRoiBegin,
		xRoiEnd,
		yRoiBegin,
		yRoiEnd,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_uint (xRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (xRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiEnd, rpp::deref(rppHandle), paramIndex++);
	resize_crop_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[2].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[3].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	resize_crop_host_batch<Rpp8u>(
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
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,rppHandle_t rppHandle )
{ 
	resize_crop_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize,
			xRoiBegin,
			xRoiEnd,
			yRoiBegin,
			yRoiEnd,
			RPPI_CHN_PACKED, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_uint (xRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (xRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiEnd, rpp::deref(rppHandle), paramIndex++);
	resize_crop_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[2].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[3].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	resize_crop_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		xRoiBegin,
		xRoiEnd,
		yRoiBegin,
		yRoiEnd,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_uint (xRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (xRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiEnd, rpp::deref(rppHandle), paramIndex++);
	resize_crop_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[2].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[3].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	resize_crop_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		xRoiBegin,
		xRoiEnd,
		yRoiBegin,
		yRoiEnd,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32u xRoiBegin ,Rpp32u xRoiEnd ,Rpp32u yRoiBegin ,Rpp32u yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_param_uint (xRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (xRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (yRoiEnd, rpp::deref(rppHandle), paramIndex++);
	resize_crop_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[2].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[3].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_resize_crop_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32u *xRoiBegin ,Rpp32u *xRoiEnd ,Rpp32u *yRoiBegin ,Rpp32u *yRoiEnd ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	resize_crop_host_batch<Rpp8u>(
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
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		warp_affine_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			 dstSize, affineMatrix,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize, affineMatrix,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		warp_affine_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			 dstSize, affineMatrix,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize, affineMatrix,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		warp_affine_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			 dstSize, affineMatrix,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize, affineMatrix,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_affine_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_affine_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), affineMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}


RppStatus  
rppi_warp_affine_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,rppHandle_t rppHandle )
{ 
	warp_affine_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize, affineMatrix,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,rppHandle_t rppHandle )
{ 
	warp_affine_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize, affineMatrix,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,rppHandle_t rppHandle )
{ 
	warp_affine_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize, affineMatrix,
			RPPI_CHN_PACKED, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_affine_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *affineMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_affine_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		roiPoints, affineMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

//Warp-Perspective


RppStatus  
rppi_warp_perspective_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		warp_perspective_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			 dstSize, perspectiveMatrix,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize, perspectiveMatrix,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		warp_perspective_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			 dstSize, perspectiveMatrix,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize, perspectiveMatrix,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		warp_perspective_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			 dstSize, perspectiveMatrix,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize, perspectiveMatrix,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize (maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	get_dstBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		warp_perspective_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		warp_perspective_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle), perspectiveMatrix,
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}


RppStatus  
rppi_warp_perspective_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,rppHandle_t rppHandle )
{ 
	warp_perspective_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize, perspectiveMatrix,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,rppHandle_t rppHandle )
{ 
	warp_perspective_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize, perspectiveMatrix,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,rppHandle_t rppHandle )
{ 
	warp_perspective_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 dstSize, perspectiveMatrix,
			RPPI_CHN_PACKED, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_dstSize(dstSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		dstSize,
		roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_warp_perspective_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiSize *dstSize ,RppiSize maxDstSize ,Rpp32f *perspectiveMatrix ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	warp_perspective_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		dstSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
		roiPoints, perspectiveMatrix, 
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}