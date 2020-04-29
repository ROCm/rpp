#include <rppi_logical_operations.h>
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

#include "cpu/host_logical_operations.hpp" 

RppStatus  
rppi_exclusive_OR_u8_pln1_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip(
			static_cast<Rpp8u *>(srcPtr1),
			static_cast<Rpp8u *>(srcPtr2),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln1_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln1_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln1_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln1_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln1_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln1_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln3_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip(
			static_cast<Rpp8u *>(srcPtr1),
			static_cast<Rpp8u *>(srcPtr2),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln3_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln3_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln3_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln3_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln3_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln3_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pkd3_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip(
			static_cast<Rpp8u *>(srcPtr1),
			static_cast<Rpp8u *>(srcPtr2),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pkd3_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		exclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		exclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln1_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 
	exclusive_OR_host(
			static_cast<Rpp8u *>(srcPtr1),
			static_cast<Rpp8u *>(srcPtr2),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln1_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln1_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln1_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln1_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln1_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln1_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln1_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln3_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 
	exclusive_OR_host(
			static_cast<Rpp8u *>(srcPtr1),
			static_cast<Rpp8u *>(srcPtr2),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pln3_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln3_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln3_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln3_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln3_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln3_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pkd3_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 
	exclusive_OR_host(
			static_cast<Rpp8u *>(srcPtr1),
			static_cast<Rpp8u *>(srcPtr2),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PACKED, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_exclusive_OR_u8_pkd3_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pkd3_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pkd3_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pkd3_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pkd3_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pkd3_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pkd3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_exclusive_OR_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	exclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln1_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip(
			static_cast<Rpp8u *>(srcPtr1),
			static_cast<Rpp8u *>(srcPtr2),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln1_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln1_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln1_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln1_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln1_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln1_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln3_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip(
			static_cast<Rpp8u *>(srcPtr1),
			static_cast<Rpp8u *>(srcPtr2),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln3_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln3_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln3_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln3_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln3_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln3_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pkd3_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip(
			static_cast<Rpp8u *>(srcPtr1),
			static_cast<Rpp8u *>(srcPtr2),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pkd3_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		inclusive_OR_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		inclusive_OR_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln1_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 
	inclusive_OR_host(
			static_cast<Rpp8u *>(srcPtr1),
			static_cast<Rpp8u *>(srcPtr2),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln1_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln1_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln1_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln1_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln1_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln1_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln1_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln3_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 
	inclusive_OR_host(
			static_cast<Rpp8u *>(srcPtr1),
			static_cast<Rpp8u *>(srcPtr2),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pln3_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln3_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln3_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln3_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln3_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln3_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pkd3_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 
	inclusive_OR_host(
			static_cast<Rpp8u *>(srcPtr1),
			static_cast<Rpp8u *>(srcPtr2),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PACKED, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_inclusive_OR_u8_pkd3_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pkd3_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pkd3_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pkd3_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pkd3_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pkd3_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pkd3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_inclusive_OR_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	inclusive_OR_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln1_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip(
			static_cast<Rpp8u *>(srcPtr1),
			static_cast<Rpp8u *>(srcPtr2),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln1_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln1_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln1_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln1_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln1_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln1_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln3_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip(
			static_cast<Rpp8u *>(srcPtr1),
			static_cast<Rpp8u *>(srcPtr2),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln3_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln3_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln3_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln3_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln3_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln3_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pkd3_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip(
			static_cast<Rpp8u *>(srcPtr1),
			static_cast<Rpp8u *>(srcPtr2),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pkd3_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		bitwise_AND_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_AND_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln1_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 
	bitwise_AND_host(
			static_cast<Rpp8u *>(srcPtr1),
			static_cast<Rpp8u *>(srcPtr2),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln1_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln1_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln1_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln1_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln1_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln1_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln1_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln3_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 
	bitwise_AND_host(
			static_cast<Rpp8u *>(srcPtr1),
			static_cast<Rpp8u *>(srcPtr2),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pln3_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln3_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln3_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln3_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln3_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln3_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pkd3_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 
	bitwise_AND_host(
			static_cast<Rpp8u *>(srcPtr1),
			static_cast<Rpp8u *>(srcPtr2),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PACKED, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_AND_u8_pkd3_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pkd3_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pkd3_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pkd3_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pkd3_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pkd3_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pkd3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_AND_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_AND_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr1),
		static_cast<Rpp8u*>(srcPtr2),
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
rppi_bitwise_NOT_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip(
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
rppi_bitwise_NOT_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip(
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
rppi_bitwise_NOT_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip(
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
rppi_bitwise_NOT_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		bitwise_NOT_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		bitwise_NOT_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 
	bitwise_NOT_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 
	bitwise_NOT_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 
	bitwise_NOT_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PACKED, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_bitwise_NOT_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	bitwise_NOT_host_batch<Rpp8u>(
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
rppi_bitwise_NOT_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	bitwise_NOT_host_batch<Rpp8u>(
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