#include <rppi_fused_functions.h>
#include <rppdefs.h>
#include <iostream>
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

RppStatus color_twist_helper(RppiChnFormat chn_format, Rpp32u num_of_channels,
							 RPPTensorDataType tensor_type,
							 RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta,
							 Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), num_of_channels, chn_format);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			chn_format, num_of_channels, tensor_type);
	}
#elif defined(HIP_COMPILE)
	{
		if (tensor_type == RPPTensorDataType::U8)
		{
			color_twist_hip_batch(
				static_cast<Rpp8u *>(srcPtr),
				static_cast<Rpp8u *>(dstPtr),
				rpp::deref(rppHandle),
				chn_format, num_of_channels, tensor_type);
		}
		else if (tensor_type == RPPTensorDataType::FP32)
		{
			color_twist_hip_batch(
				static_cast<Rpp32f *>(srcPtr),
				static_cast<Rpp32f *>(dstPtr),
				rpp::deref(rppHandle),
				chn_format, num_of_channels, tensor_type);
		}
		else if (tensor_type == RPPTensorDataType::FP16)
		{
			color_twist_hip_batch(
				static_cast<data_type_t *>(srcPtr),
				static_cast<data_type_t *>(dstPtr),
				rpp::deref(rppHandle),
				chn_format, num_of_channels, tensor_type);
		}
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, rppHandle_t rppHandle)
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
			RPPI_CHN_PLANAR, 1, rpp::deref(rppHandle));
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip(
			static_cast<Rpp8u *>(srcPtr),
			srcSize,
			static_cast<Rpp8u *>(dstPtr),
			alpha,
			beta,
			hueShift,
			saturationFactor,
			RPPI_CHN_PLANAR, 1, rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI roiPoints, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (color_twist_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8,
							   srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta,
							   hueShift, saturationFactor, nbatchSize, rppHandle));
}

RppStatus
rppi_color_twist_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, rppHandle_t rppHandle)
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
			RPPI_CHN_PLANAR, 1, rpp::deref(rppHandle));
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip(
			static_cast<Rpp8u *>(srcPtr),
			srcSize,
			static_cast<Rpp8u *>(dstPtr),
			alpha,
			beta,
			hueShift,
			saturationFactor,
			RPPI_CHN_PLANAR, 1, rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI roiPoints, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (color_twist_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8,
							   srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta,
							   hueShift, saturationFactor, nbatchSize, rppHandle));
}

RppStatus
rppi_color_twist_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, rppHandle_t rppHandle)
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
			RPPI_CHN_PACKED, 3, rpp::deref(rppHandle));
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip(
			static_cast<Rpp8u *>(srcPtr),
			srcSize,
			static_cast<Rpp8u *>(dstPtr),
			alpha,
			beta,
			hueShift,
			saturationFactor,
			RPPI_CHN_PACKED, 3, rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_ROI_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI roiPoints, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (color_twist_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8,
							   srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta,
							   hueShift, saturationFactor, nbatchSize, rppHandle));
}

RppStatus
rppi_color_twist_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		color_twist_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#elif defined(HIP_COMPILE)
	{
		color_twist_hip_batch(
			static_cast<Rpp8u *>(srcPtr),
			static_cast<Rpp8u *>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (color_twist_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16,
							   srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta,
							   hueShift, saturationFactor, nbatchSize, rppHandle));
}

RppStatus
rppi_color_twist_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (color_twist_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16,
							   srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta,
							   hueShift, saturationFactor, nbatchSize, rppHandle));
}

RppStatus
rppi_color_twist_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (color_twist_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16,
							   srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta,
							   hueShift, saturationFactor, nbatchSize, rppHandle));
}

RppStatus
rppi_color_twist_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (color_twist_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32,
							   srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta,
							   hueShift, saturationFactor, nbatchSize, rppHandle));
}

RppStatus
rppi_color_twist_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (color_twist_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32,
							   srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta,
							   hueShift, saturationFactor, nbatchSize, rppHandle));
}

RppStatus
rppi_color_twist_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (color_twist_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32,
							   srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta,
							   hueShift, saturationFactor, nbatchSize, rppHandle));
}

RppStatus
rppi_color_twist_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (color_twist_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8,
							   srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta,
							   hueShift, saturationFactor, nbatchSize, rppHandle));
}

RppStatus
rppi_color_twist_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (color_twist_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8,
							   srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta,
							   hueShift, saturationFactor, nbatchSize, rppHandle));
}

RppStatus
rppi_color_twist_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (color_twist_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8,
							   srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta,
							   hueShift, saturationFactor, nbatchSize, rppHandle));
}


RppStatus color_twist_host_helper(RppiChnFormat chn_format, Rpp32u num_of_channels,
								  RPPTensorDataType tensor_type,
								  RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta,
								  Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle,
								  Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

	if (tensor_type == RPPTensorDataType::U8)
	{
		color_twist_host_batch<Rpp8u>(
			static_cast<Rpp8u *>(srcPtr),
			srcSize,
			rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
			static_cast<Rpp8u *>(dstPtr),
			alpha,
			beta,
			hueShift,
			saturationFactor,
			rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
			outputFormatToggle,
			rpp::deref(rppHandle).GetBatchSize(),
			chn_format, num_of_channels);
	}
	else if (tensor_type == RPPTensorDataType::FP16)
	{
		color_twist_f16_host_batch<Rpp16f>(
			static_cast<Rpp16f *>(srcPtr),
			srcSize,
			rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
			static_cast<Rpp16f *>(dstPtr),
			alpha,
			beta,
			hueShift,
			saturationFactor,
			rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
			outputFormatToggle,
			rpp::deref(rppHandle).GetBatchSize(),
			chn_format, num_of_channels);
	}
	else if (tensor_type == RPPTensorDataType::FP32)
	{
		color_twist_f32_host_batch<Rpp32f>(
			static_cast<Rpp32f *>(srcPtr),
			srcSize,
			rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
			static_cast<Rpp32f *>(dstPtr),
			alpha,
			beta,
			hueShift,
			saturationFactor,
			rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
			outputFormatToggle,
			rpp::deref(rppHandle).GetBatchSize(),
			chn_format, num_of_channels);
	}
	else if (tensor_type == RPPTensorDataType::I8)
	{
		color_twist_i8_host_batch<Rpp8s>(
			static_cast<Rpp8s *>(srcPtr),
			srcSize,
			rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
			static_cast<Rpp8s *>(dstPtr),
			alpha,
			beta,
			hueShift,
			saturationFactor,
			rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
			outputFormatToggle,
			rpp::deref(rppHandle).GetBatchSize(),
			chn_format, num_of_channels);
	}

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, rppHandle_t rppHandle)
{
	color_twist_host(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		static_cast<Rpp8u *>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		RPPI_CHN_PLANAR, 1);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI roiPoints, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
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
		static_cast<Rpp8u *>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u *>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u *>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (color_twist_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta, hueShift, saturationFactor, 0, nbatchSize, rppHandle));
}

RppStatus
rppi_color_twist_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u *>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u *>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u *>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u *>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u *>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u *>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, rppHandle_t rppHandle)
{
	color_twist_host(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		static_cast<Rpp8u *>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		RPPI_CHN_PLANAR, 1);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI roiPoints, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
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
		static_cast<Rpp8u *>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u *>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u *>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (color_twist_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta, hueShift, saturationFactor, 0, nbatchSize, rppHandle));
}

RppStatus
rppi_color_twist_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u *>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u *>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u *>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u *>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u *>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u *>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, rppHandle_t rppHandle)
{
	color_twist_host(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		static_cast<Rpp8u *>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		RPPI_CHN_PACKED, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_ROI_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI roiPoints, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchSS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchDS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchPS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchSD_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
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
		static_cast<Rpp8u *>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u *>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchDD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u *>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (color_twist_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta, hueShift, saturationFactor, 0, nbatchSize, rppHandle));
}

RppStatus
rppi_color_twist_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u *>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u *>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u *>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32f beta, Rpp32f hueShift, Rpp32f saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(hueShift, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(saturationFactor, rpp::deref(rppHandle), paramIndex++);
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u *>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[2].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u *>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u *>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, RppiROI *roiPoints, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	color_twist_host_batch<Rpp8u>(
		static_cast<Rpp8u *>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u *>(dstPtr),
		alpha,
		beta,
		hueShift,
		saturationFactor,
		roiPoints,
		0,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_color_twist_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (color_twist_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta, hueShift, saturationFactor, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_color_twist_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (color_twist_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta, hueShift, saturationFactor, outputFormatToggle, nbatchSize, rppHandle));
}

RppStatus
rppi_color_twist_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (color_twist_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta, hueShift, saturationFactor, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_color_twist_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (color_twist_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta, hueShift, saturationFactor, outputFormatToggle, nbatchSize, rppHandle));
}

RppStatus
rppi_color_twist_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (color_twist_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta, hueShift, saturationFactor, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_color_twist_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (color_twist_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, alpha, beta, hueShift, saturationFactor, outputFormatToggle, nbatchSize, rppHandle));
}

RppStatus
rppi_crop_mirror_normalize_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u crop_pos_x, Rpp32u crop_pos_y, Rpp32f mean, Rpp32f std_dev, Rpp32u mirrorFlag, Rpp32u outputFormatToggle, rppHandle_t rppHandle)
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
			RPPI_CHN_PLANAR, 1, rpp::deref(rppHandle));
	}
#elif defined(HIP_COMPILE)
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
			RPPI_CHN_PLANAR, 1, rpp::deref(rppHandle));
	}
#endif //BACKEND
	return RPP_SUCCESS;
}
RppStatus
rppi_crop_mirror_normalize_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u crop_pos_x, Rpp32u crop_pos_y, Rpp32f mean, Rpp32f std_dev, Rpp32u mirrorFlag, Rpp32u outputFormatToggle, rppHandle_t rppHandle)
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
			RPPI_CHN_PLANAR, 3, rpp::deref(rppHandle));
	}
#elif defined(HIP_COMPILE)
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
			RPPI_CHN_PLANAR, 3, rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_crop_mirror_normalize_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u crop_pos_x, Rpp32u crop_pos_y, Rpp32f mean, Rpp32f std_dev, Rpp32u mirrorFlag, Rpp32u outputFormatToggle, rppHandle_t rppHandle)
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
			RPPI_CHN_PACKED, 3, rpp::deref(rppHandle));
	}
#elif defined(HIP_COMPILE)
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
			RPPI_CHN_PACKED, 3, rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
crop_mirror_normalize_helper(RppiChnFormat chn_format, Rpp32u num_of_channels,
							 RPPTensorDataType in_tensor_type, RPPTensorDataType out_tensor_type,
							 RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize,
							 RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize,
							 Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev,
							 Rpp32u *mirrorFlag, Rpp32u outputFormatToggle,
							 Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;

	RPPTensorFunctionMetaData tensor_info(chn_format, in_tensor_type, out_tensor_type, num_of_channels,
							(bool)outputFormatToggle);
	
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize(maxDstSize, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._in_format);
	get_dstBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._out_format);
	copy_param_uint(crop_pos_x, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint(crop_pos_y, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(mean, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(std_dev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint(mirrorFlag, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		crop_mirror_normalize_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
		    tensor_info);
	}
#elif defined(HIP_COMPILE)
	{
		if (tensor_type == RPPTensorDataType::U8)
		{
			crop_mirror_normalize_hip_batch(
				static_cast<Rpp8u *>(srcPtr),
				static_cast<Rpp8u *>(dstPtr),
				rpp::deref(rppHandle),
				chn_format, num_of_channels, tensor_type);
		}
		else if (tensor_type == RPPTensorDataType::FP32)
		{
			crop_mirror_normalize_hip_batch(
				static_cast<Rpp32f *>(srcPtr),
				static_cast<Rpp32f *>(dstPtr),
				rpp::deref(rppHandle),
				chn_format, num_of_channels, tensor_type);
		}
		else if (tensor_type == RPPTensorDataType::FP16)
		{
			crop_mirror_normalize_hip_batch(
				static_cast<data_type_t *>(srcPtr),
				static_cast<data_type_t *>(dstPtr),
				rpp::deref(rppHandle),
				chn_format, num_of_channels, tensor_type);
		}
	}
#endif //BACKEND

	return RPP_SUCCESS;
}
RppStatus
rppi_crop_mirror_normalize_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8,
										 srcPtr, srcSize, maxSrcSize,
										 dstPtr, dstSize, maxDstSize,
										 crop_pos_x, crop_pos_y, mean,
										 std_dev, mirrorFlag, outputFormatToggle,
										 nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32,
										 srcPtr, srcSize, maxSrcSize,
										 dstPtr, dstSize, maxDstSize,
										 crop_pos_x, crop_pos_y, mean,
										 std_dev, mirrorFlag, outputFormatToggle,
										 nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16,
										 srcPtr, srcSize, maxSrcSize,
										 dstPtr, dstSize, maxDstSize,
										 crop_pos_x, crop_pos_y, mean,
										 std_dev, mirrorFlag, outputFormatToggle,
										 nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8,
										 srcPtr, srcSize, maxSrcSize,
										 dstPtr, dstSize, maxDstSize,
										 crop_pos_x, crop_pos_y, mean,
										 std_dev, mirrorFlag, outputFormatToggle,
										 nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32,
										 srcPtr, srcSize, maxSrcSize,
										 dstPtr, dstSize, maxDstSize,
										 crop_pos_x, crop_pos_y, mean,
										 std_dev, mirrorFlag, outputFormatToggle,
										 nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16,
										 srcPtr, srcSize, maxSrcSize,
										 dstPtr, dstSize, maxDstSize,
										 crop_pos_x, crop_pos_y, mean,
										 std_dev, mirrorFlag, outputFormatToggle,
										 nbatchSize, rppHandle));
}

RppStatus
rppi_crop_mirror_normalize_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::U8,
										 srcPtr, srcSize, maxSrcSize,
										 dstPtr, dstSize, maxDstSize,
										 crop_pos_x, crop_pos_y, mean,
										 std_dev, mirrorFlag, outputFormatToggle,
										 nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32,
										 srcPtr, srcSize, maxSrcSize,
										 dstPtr, dstSize, maxDstSize,
										 crop_pos_x, crop_pos_y, mean,
										 std_dev, mirrorFlag, outputFormatToggle,
										 nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16,
										 srcPtr, srcSize, maxSrcSize,
										 dstPtr, dstSize, maxDstSize,
										 crop_pos_x, crop_pos_y, mean,
										 std_dev, mirrorFlag, outputFormatToggle,
										 nbatchSize, rppHandle));
}

RppStatus
rppi_crop_mirror_normalize_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, RPPTensorDataType::I8,
										 srcPtr, srcSize, maxSrcSize,
										 dstPtr, dstSize, maxDstSize,
										 crop_pos_x, crop_pos_y, mean,
										 std_dev, mirrorFlag, outputFormatToggle,
										 nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8,
										 srcPtr, srcSize, maxSrcSize,
										 dstPtr, dstSize, maxDstSize,
										 crop_pos_x, crop_pos_y, mean,
										 std_dev, mirrorFlag, outputFormatToggle,
										 nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8,
										 srcPtr, srcSize, maxSrcSize,
										 dstPtr, dstSize, maxDstSize,
										 crop_pos_x, crop_pos_y, mean,
										 std_dev, mirrorFlag, outputFormatToggle,
										 nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::I8,
										 srcPtr, srcSize, maxSrcSize,
										 dstPtr, dstSize, maxDstSize,
										 crop_pos_x, crop_pos_y, mean,
										 std_dev, mirrorFlag, outputFormatToggle,
										 nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::I8,
										 srcPtr, srcSize, maxSrcSize,
										 dstPtr, dstSize, maxDstSize,
										 crop_pos_x, crop_pos_y, mean,
										 std_dev, mirrorFlag, outputFormatToggle,
										 nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::I8,
										 srcPtr, srcSize, maxSrcSize,
										 dstPtr, dstSize, maxDstSize,
										 crop_pos_x, crop_pos_y, mean,
										 std_dev, mirrorFlag, outputFormatToggle,
										 nbatchSize, rppHandle));
}

RppStatus
rppi_crop_mirror_normalize_u8_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::FP32,
										 srcPtr, srcSize, maxSrcSize,
										 dstPtr, dstSize, maxDstSize,
										 crop_pos_x, crop_pos_y, mean,
										 std_dev, mirrorFlag, outputFormatToggle,
										 nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::FP32,
										 srcPtr, srcSize, maxSrcSize,
										 dstPtr, dstSize, maxDstSize,
										 crop_pos_x, crop_pos_y, mean,
										 std_dev, mirrorFlag, outputFormatToggle,
										 nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::FP32,
										 srcPtr, srcSize, maxSrcSize,
										 dstPtr, dstSize, maxDstSize,
										 crop_pos_x, crop_pos_y, mean,
										 std_dev, mirrorFlag, outputFormatToggle,
										 nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::FP16,
										 srcPtr, srcSize, maxSrcSize,
										 dstPtr, dstSize, maxDstSize,
										 crop_pos_x, crop_pos_y, mean,
										 std_dev, mirrorFlag, outputFormatToggle,
										 nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::FP16,
										 srcPtr, srcSize, maxSrcSize,
										 dstPtr, dstSize, maxDstSize,
										 crop_pos_x, crop_pos_y, mean,
										 std_dev, mirrorFlag, outputFormatToggle,
										 nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::FP16,
										 srcPtr, srcSize, maxSrcSize,
										 dstPtr, dstSize, maxDstSize,
										 crop_pos_x, crop_pos_y, mean,
										 std_dev, mirrorFlag, outputFormatToggle,
										 nbatchSize, rppHandle));
}



RppStatus
rppi_crop_mirror_normalize_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u crop_pos_x, Rpp32u crop_pos_y, Rpp32f mean, Rpp32f stdDev, Rpp32u mirrorFlag, Rpp32u outputFormatToggle, rppHandle_t rppHandle)
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
		RPPI_CHN_PLANAR, 1);

	return RPP_SUCCESS;
}

RppStatus
rppi_crop_mirror_normalize_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u crop_pos_x, Rpp32u crop_pos_y, Rpp32f mean, Rpp32f stdDev, Rpp32u mirrorFlag, Rpp32u outputFormatToggle, rppHandle_t rppHandle)
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
		RPPI_CHN_PLANAR, 3);

	return RPP_SUCCESS;
}

RppStatus
rppi_crop_mirror_normalize_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u crop_pos_x, Rpp32u crop_pos_y, Rpp32f mean, Rpp32f stdDev, Rpp32u mirrorFlag, Rpp32u outputFormatToggle, rppHandle_t rppHandle)
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
		RPPI_CHN_PACKED, 3);

	return RPP_SUCCESS;
}

RppStatus
crop_mirror_normalize_host_helper(RppiChnFormat chn_format, Rpp32u num_of_channels,
								  RPPTensorDataType tensorInType, RPPTensorDataType tensorOutType,
								  RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize,
								  RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize,
								  Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev,
								  Rpp32u *mirrorFlag, Rpp32u outputFormatToggle,
								  Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	if (tensorInType == RPPTensorDataType::U8)
	{
		if (tensorOutType == RPPTensorDataType::U8)
		{
			crop_mirror_normalize_host_batch<Rpp8u>(
				static_cast<Rpp8u *>(srcPtr),
				srcSize,
				rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
				static_cast<Rpp8u *>(dstPtr),
				dstSize,
				rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
				crop_pos_x,
				crop_pos_y,
				mean,
				stdDev,
				mirrorFlag,
				outputFormatToggle,
				rpp::deref(rppHandle).GetBatchSize(),
				chn_format, num_of_channels);
		}
		else if (tensorOutType == RPPTensorDataType::FP16)
		{
			crop_mirror_normalize_u8_f_host_batch<Rpp8u, Rpp16f>(
				static_cast<Rpp8u *>(srcPtr),
				srcSize,
				rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
				static_cast<Rpp16f *>(dstPtr),
				dstSize,
				rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
				crop_pos_x,
				crop_pos_y,
				mean,
				stdDev,
				mirrorFlag,
				outputFormatToggle,
				rpp::deref(rppHandle).GetBatchSize(),
				chn_format, num_of_channels);
		}
		else if (tensorOutType == RPPTensorDataType::FP32)
		{
			crop_mirror_normalize_u8_f_host_batch<Rpp8u, Rpp32f>(
				static_cast<Rpp8u *>(srcPtr),
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
				chn_format, num_of_channels);
		}
		else if (tensorOutType == RPPTensorDataType::I8)
		{
			crop_mirror_normalize_u8_i8_host_batch(
				static_cast<Rpp8u *>(srcPtr),
				srcSize,
				rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
				static_cast<Rpp8s *>(dstPtr),
				dstSize,
				rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
				crop_pos_x,
				crop_pos_y,
				mean,
				stdDev,
				mirrorFlag,
				outputFormatToggle,
				rpp::deref(rppHandle).GetBatchSize(),
				chn_format, num_of_channels);
		}
	}
	else if (tensorInType == RPPTensorDataType::FP16)
	{
		crop_mirror_normalize_f16_host_batch(
			static_cast<Rpp16f *>(srcPtr),
			srcSize,
			rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
			static_cast<Rpp16f *>(dstPtr),
			dstSize,
			rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
			crop_pos_x,
			crop_pos_y,
			mean,
			stdDev,
			mirrorFlag,
			outputFormatToggle,
			rpp::deref(rppHandle).GetBatchSize(),
			chn_format, num_of_channels);
	}
	else if (tensorInType == RPPTensorDataType::FP32)
	{
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
			chn_format, num_of_channels);
	}
	else if (tensorInType == RPPTensorDataType::I8)
	{
		crop_mirror_normalize_host_batch<Rpp8s>(
			static_cast<Rpp8s *>(srcPtr),
			srcSize,
			rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
			static_cast<Rpp8s *>(dstPtr),
			dstSize,
			rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
			crop_pos_x,
			crop_pos_y,
			mean,
			stdDev,
			mirrorFlag,
			outputFormatToggle,
			rpp::deref(rppHandle).GetBatchSize(),
			chn_format, num_of_channels);
	}

	return RPP_SUCCESS;
}

RppStatus
rppi_crop_mirror_normalize_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}

RppStatus
rppi_crop_mirror_normalize_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}

RppStatus
rppi_crop_mirror_normalize_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}

RppStatus
rppi_crop_mirror_normalize_u8_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}

RppStatus
rppi_crop_mirror_normalize_u8_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}

RppStatus
rppi_crop_mirror_normalize_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}

RppStatus
rppi_crop_mirror_normalize_u8_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_mirror_normalize_u8_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_mirror_normalize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, mean, stdDev, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}

RppStatus
crop_helper(RppiChnFormat chn_format, Rpp32u num_of_channels,
			RPPTensorDataType in_tensor_type, RPPTensorDataType out_tensor_type,
			RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize,
			RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize,
			Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle,
			Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	RPPTensorFunctionMetaData tensor_info(chn_format, in_tensor_type, out_tensor_type, num_of_channels,
							(bool)outputFormatToggle);
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize(maxDstSize, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), num_of_channels, chn_format);
	get_dstBatchIndex(rpp::deref(rppHandle), num_of_channels, chn_format);
	copy_param_uint(crop_pos_x, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint(crop_pos_y, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		crop_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			tensor_info);
	}
#elif defined(HIP_COMPILE)
	{
		if (tensor_type == RPPTensorDataType::U8)
		{
			crop_hip_batch(
				static_cast<Rpp8u *>(srcPtr),
				static_cast<Rpp8u *>(dstPtr),
				rpp::deref(rppHandle),
				chn_format, num_of_channels, tensor_type);
		}
		else if (tensor_type == RPPTensorDataType::FP32)
		{
			crop_hip_batch(
				static_cast<Rpp32f *>(srcPtr),
				static_cast<Rpp32f *>(dstPtr),
				rpp::deref(rppHandle),
				chn_format, num_of_channels, tensor_type);
		}
		else if (tensor_type == RPPTensorDataType::FP16)
		{
			crop_hip_batch(
				static_cast<data_type_t *>(srcPtr),
				static_cast<data_type_t *>(dstPtr),
				rpp::deref(rppHandle),
				chn_format, num_of_channels, tensor_type);
		}
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_crop_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle,  Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::U8,
						srcPtr, srcSize, maxSrcSize,
						dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle,
						nbatchSize, rppHandle));
}

RppStatus
rppi_crop_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle,  Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8,
						srcPtr, srcSize, maxSrcSize,
						dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle,
						nbatchSize, rppHandle));
}

RppStatus
rppi_crop_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle,  Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8,
						srcPtr, srcSize, maxSrcSize,
						dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle,
						nbatchSize, rppHandle));
}

RppStatus
rppi_crop_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle,  Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16,
						srcPtr, srcSize, maxSrcSize,
						dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle,
						nbatchSize, rppHandle));
}

RppStatus
rppi_crop_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle,  Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16,
						srcPtr, srcSize, maxSrcSize,
						dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle,
						nbatchSize, rppHandle));
}

RppStatus
rppi_crop_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle,  Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16,
						srcPtr, srcSize, maxSrcSize,
						dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle,
						nbatchSize, rppHandle));
}

RppStatus
rppi_crop_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle,  Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32,
						srcPtr, srcSize, maxSrcSize,
						dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle,
						nbatchSize, rppHandle));
}

RppStatus
rppi_crop_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle,  Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32,
						srcPtr, srcSize, maxSrcSize,
						dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle,
						nbatchSize, rppHandle));
}

RppStatus
rppi_crop_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle,  Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32,
						srcPtr, srcSize, maxSrcSize,
						dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle,
						nbatchSize, rppHandle));
}

RppStatus
rppi_crop_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle,  Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, RPPTensorDataType::I8,
						srcPtr, srcSize, maxSrcSize,
						dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle,
						nbatchSize, rppHandle));
}

RppStatus
rppi_crop_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle,  Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8,
						srcPtr, srcSize, maxSrcSize,
						dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle,
						nbatchSize, rppHandle));
}

RppStatus
rppi_crop_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle,  Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8,
						srcPtr, srcSize, maxSrcSize,
						dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle,
						nbatchSize, rppHandle));
}

RppStatus
rppi_crop_u8_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle,  Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::I8,
						srcPtr, srcSize, maxSrcSize,
						dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle,
						nbatchSize, rppHandle));
}

RppStatus
rppi_crop_u8_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle,  Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::I8,
						srcPtr, srcSize, maxSrcSize,
						dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle,
						nbatchSize, rppHandle));
}

RppStatus
rppi_crop_u8_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle,  Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::I8,
						srcPtr, srcSize, maxSrcSize,
						dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle,
						nbatchSize, rppHandle));
}

RppStatus
rppi_crop_u8_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle,  Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::FP32,
						srcPtr, srcSize, maxSrcSize,
						dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle,
						nbatchSize, rppHandle));
}

RppStatus
rppi_crop_u8_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle,  Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::FP32,
						srcPtr, srcSize, maxSrcSize,
						dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle,
						nbatchSize, rppHandle));
}

RppStatus
rppi_crop_u8_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle,  Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::FP32,
						srcPtr, srcSize, maxSrcSize,
						dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle,
						nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle,  Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::FP16,
						srcPtr, srcSize, maxSrcSize,
						dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle,
						nbatchSize, rppHandle));
}

RppStatus
rppi_crop_u8_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle,  Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::FP16,
						srcPtr, srcSize, maxSrcSize,
						dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle,
						nbatchSize, rppHandle));
}

RppStatus
rppi_crop_u8_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u output_format_toggle,  Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::FP16,
						srcPtr, srcSize, maxSrcSize,
						dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, output_format_toggle,
						nbatchSize, rppHandle));
}



RppStatus
crop_host_helper(RppiChnFormat chn_format, Rpp32u num_of_channels,
				 RPPTensorDataType tensorInType, RPPTensorDataType tensorOutType,
				 RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize,
				 RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize,
				 Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle,
				 Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));

	if (tensorInType == RPPTensorDataType::U8)
	{
		if (tensorOutType == RPPTensorDataType::U8)
		{
			crop_host_batch<Rpp8u>(
				static_cast<Rpp8u *>(srcPtr),
				srcSize,
				rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
				static_cast<Rpp8u *>(dstPtr),
				dstSize,
				rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
				crop_pos_x,
				crop_pos_y,
				outputFormatToggle,
				rpp::deref(rppHandle).GetBatchSize(),
				chn_format, num_of_channels);
		}
		else if (tensorOutType == RPPTensorDataType::FP16)
		{
			crop_host_u_f_batch<Rpp8u, Rpp16f>(
				static_cast<Rpp8u *>(srcPtr),
				srcSize,
				rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
				static_cast<Rpp16f *>(dstPtr),
				dstSize,
				rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
				crop_pos_x,
				crop_pos_y,
				outputFormatToggle,
				rpp::deref(rppHandle).GetBatchSize(),
				chn_format, num_of_channels);
		}
		else if (tensorOutType == RPPTensorDataType::FP32)
		{
			crop_host_u_f_batch<Rpp8u, Rpp32f>(
				static_cast<Rpp8u *>(srcPtr),
				srcSize,
				rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
				static_cast<Rpp32f *>(dstPtr),
				dstSize,
				rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
				crop_pos_x,
				crop_pos_y,
				outputFormatToggle,
				rpp::deref(rppHandle).GetBatchSize(),
				chn_format, num_of_channels);
		}
		else if (tensorOutType == RPPTensorDataType::I8)
		{
			crop_host_u_i_batch<Rpp8u, Rpp8s>(
				static_cast<Rpp8u *>(srcPtr),
				srcSize,
				rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
				static_cast<Rpp8s *>(dstPtr),
				dstSize,
				rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
				crop_pos_x,
				crop_pos_y,
				outputFormatToggle,
				rpp::deref(rppHandle).GetBatchSize(),
				chn_format, num_of_channels);
		}
	}
	else if (tensorInType == RPPTensorDataType::FP16)
	{
		crop_host_batch<Rpp16f>(
			static_cast<Rpp16f *>(srcPtr),
			srcSize,
			rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
			static_cast<Rpp16f *>(dstPtr),
			dstSize,
			rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
			crop_pos_x,
			crop_pos_y,
			outputFormatToggle,
			rpp::deref(rppHandle).GetBatchSize(),
			chn_format, num_of_channels);
	}
	else if (tensorInType == RPPTensorDataType::FP32)
	{
		crop_host_batch<Rpp32f>(
			static_cast<Rpp32f *>(srcPtr),
			srcSize,
			rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
			static_cast<Rpp32f *>(dstPtr),
			dstSize,
			rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
			crop_pos_x,
			crop_pos_y,
			outputFormatToggle,
			rpp::deref(rppHandle).GetBatchSize(),
			chn_format, num_of_channels);
	}
	else if (tensorInType == RPPTensorDataType::I8)
	{
		crop_host_batch<Rpp8s>(
			static_cast<Rpp8s *>(srcPtr),
			srcSize,
			rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
			static_cast<Rpp8s *>(dstPtr),
			dstSize,
			rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
			crop_pos_x,
			crop_pos_y,
			outputFormatToggle,
			rpp::deref(rppHandle).GetBatchSize(),
			chn_format, num_of_channels);
	}

	return RPP_SUCCESS;
}

RppStatus
rppi_crop_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, 0, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, 0, nbatchSize, rppHandle));
}

RppStatus
rppi_crop_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}

RppStatus
rppi_crop_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}

RppStatus
rppi_crop_u8_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}

RppStatus
rppi_crop_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}

RppStatus
rppi_crop_u8_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_crop_u8_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (crop_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, crop_pos_x, crop_pos_y, outputFormatToggle, nbatchSize, rppHandle));
}

RppStatus
resize_crop_mirror_helper(
	RppiChnFormat chn_format, Rpp32u num_of_channels,
	RPPTensorDataType tensor_type,
	RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize,
	RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize,
	Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin,
	Rpp32u *yRoiEnd, Rpp32u *mirrorFlag,
	Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
	copy_dstSize(dstSize, rpp::deref(rppHandle));
	copy_dstMaxSize(maxDstSize, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), num_of_channels, chn_format);
	get_dstBatchIndex(rpp::deref(rppHandle), num_of_channels, chn_format);
	copy_param_uint(xRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint(xRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint(yRoiBegin, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint(yRoiEnd, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint(mirrorFlag, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		resize_crop_mirror_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			chn_format, num_of_channels, tensor_type);
	}
#elif defined(HIP_COMPILE)
	{
		if (tensor_type == RPPTensorDataType::U8)
		{
			resize_crop_mirror_hip_batch(
				static_cast<Rpp8u *>(srcPtr),
				static_cast<Rpp8u *>(dstPtr),
				rpp::deref(rppHandle),
				chn_format, num_of_channels, tensor_type);
		}
		else if (tensor_type == RPPTensorDataType::FP32)
		{
			resize_crop_mirror_hip_batch(
				static_cast<Rpp32f *>(srcPtr),
				static_cast<Rpp32f *>(dstPtr),
				rpp::deref(rppHandle),
				chn_format, num_of_channels, tensor_type);
		}
		else if (tensor_type == RPPTensorDataType::FP16)
		{
			resize_crop_mirror_hip_batch(
				static_cast<data_type_t *>(srcPtr),
				static_cast<data_type_t *>(dstPtr),
				rpp::deref(rppHandle),
				chn_format, num_of_channels, tensor_type);
		}
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_resize_crop_mirror_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (resize_crop_mirror_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8,
									  srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize,
									  xRoiBegin, yRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, nbatchSize, rppHandle));
}

RppStatus
rppi_resize_crop_mirror_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (resize_crop_mirror_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8,
									  srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize,
									  xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, nbatchSize, rppHandle));
}

RppStatus
rppi_resize_crop_mirror_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (resize_crop_mirror_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8,
									  srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize,
									  xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, nbatchSize, rppHandle));
}

RppStatus
rppi_resize_crop_mirror_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (resize_crop_mirror_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8,
									  srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize,
									  xRoiBegin, yRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, nbatchSize, rppHandle));
}

RppStatus
rppi_resize_crop_mirror_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (resize_crop_mirror_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8,
									  srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize,
									  xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, nbatchSize, rppHandle));
}

RppStatus
rppi_resize_crop_mirror_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (resize_crop_mirror_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8,
									  srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize,
									  xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, nbatchSize, rppHandle));
}

RppStatus
rppi_resize_crop_mirror_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (resize_crop_mirror_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16,
									  srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize,
									  xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, nbatchSize, rppHandle));
}

RppStatus
rppi_resize_crop_mirror_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (resize_crop_mirror_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16,
									  srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize,
									  xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, nbatchSize, rppHandle));
}

RppStatus
rppi_resize_crop_mirror_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (resize_crop_mirror_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16,
									  srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize,
									  xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, nbatchSize, rppHandle));
}

RppStatus
rppi_resize_crop_mirror_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (resize_crop_mirror_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32,
									  srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize,
									  xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, nbatchSize, rppHandle));
}

RppStatus
rppi_resize_crop_mirror_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (resize_crop_mirror_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32,
									  srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize,
									  xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, nbatchSize, rppHandle));
}

RppStatus
rppi_resize_crop_mirror_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (resize_crop_mirror_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32,
									  srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize,
									  xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, nbatchSize, rppHandle));
}

RppStatus
resize_crop_mirror_host_helper(
	RppiChnFormat chn_format, Rpp32u num_of_channels,
	RPPTensorDataType tensor_type,
	RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize,
	RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize,
	Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin,
	Rpp32u *yRoiEnd, Rpp32u *mirrorFlag,
	Rpp32u outputFormatToggle,
	Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));
	if (tensor_type == RPPTensorDataType::U8)
	{
		resize_crop_mirror_host_batch<Rpp8u>(
			static_cast<Rpp8u *>(srcPtr),
			srcSize,
			rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
			static_cast<Rpp8u *>(dstPtr),
			dstSize,
			rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
			xRoiBegin,
			xRoiEnd,
			yRoiBegin,
			yRoiEnd,
			mirrorFlag,
			outputFormatToggle,
			rpp::deref(rppHandle).GetBatchSize(),
			chn_format, num_of_channels);
	}
	else if (tensor_type == RPPTensorDataType::FP16)
	{
		resize_crop_mirror_f16_host_batch(
			static_cast<Rpp16f *>(srcPtr),
			srcSize,
			rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
			static_cast<Rpp16f *>(dstPtr),
			dstSize,
			rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
			xRoiBegin,
			xRoiEnd,
			yRoiBegin,
			yRoiEnd,
			mirrorFlag,
			outputFormatToggle,
			rpp::deref(rppHandle).GetBatchSize(),
			chn_format, num_of_channels);
	}
	else if (tensor_type == RPPTensorDataType::FP32)
	{
		resize_crop_mirror_f32_host_batch(
			static_cast<Rpp32f *>(srcPtr),
			srcSize,
			rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
			static_cast<Rpp32f *>(dstPtr),
			dstSize,
			rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
			xRoiBegin,
			xRoiEnd,
			yRoiBegin,
			yRoiEnd,
			mirrorFlag,
			outputFormatToggle,
			rpp::deref(rppHandle).GetBatchSize(),
			chn_format, num_of_channels);
	}
	else if (tensor_type == RPPTensorDataType::I8)
	{
		resize_crop_mirror_host_batch<Rpp8s>(
			static_cast<Rpp8s *>(srcPtr),
			srcSize,
			rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
			static_cast<Rpp8s *>(dstPtr),
			dstSize,
			rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
			xRoiBegin,
			xRoiEnd,
			yRoiBegin,
			yRoiEnd,
			mirrorFlag,
			outputFormatToggle,
			rpp::deref(rppHandle).GetBatchSize(),
			chn_format, num_of_channels);
	}

	return RPP_SUCCESS;
}

RppStatus
rppi_resize_crop_mirror_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (resize_crop_mirror_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, 0, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (resize_crop_mirror_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, 0, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (resize_crop_mirror_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, 0, nbatchSize, rppHandle));
}

RppStatus
rppi_resize_crop_mirror_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (resize_crop_mirror_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (resize_crop_mirror_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (resize_crop_mirror_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}

RppStatus
rppi_resize_crop_mirror_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (resize_crop_mirror_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (resize_crop_mirror_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (resize_crop_mirror_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}

RppStatus
rppi_resize_crop_mirror_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (resize_crop_mirror_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (resize_crop_mirror_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_mirror_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return (resize_crop_mirror_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, mirrorFlag, outputFormatToggle, nbatchSize, rppHandle));
}