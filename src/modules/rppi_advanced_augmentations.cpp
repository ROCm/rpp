#include <rppi_advanced_augmentations.h>
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


RppStatus non_linear_blend_helper(RppiChnFormat chn_format, Rpp32u num_of_channels,
							 RPPTensorDataType in_tensor_type, RPPTensorDataType out_tensor_type, Rpp8u outputFormatToggle,
							 RppPtr_t srcPtr1,  RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev,
							 Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	RppiROI roiPoints;
	bool is_padded = true;
	RPPTensorFunctionMetaData tensor_info(chn_format, in_tensor_type, out_tensor_type, num_of_channels,
										  (bool)outputFormatToggle);
	RppiSize maxDstSize = maxSrcSize;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
	copy_dstMaxSize(maxDstSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._in_format, is_padded);
	get_dstBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._out_format, is_padded);
	copy_param_float(std_dev, rpp::deref(rppHandle), paramIndex++);
	

#ifdef OCL_COMPILE
	{
        non_linear_blend_cl_batch(
        static_cast<cl_mem>(srcPtr1),
        static_cast<cl_mem>(srcPtr2),    
        static_cast<cl_mem>(dstPtr),
        rpp::deref(rppHandle),
        tensor_info);
	}
#elif defined(HIP_COMPILE)
// Yet to be done
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_non_linear_blend_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( non_linear_blend_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle,
							   srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}

RppStatus
rppi_non_linear_blend_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( non_linear_blend_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle,
							   srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}

RppStatus
rppi_non_linear_blend_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( non_linear_blend_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle,
							   srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}

RppStatus
rppi_non_linear_blend_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( non_linear_blend_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle,
							   srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}
