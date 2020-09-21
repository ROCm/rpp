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
	copy_dstMaxSize(maxSrcSize, rpp::deref(rppHandle));
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

RppStatus
rppi_non_linear_blend_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( non_linear_blend_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle,
							   srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}

RppStatus
rppi_non_linear_blend_f32_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( non_linear_blend_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle,
							   srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}

RppStatus
rppi_non_linear_blend_f16_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( non_linear_blend_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle,
							   srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}

RppStatus
rppi_non_linear_blend_i8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( non_linear_blend_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle,
							   srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}

RppStatus
rppi_non_linear_blend_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( non_linear_blend_helper(RPPI_CHN_PLANAR,1, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle,
							   srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}

RppStatus
rppi_non_linear_blend_f32_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( non_linear_blend_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle,
							   srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}

RppStatus
rppi_non_linear_blend_f16_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( non_linear_blend_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle,
							   srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}

RppStatus
rppi_non_linear_blend_i8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *std_dev,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( non_linear_blend_helper(RPPI_CHN_PLANAR,13, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle,
							   srcPtr1, srcPtr2, srcSize, maxSrcSize, dstPtr, std_dev, nbatchSize, rppHandle));
}

/*************************************** Water Augmentation ************************************/

RppStatus water_helper(RppiChnFormat chn_format, Rpp32u num_of_channels,
							 RPPTensorDataType in_tensor_type, RPPTensorDataType out_tensor_type, Rpp8u outputFormatToggle,
							 RppPtr_t srcPtr,  RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr,
                             Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y,
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
	copy_param_float(ampl_x, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(ampl_y, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(freq_x, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(freq_y, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(phase_x, rpp::deref(rppHandle), paramIndex++);
	copy_param_float(phase_y, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
        water_cl_batch(
        static_cast<cl_mem>(srcPtr),  
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
rppi_water_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr,
                                     Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( water_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle,
							   srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}

RppStatus
rppi_water_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr,
                                     Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( water_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle,
							   srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}

RppStatus
rppi_water_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr,
                                     Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( water_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle,
							   srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}

RppStatus
rppi_water_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr,
                                     Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( water_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle,
							   srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}

RppStatus
rppi_water_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr,
                                     Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( water_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle,
							   srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}

RppStatus
rppi_water_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr,
                                     Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( water_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle,
							   srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}

RppStatus
rppi_water_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr,
                                     Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( water_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle,
							   srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}

RppStatus
rppi_water_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr,
                                     Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( water_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle,
							   srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}

RppStatus
rppi_water_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr,
                                     Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( water_helper(RPPI_CHN_PLANAR,1, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle,
							   srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}

RppStatus
rppi_water_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr,
                                     Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( water_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle,
							   srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}

RppStatus
rppi_water_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr,
                                     Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( water_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle,
							   srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}

RppStatus
rppi_water_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr,
                                     Rpp32f *ampl_x, Rpp32f *ampl_y, Rpp32f *freq_x, Rpp32f *freq_y, Rpp32f *phase_x, Rpp32f *phase_y,
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( water_helper(RPPI_CHN_PLANAR,1, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle,
							   srcPtr, srcSize, maxSrcSize, dstPtr, ampl_x, ampl_y, freq_x, freq_y, phase_x, phase_y, nbatchSize, rppHandle));
}


/*************************************** Erase ************************************/
RppStatus erase_helper(RppiChnFormat chn_format, Rpp32u num_of_channels,
							 RPPTensorDataType in_tensor_type, RPPTensorDataType out_tensor_type, Rpp8u outputFormatToggle,
							 RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, 
							 RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes,
							 Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	bool is_padded = true;
	RPPTensorFunctionMetaData tensor_info(chn_format, in_tensor_type, out_tensor_type, num_of_channels,
										  (bool)outputFormatToggle);
	RppiSize maxDstSize = maxSrcSize;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
	copy_dstMaxSize(maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._in_format, is_padded);
	get_dstBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._out_format, is_padded);
	copy_param_uint(num_of_boxes, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
        erase_cl_batch(
        static_cast<cl_mem>(srcPtr),
        static_cast<cl_mem>(dstPtr),
		static_cast<cl_mem>(anchor_box_info),
		static_cast<cl_mem>(colors),
		static_cast<cl_mem>(box_offset),
        rpp::deref(rppHandle),
        tensor_info);
	}
#elif defined(HIP_COMPILE)
// Yet to be done
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus
rppi_erase_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr,
                                     RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, 
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( erase_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle,
							   srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, 
							   num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr,
                                     RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, 
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( erase_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle,
							   srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, 
							   num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr,
                                     RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, 
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( erase_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle,
							   srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, 
							   num_of_boxes, nbatchSize, rppHandle));
}

RppStatus
rppi_erase_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr,
                                     RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, 
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( erase_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle,
							   srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, 
							   num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr,
                                     RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, 
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( erase_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle,
							   srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, 
							   num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr,
                                     RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, 
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( erase_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle,
							   srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, 
							   num_of_boxes, nbatchSize, rppHandle));
}

RppStatus
rppi_erase_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr,
                                     RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, 
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( erase_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle,
							   srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, 
							   num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr,
                                     RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, 
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( erase_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle,
							   srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, 
							   num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr,
                                     RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, 
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( erase_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle,
							   srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, 
							   num_of_boxes, nbatchSize, rppHandle));
}

RppStatus
rppi_erase_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr,
                                     RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, 
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( erase_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle,
							   srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, 
							   num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr,
                                     RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, 
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( erase_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle,
							   srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, 
							   num_of_boxes, nbatchSize, rppHandle));
}
RppStatus
rppi_erase_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr,
                                     RppPtr_t anchor_box_info, RppPtr_t colors, RppPtr_t box_offset, Rpp32u *num_of_boxes, 
									 Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
	return ( erase_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle,
							   srcPtr, srcSize, maxSrcSize, dstPtr, anchor_box_info, colors, box_offset, 
							   num_of_boxes, nbatchSize, rppHandle));
}
