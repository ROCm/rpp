#include <rppi_color_model_conversions.h>
#include <rppdefs.h>
#include "rppi_validate.hpp"

#ifdef HIP_COMPILE
#include <hip/rpp_hip_common.hpp>

#elif defined(OCL_COMPILE)
#include <cl/rpp_cl_common.hpp>
#include "cl/cl_declarations.hpp"
#endif //backend
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
using namespace std::chrono; 

#include "cpu/host_color_model_conversions.hpp" 
 
// ----------------------------------------
// Host rgb_to_hsv functions calls 
// ----------------------------------------


RppStatus
rppi_rgb_to_hsv_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
	 rgb_to_hsv_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_rgb_to_hsv_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
	 rgb_to_hsv_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_rgb_to_hsv_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
	 rgb_to_hsv_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host hsv_to_rgb functions calls 
// ----------------------------------------


RppStatus
rppi_hsv_to_rgb_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
	 hsv_to_rgb_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_hsv_to_rgb_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
	 hsv_to_rgb_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_hsv_to_rgb_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
	 hsv_to_rgb_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host hueRGB functions calls 
// ----------------------------------------


RppStatus
rppi_hueRGB_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift)
{

 	 validate_image_size(srcSize);
	 hueRGB_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			hueShift,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_hueRGB_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift)
{

 	 validate_image_size(srcSize);
	 hueRGB_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			hueShift,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_hueRGB_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift)
{

 	 validate_image_size(srcSize);
	 hueRGB_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			hueShift,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host hueHSV functions calls 
// ----------------------------------------


RppStatus
rppi_hueHSV_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift)
{

 	 validate_image_size(srcSize);
	 hueHSV_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			hueShift,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_hueHSV_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift)
{

 	 validate_image_size(srcSize);
	 hueHSV_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			hueShift,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_hueHSV_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift)
{

 	 validate_image_size(srcSize);
	 hueHSV_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			hueShift,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host saturationRGB functions calls 
// ----------------------------------------


RppStatus
rppi_saturationRGB_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor)
{

 	 validate_image_size(srcSize);
	 saturationRGB_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			saturationFactor,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_saturationRGB_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor)
{

 	 validate_image_size(srcSize);
	 saturationRGB_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			saturationFactor,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_saturationRGB_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor)
{

 	 validate_image_size(srcSize);
	 saturationRGB_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			saturationFactor,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host saturationHSV functions calls 
// ----------------------------------------


RppStatus
rppi_saturationHSV_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor)
{

 	 validate_image_size(srcSize);
	 saturationHSV_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			saturationFactor,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_saturationHSV_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor)
{

 	 validate_image_size(srcSize);
	 saturationHSV_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			saturationFactor,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_saturationHSV_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor)
{

 	 validate_image_size(srcSize);
	 saturationHSV_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			saturationFactor,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host rgb_to_hsl functions calls 
// ----------------------------------------


RppStatus
rppi_rgb_to_hsl_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
	 rgb_to_hsl_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_rgb_to_hsl_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
	 rgb_to_hsl_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_rgb_to_hsl_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
	 rgb_to_hsl_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host hsl_to_rgb functions calls 
// ----------------------------------------


RppStatus
rppi_hsl_to_rgb_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
	 hsl_to_rgb_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_hsl_to_rgb_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
	 hsl_to_rgb_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_hsl_to_rgb_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
	 hsl_to_rgb_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host exposure functions calls 
// ----------------------------------------


RppStatus
rppi_exposureRGB_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f exposureFactor)
{

    exposure_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                           exposureFactor,
                           RPPI_CHN_PLANAR, 3, RGB);
    return RPP_SUCCESS;
}

RppStatus
rppi_exposureRGB_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f exposureFactor)
{

    exposure_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                           exposureFactor,
                           RPPI_CHN_PACKED, 3, RGB);
    return RPP_SUCCESS;
}

RppStatus
rppi_exposureHSV_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f exposureFactor)
{

    exposure_host<Rpp32f>(static_cast<Rpp32f*>(srcPtr), srcSize, static_cast<Rpp32f*>(dstPtr),
                           exposureFactor,
                           RPPI_CHN_PLANAR, 3, HSV);
    return RPP_SUCCESS;
}

RppStatus
rppi_exposureHSV_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f exposureFactor)
{

    exposure_host<Rpp32f>(static_cast<Rpp32f*>(srcPtr), srcSize, static_cast<Rpp32f*>(dstPtr),
                           exposureFactor,
                           RPPI_CHN_PACKED, 3, HSV);
    return RPP_SUCCESS;
}
 
// ----------------------------------------
// GPU rgb_to_hsv functions  calls 
// ----------------------------------------


RppStatus
rppi_rgb_to_hsv_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 rgb_to_hsv_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_rgb_to_hsv_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 rgb_to_hsv_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_rgb_to_hsv_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 rgb_to_hsv_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}
 
// ----------------------------------------
// GPU hsv_to_rgb functions  calls 
// ----------------------------------------


RppStatus
rppi_hsv_to_rgb_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 hsv_to_rgb_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_hsv_to_rgb_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 hsv_to_rgb_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_hsv_to_rgb_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 hsv_to_rgb_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}
 
// ----------------------------------------
// GPU hueRGB functions  calls 
// ----------------------------------------


RppStatus
rppi_hueRGB_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 hueRGB_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			hueShift, 0.0/*Saturation*/,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_hueRGB_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 hueRGB_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			hueShift, 0.0/*Saturation*/,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_hueRGB_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 hueRGB_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			hueShift, 0.0/*Saturation*/,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}
 
// ----------------------------------------
// GPU hueHSV functions  calls 
// ----------------------------------------


RppStatus
rppi_hueHSV_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 hueHSV_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			hueShift, 0.0/*Saturation*/,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_hueHSV_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 hueHSV_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			hueShift, 0.0/*Saturation*/,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_hueHSV_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 hueHSV_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			hueShift, 0.0/*Saturation*/,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}
 
// ----------------------------------------
// GPU saturationRGB functions  calls 
// ----------------------------------------


RppStatus
rppi_saturationRGB_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 hueRGB_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			 0.0/*hue*/,saturationFactor,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_saturationRGB_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 hueRGB_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			 0.0/*hue*/,saturationFactor,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_saturationRGB_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 hueRGB_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			 0.0/*hue*/,saturationFactor,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}
 
// ----------------------------------------
// GPU saturationHSV functions  calls 
// ----------------------------------------


RppStatus
rppi_saturationHSV_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 hueHSV_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			 0.0/*hue*/,saturationFactor,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_saturationHSV_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 hueHSV_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			0.0/*hue*/,saturationFactor,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_saturationHSV_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor, RppHandle_t rppHandle) 
{
 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 hueHSV_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			 0.0/*hue*/, saturationFactor,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}




/******* Exposure Modification ********/

// GPU calls for Exposure Modification function

// Host calls for Exposure Modification function

RppStatus
rppi_exposureRGB_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f exposureFactor)
{

    exposure_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                           exposureFactor,
                           RPPI_CHN_PLANAR, 3, RGB);
    return RPP_SUCCESS;
}

RppStatus
rppi_exposureRGB_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f exposureFactor)
{

    exposure_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                           exposureFactor,
                           RPPI_CHN_PACKED, 3, RGB);
    return RPP_SUCCESS;
}

RppStatus
rppi_exposureHSV_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f exposureFactor)
{

    exposure_host<Rpp32f>(static_cast<Rpp32f*>(srcPtr), srcSize, static_cast<Rpp32f*>(dstPtr),
                           exposureFactor,
                           RPPI_CHN_PLANAR, 3, HSV);
    return RPP_SUCCESS;
}

RppStatus
rppi_exposureHSV_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f exposureFactor)
{

    exposure_host<Rpp32f>(static_cast<Rpp32f*>(srcPtr), srcSize, static_cast<Rpp32f*>(dstPtr),
                           exposureFactor,
                           RPPI_CHN_PACKED, 3, HSV);
    return RPP_SUCCESS;
}




/******* RGB 2 HSL ********/

// GPU calls for RGB 2 HSL function

// Host calls for RGB 2 HSL function

RppStatus
rppi_rgb2hsl_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{

    rgb2hsl_host<Rpp8u, Rpp32f>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp32f*>(dstPtr),
                         RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;

}

RppStatus
rppi_rgb2hsl_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{

    rgb2hsl_host<Rpp8u, Rpp32f>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp32f*>(dstPtr),
                         RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;

}




/******* HSL 2 RGB ********/

// GPU calls for HSL 2 RGB function

// Host calls for HSL 2 RGB function

RppStatus
rppi_hsl2rgb_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{

    hsl2rgb_host<Rpp32f, Rpp8u>(static_cast<Rpp32f*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                         RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;

}

RppStatus
rppi_hsl2rgb_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{

    hsl2rgb_host<Rpp32f, Rpp8u>(static_cast<Rpp32f*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                         RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}