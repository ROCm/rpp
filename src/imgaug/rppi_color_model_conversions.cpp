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
// Host color_temperature functions calls 
// ----------------------------------------


RppStatus
rppi_color_temperature_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32s adjustmentValue)
{

 	 validate_image_size(srcSize);
 	 validate_int_range( -100, 100, &adjustmentValue);
	 color_temperature_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			adjustmentValue,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_color_temperature_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32s adjustmentValue)
{

 	 validate_image_size(srcSize);
 	 validate_int_range( -100, 100, &adjustmentValue);
	 color_temperature_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			adjustmentValue,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_color_temperature_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32s adjustmentValue)
{

 	 validate_image_size(srcSize);
 	 validate_int_range( -100, 100, &adjustmentValue);
	 color_temperature_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			adjustmentValue,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host vignette functions calls 
// ----------------------------------------


RppStatus
rppi_vignette_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev)
{

 	 validate_image_size(srcSize);
 	 validate_float_min( 0, &stdDev);
	 vignette_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			stdDev,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_vignette_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev)
{

 	 validate_image_size(srcSize);
 	 validate_float_min( 0, &stdDev);
	 vignette_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			stdDev,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_vignette_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev)
{

 	 validate_image_size(srcSize);
 	 validate_float_min( 0, &stdDev);
	 vignette_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			stdDev,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}
  

// ----------------------------------------
// GPU color_temperature functions  calls 
// ----------------------------------------


RppStatus
rppi_color_temperature_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32s adjustmentValue, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_range( -100, 100, &adjustmentValue);

#ifdef OCL_COMPILE
 	 {
 	 color_temperature_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			adjustmentValue,
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
rppi_color_temperature_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32s adjustmentValue, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_range( -100, 100, &adjustmentValue);

#ifdef OCL_COMPILE
 	 {
 	 color_temperature_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			adjustmentValue,
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
rppi_color_temperature_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32s adjustmentValue, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_range( -100, 100, &adjustmentValue);

#ifdef OCL_COMPILE
 	 {
 	 color_temperature_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			adjustmentValue,
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
// GPU vignette functions  calls 
// ----------------------------------------


RppStatus
rppi_vignette_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_float_min( 0,&stdDev);

#ifdef OCL_COMPILE
 	 {
 	 vignette_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			stdDev,
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
rppi_vignette_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_float_min( 0, &stdDev);

#ifdef OCL_COMPILE
 	 {
 	 vignette_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			stdDev,
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
rppi_vignette_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_float_min( 0, &stdDev);

#ifdef OCL_COMPILE
 	 {
 	 vignette_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			stdDev,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}