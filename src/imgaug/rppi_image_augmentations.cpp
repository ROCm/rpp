#include <rppi_image_augmentations.h>
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

#include "cpu/host_image_augmentations.hpp"

// ----------------------------------------
// Host blur functions calls
// ----------------------------------------


RppStatus
rppi_blur_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev)
{
	 RppStatus status;
 	 validate_image_size(srcSize);
 	 validate_float_min(0, &stdDev);
	 unsigned int kernelSize = 3;
	 status = blur_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PLANAR, 1);
	return status;
}

RppStatus
rppi_blur_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev)
{

 	 validate_image_size(srcSize);
 	 validate_float_min(0, &stdDev);
	 unsigned int kernelSize = 3;
	 blur_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_blur_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev)
{

 	 validate_image_size(srcSize);
 	 validate_float_min(0, &stdDev);
	 unsigned int kernelSize = 3;
	 blur_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}

// ----------------------------------------
// Host contrast functions calls
// ----------------------------------------


RppStatus
rppi_contrast_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u newMin,Rpp32u newMax)
{

	 RppStatus status;
 	 validate_image_size(srcSize);
 	 validate_unsigned_int_max(newMax, &newMin);
	 if (newMax > 255) newMax = 255;
	 if  (newMax < 0)  newMin = 0;

	 status = contrast_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			newMin,
			newMax,
			RPPI_CHN_PLANAR, 1);

	return RPP_SUCCESS;
}

RppStatus
rppi_contrast_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u newMin,Rpp32u newMax)
{

	 RppStatus status;
 	 validate_image_size(srcSize);
 	 validate_unsigned_int_max(newMax, &newMin);
	 if (newMax > 255) newMax = 255;
	 if  (newMax < 0)  newMin = 0;

	 status = contrast_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			newMin,
			newMax,
			RPPI_CHN_PLANAR, 3);

	return status;
}

RppStatus
rppi_contrast_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u newMin,Rpp32u newMax)
{

	 RppStatus status;
 	 validate_image_size(srcSize);
 	 validate_unsigned_int_max(newMax, &newMin);
	 if (newMax > 255) newMax = 255;
	 if  (newMax < 0)  newMin = 0;

	 status = contrast_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			newMin,
			newMax,
			RPPI_CHN_PACKED, 3);

	return RPP_SUCCESS;
}

// ----------------------------------------
// Host brightness functions calls
// ----------------------------------------


RppStatus
rppi_brightness_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha,Rpp32f beta)
{
	 RppStatus status;
 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 2, &alpha);
 	 validate_float_range( 0, 255, &beta);

	 status = brightness_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			alpha,
			beta,
			RPPI_CHN_PLANAR, 1);

	return status;
}

RppStatus
rppi_brightness_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha,Rpp32f beta)
{

	 RppStatus status;
 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 2, &alpha);
 	 validate_float_range( 0, 255, &beta);

	 status = brightness_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			alpha,
			beta,
			RPPI_CHN_PLANAR, 3);

	return status;
}

RppStatus
rppi_brightness_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha,Rpp32f beta)
{
	 RppStatus status;
 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 2, &alpha);
 	 validate_float_range( 0, 255, &beta);

	 status = brightness_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			alpha,
			beta,
			RPPI_CHN_PACKED, 3);

	return status;
}

// ----------------------------------------
// Host gamma_correction functions calls
// ----------------------------------------


RppStatus
rppi_gamma_correction_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f gamma)
{

 	 validate_image_size(srcSize);
 	 validate_float_min(0, &gamma);
	 gamma_correction_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			gamma,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_gamma_correction_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f gamma)
{

 	 validate_image_size(srcSize);
 	 validate_float_min(0, &gamma);
	 gamma_correction_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			gamma,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_gamma_correction_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f gamma)
{

 	 validate_image_size(srcSize);
 	 validate_float_min(0, &gamma);
	 gamma_correction_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			gamma,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}



// // ----------------------------------------
// Host jitter functions calls
// ----------------------------------------


RppStatus
rppi_jitter_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize)
{
	validate_image_size(srcSize);
    jitter_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     kernelSize,
                     RPPI_CHN_PLANAR, 1);
    return RPP_SUCCESS;
}

RppStatus
rppi_jitter_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize)
{
	validate_image_size(srcSize);
    jitter_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     kernelSize,
                     RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_jitter_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize)
{
	validate_image_size(srcSize);
    jitter_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     kernelSize,
                     RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}


// ----------------------------------------
// Host snpNoise functions calls
// ----------------------------------------


RppStatus
rppi_snpNoise_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f noiseProbability)
{
	validate_image_size(srcSize);
 	validate_float_range( 0, 1, &noiseProbability);
    noise_snp_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                        noiseProbability,
                        RPPI_CHN_PACKED, 1);
    return RPP_SUCCESS;
}

RppStatus
rppi_snpNoise_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f noiseProbability)
{
	validate_image_size(srcSize);
 	validate_float_range( 0, 1, &noiseProbability);
    noise_snp_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                        noiseProbability,
                        RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_snpNoise_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f noiseProbability)
{
	validate_image_size(srcSize);
 	validate_float_range( 0, 1, &noiseProbability);
    noise_snp_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                        noiseProbability,
                        RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}

// ----------------------------------------
// Host fog functions call
// ----------------------------------------

RppStatus
rppi_fog_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32f fogValue)
{
 	validate_float_range( 0, 1,&fogValue);
    validate_image_size(srcSize);
    Rpp32f stdDev=fogValue*50;
	unsigned int kernelSize = 5;
    if(fogValue!=0)
	blur_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                srcSize,
                static_cast<Rpp8u*>(dstPtr),
                stdDev,
                kernelSize,
                RPPI_CHN_PLANAR, 1);

    fog_host<Rpp8u>(static_cast<Rpp8u*>(dstPtr),
			srcSize,
			fogValue,
			RPPI_CHN_PLANAR, 1, static_cast<Rpp8u*>(srcPtr) );

    return RPP_SUCCESS;
}

RppStatus
rppi_fog_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32f fogValue)
{

    validate_float_range( 0, 1,&fogValue);
 	validate_image_size(srcSize);
    Rpp32f stdDev=fogValue*50;
	unsigned int kernelSize = 5;
    if(fogValue!=0)
	blur_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PLANAR, 3);

    fog_host<Rpp8u>(static_cast<Rpp8u*>(dstPtr),
			srcSize,
			fogValue,
			RPPI_CHN_PLANAR, 3, static_cast<Rpp8u*>(srcPtr));
	return RPP_SUCCESS;
}

RppStatus
rppi_fog_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32f fogValue)
{
 	validate_float_range( 0, 1,&fogValue);
    validate_image_size(srcSize);
    Rpp32f stdDev=fogValue*10;
	unsigned int kernelSize = 5;
    if(fogValue!=0)
	blur_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PACKED, 3);

    fog_host<Rpp8u>(static_cast<Rpp8u*>(dstPtr),
			srcSize,
			fogValue,
			RPPI_CHN_PACKED, 3, static_cast<Rpp8u*>(srcPtr));

    return RPP_SUCCESS;
}


// ----------------------------------------
// Host rain functions calls
// ----------------------------------------


RppStatus
rppi_rain_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f rainPercentage,Rpp32u rainWidth,Rpp32u rainHeight, Rpp32f transparency)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, &rainPercentage);
	 rain_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			rainPercentage,
			rainWidth,
			rainHeight, transparency,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_rain_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f rainPercentage,Rpp32u rainWidth,Rpp32u rainHeight, Rpp32f transparency)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, &rainPercentage);
	 rain_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			rainPercentage,
			rainWidth,
			rainHeight, transparency,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_rain_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f rainPercentage,Rpp32u rainWidth,Rpp32u rainHeight, Rpp32f transparency)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, &rainPercentage);
	 rain_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			rainPercentage,
			rainWidth,
			rainHeight, transparency,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}

// ----------------------------------------
// Host snow functions calls
// ----------------------------------------


RppStatus
rppi_snow_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f snowValue)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1,&snowValue);
	 snow_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			snowValue,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_snow_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f snowValue)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1,&snowValue);
	 snow_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			snowValue,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_snow_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f snowValue)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1,&snowValue);
	 snow_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			snowValue,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}


// ----------------------------------------
// Host blend functions calls
// ----------------------------------------


RppStatus
rppi_blend_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha)
{
	RppStatus status;
	validate_image_size(srcSize);
	validate_float_range( 0, 1, &alpha);

	status = blend_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			alpha,
			RPPI_CHN_PLANAR, 1);

	return status;
}

RppStatus
rppi_blend_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha)
{

	RppStatus status;
	validate_image_size(srcSize);
	validate_float_range( 0, 1, &alpha);

	status = blend_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			alpha,
			RPPI_CHN_PLANAR, 3);

	return status;
}

RppStatus
rppi_blend_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha)
{

	RppStatus status;
	validate_image_size(srcSize);
	validate_float_range( 0, 1, &alpha);

	blend_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			alpha,
			RPPI_CHN_PACKED, 3);

	return RPP_SUCCESS;
}

// ----------------------------------------
// Host pixelate functions calls
// ----------------------------------------


RppStatus
rppi_pixelate_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
	 pixelate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_pixelate_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
	 pixelate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_pixelate_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
	 pixelate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}


// ----------------------------------------
// GPU blur functions  calls
// ----------------------------------------


RppStatus
rppi_blur_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_min(0, &stdDev);
	 unsigned int kernelSize = 3;
#ifdef OCL_COMPILE
 	 {
 	 blur_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
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
rppi_blur_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_min(0, &stdDev);
	 unsigned int kernelSize = 3;
#ifdef OCL_COMPILE
 	 {
 	 blur_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
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
rppi_blur_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_min(0, &stdDev);
	 unsigned int kernelSize = 3;

#ifdef OCL_COMPILE
 	 {
 	 blur_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
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
// GPU contrast functions  calls
// ----------------------------------------


RppStatus
rppi_contrast_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u newMin,Rpp32u newMax, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 //validate_unsigned_int_max(newMax, &newMin);
      if(newMin > newMax){
          Rpp32u temp;
          temp = newMin;
          newMin = newMax;
          newMax = temp;
      }
      if (newMax > 255) newMax = 255;
      if  (newMin < 0)  newMin = 0;

#ifdef OCL_COMPILE
 	 {
 	 contrast_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			newMin,
			newMax,
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
rppi_contrast_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u newMin,Rpp32u newMax, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 //validate_unsigned_int_max(newMax, &newMin);
      if(newMin > newMax){
          Rpp32u temp;
          temp = newMin;
          newMin = newMax;
          newMax = temp;
      }
      if (newMax > 255) newMax = 255;
      if  (newMin < 0)  newMin = 0;


#ifdef OCL_COMPILE
 	 {
 	 contrast_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			newMin,
			newMax,
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
rppi_contrast_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u newMin,Rpp32u newMax, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 //validate_unsigned_int_max(newMax, &newMin);
      if(newMin > newMax){
          Rpp32u temp;
          temp = newMin;
          newMin = newMax;
          newMax = temp;
      }
      if (newMax > 255) newMax = 255;
      if  (newMin < 0)  newMin = 0;


#ifdef OCL_COMPILE
 	 {
 	 contrast_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			newMin,
			newMax,
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
// GPU brightness functions  calls
// ----------------------------------------


RppStatus
rppi_brightness_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha,Rpp32f beta, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range_b( 0, 2, &alpha);
 	 validate_float_range( 0, 255, &beta);

#ifdef OCL_COMPILE
 	 {
 	 brightness_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			alpha,
			beta,
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
rppi_brightness_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha,Rpp32f beta, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 2, &alpha);
 	 validate_float_range( 0, 255, &beta);

#ifdef OCL_COMPILE
 	 {
 	 brightness_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			alpha,
			beta,
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
rppi_brightness_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha,Rpp32f beta, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 2, &alpha);
 	 validate_float_range( 0, 255, &beta);

#ifdef OCL_COMPILE
 	 {
 	 brightness_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			alpha,
			beta,
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
// GPU gamma_correction functions  calls
// ----------------------------------------


RppStatus
rppi_gamma_correction_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f gamma, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_min(0, &gamma);

#ifdef OCL_COMPILE
 	 {
 	 gamma_correction_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			gamma,
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
rppi_gamma_correction_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f gamma, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_min(0, &gamma);

#ifdef OCL_COMPILE
 	 {
 	 gamma_correction_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			gamma,
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
rppi_gamma_correction_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f gamma, RppHandle_t rppHandle)
{
   	 validate_image_size(srcSize);
 	 validate_float_min(0, &gamma);

#ifdef OCL_COMPILE
 	 {
 	 gamma_correction_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			gamma,
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
// GPU rain functions  calls
// ----------------------------------------


RppStatus
rppi_rain_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f rainPercentage,Rpp32u rainWidth,Rpp32u rainHeight, Rpp32f transparency, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, &rainPercentage);

#ifdef OCL_COMPILE
 	 {
 	 rain_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			rainPercentage,
			rainWidth,
			rainHeight,
            transparency,
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
rppi_rain_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f rainPercentage,Rpp32u rainWidth,Rpp32u rainHeight, Rpp32f transparency, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, &rainPercentage);

#ifdef OCL_COMPILE
 	 {
 	 rain_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			rainPercentage,
			rainWidth,
			rainHeight,
            transparency,
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
rppi_rain_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f rainPercentage,Rpp32u rainWidth,Rpp32u rainHeight, Rpp32f transparency, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, &rainPercentage);

#ifdef OCL_COMPILE
 	 {
 	 rain_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			rainPercentage,
			rainWidth,
			rainHeight,
            transparency,
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
// GPU snow functions  calls
// ----------------------------------------


RppStatus
rppi_snow_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f snowValue, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, &snowValue);

#ifdef OCL_COMPILE
 	 {
 	 snow_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			snowValue,
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
rppi_snow_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f snowValue, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, &snowValue);

#ifdef OCL_COMPILE
 	 {
 	 snow_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			snowValue,
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
rppi_snow_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f snowValue, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, &snowValue);

#ifdef OCL_COMPILE
 	 {
 	 snow_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			snowValue,
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
// GPU blend functions  calls
// ----------------------------------------


RppStatus
rppi_blend_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, &alpha);

#ifdef OCL_COMPILE
 	 {
 	 blend_cl(static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			alpha,
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
rppi_blend_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, &alpha);

#ifdef OCL_COMPILE
 	 {
 	 blend_cl(static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			alpha,
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
rppi_blend_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, &alpha);

#ifdef OCL_COMPILE
 	 {
 	 blend_cl(static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			alpha,
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
// GPU pixelate functions  calls
// ----------------------------------------


RppStatus
rppi_pixelate_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
#ifdef OCL_COMPILE
 	 {
 	 pixelate_cl(static_cast<cl_mem>(srcPtr),
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
rppi_pixelate_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 pixelate_cl(static_cast<cl_mem>(srcPtr),
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
rppi_pixelate_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 pixelate_cl(static_cast<cl_mem>(srcPtr),
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
// GPU jitter functions  calls
// ----------------------------------------
RppStatus
rppi_jitter_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize, RppHandle_t rppHandle)
{
   	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 jitter_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
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
rppi_jitter_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize, RppHandle_t rppHandle)
{
   	 validate_image_size(srcSize);
#ifdef OCL_COMPILE
 	 {
 	 jitter_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
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
rppi_jitter_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize, RppHandle_t rppHandle)
{
   	 validate_image_size(srcSize);
#ifdef OCL_COMPILE
 	 {
 	 jitter_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
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
// GPU noise functions  calls
// ----------------------------------------

RppStatus
rppi_snpNoise_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f noiseProbability, RppHandle_t rppHandle)
{
   	validate_image_size(srcSize);
 	validate_float_range( 0, 1, &noiseProbability);
#ifdef OCL_COMPILE
 	{
            snpNoise_cl(static_cast<cl_mem>(srcPtr),
                srcSize,
                static_cast<cl_mem>(dstPtr),
                noiseProbability,
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
rppi_snpNoise_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f noiseProbability, RppHandle_t rppHandle)
{
   	validate_image_size(srcSize);
 	validate_float_range( 0, 1, &noiseProbability);

#ifdef OCL_COMPILE
 	{
            snpNoise_cl(static_cast<cl_mem>(srcPtr),
                srcSize,
                static_cast<cl_mem>(dstPtr),
                noiseProbability,
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
rppi_snpNoise_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f noiseProbability, RppHandle_t rppHandle)
{
   	validate_image_size(srcSize);
 	validate_float_range( 0, 1, &noiseProbability);

#ifdef OCL_COMPILE
 	{
            snpNoise_cl(static_cast<cl_mem>(srcPtr),
                srcSize,
                static_cast<cl_mem>(dstPtr),
                noiseProbability,
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
// GPU fog functions  calls
// ----------------------------------------


RppStatus
rppi_fog_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32f fogValue,RppHandle_t rppHandle)
{
 	validate_float_range( 0, 1,&fogValue);
    validate_image_size(srcSize);
	unsigned int kernelSize = 3;
#ifdef OCL_COMPILE
 	{

 	blur_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
    fog_cl(static_cast<cl_mem>(dstPtr),
			srcSize,
			fogValue,
			RPPI_CHN_PLANAR, 1, 
            static_cast<cl_command_queue>(rppHandle), static_cast<cl_mem>(srcPtr) );
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_fog_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32f fogValue, RppHandle_t rppHandle)
{
 	validate_float_range( 0, 1,&fogValue);
    validate_image_size(srcSize);
	unsigned int kernelSize = 3;
#ifdef OCL_COMPILE
 	{

 	blur_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
    fog_cl(static_cast<cl_mem>(dstPtr),
			srcSize,
			fogValue,
			RPPI_CHN_PLANAR, 3, 
            static_cast<cl_command_queue>(rppHandle) , static_cast<cl_mem>(srcPtr) );
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_fog_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32f fogValue, RppHandle_t rppHandle)
{
 	validate_float_range( 0, 1,&fogValue);
    validate_image_size(srcSize);
	unsigned int kernelSize = 3;
#ifdef OCL_COMPILE
 	{

 	blur_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
    fog_cl(static_cast<cl_mem>(dstPtr),
			srcSize,
			fogValue,
			RPPI_CHN_PACKED, 3, 
            static_cast<cl_command_queue>(rppHandle) , static_cast<cl_mem>(srcPtr) );
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

// ----------------------------------------
// Host exposure functions calls
// ----------------------------------------

RppStatus
rppi_exposure_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f exposureValue)
{
	 RppStatus status;
 	 validate_image_size(srcSize);
 	 validate_float_range( -4, 4, &exposureValue);

	 status = exposure_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			exposureValue,
			RPPI_CHN_PLANAR, 1);

	return status;
}

RppStatus
rppi_exposure_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f exposureValue)
{

	 RppStatus status;
 	 validate_image_size(srcSize);
 	 validate_float_range( -4, 4, &exposureValue);

	 status = exposure_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			exposureValue,
			RPPI_CHN_PLANAR, 3);

	return status;
}

RppStatus
rppi_exposure_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f exposureValue)
{

	 RppStatus status;
 	 validate_image_size(srcSize);
 	 validate_float_range( -4, 4, &exposureValue);

	 status = exposure_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			exposureValue,
			RPPI_CHN_PACKED, 3);

	return status;
}

// ----------------------------------------
// GPU exposure functions  calls
// ----------------------------------------


RppStatus
rppi_exposure_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f exposureValue, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( -4, 4, &exposureValue);

#ifdef OCL_COMPILE
 	 {
 	 exposure_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			exposureValue,
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
rppi_exposure_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f exposureValue, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( -4, 4, &exposureValue);

#ifdef OCL_COMPILE
 	 {
 	 exposure_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			exposureValue,
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
rppi_exposure_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f exposureValue, RppHandle_t rppHandle)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( -4, 4, &exposureValue);

#ifdef OCL_COMPILE
 	 {
 	 exposure_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr),
			exposureValue,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 }
#elif defined (HIP_COMPILE)
 	 {
 	 }
#endif //BACKEND
		return RPP_SUCCESS;
}


