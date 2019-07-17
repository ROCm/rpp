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

 	 validate_image_size(srcSize);
 	 validate_float_min(0, stdDev);
	 unsigned int kernelSize = 3;
	 blur_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			stdDev,
			kernelSize,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_blur_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev)
{

 	 validate_image_size(srcSize);
 	 validate_float_min(0, stdDev);
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
 	 validate_float_min(0, stdDev);
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

 	 validate_image_size(srcSize);
 	 validate_int_max(newMax, newMin);
	 contrast_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
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

 	 validate_image_size(srcSize);
 	 validate_int_max(newMax, newMin);
	 contrast_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			newMin,
			newMax,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_contrast_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u newMin,Rpp32u newMax)
{

 	 validate_image_size(srcSize);
 	 validate_int_max(newMax, newMin);
	 contrast_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
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

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 2, alpha);
 	 validate_float_range( 0, 255, beta);
	 brightness_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			alpha,
			beta,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_brightness_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha,Rpp32f beta)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 2, alpha);
 	 validate_float_range( 0, 255, beta);
	 brightness_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			alpha,
			beta,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_brightness_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha,Rpp32f beta)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 2, alpha);
 	 validate_float_range( 0, 255, beta);
	 brightness_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			alpha,
			beta,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host gamma_correction functions calls 
// ----------------------------------------


RppStatus
rppi_gamma_correction_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f gamma)
{

 	 validate_image_size(srcSize);
 	 validate_float_min(0, gamma);
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
 	 validate_float_min(0, gamma);
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
 	 validate_float_min(0, gamma);
	 gamma_correction_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			gamma,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}

// ----------------------------------------
// Host blend functions  calls 
// ----------------------------------------

RppStatus
rppi_blend_u8_pln1_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr, 
                        Rpp32f alpha)
{

     blend_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
                        static_cast<Rpp8u*>(dstPtr),
                        alpha, RPPI_CHN_PLANAR, 1);

    return RPP_SUCCESS;
}

RppStatus
rppi_blend_u8_pln3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr, 
                        Rpp32f alpha)
{

     blend_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
                        static_cast<Rpp8u*>(dstPtr),
                        alpha, RPPI_CHN_PLANAR, 3);

    return RPP_SUCCESS;
}

RppStatus
rppi_blend_u8_pkd3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr, 
                        Rpp32f alpha)
{

     blend_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
            static_cast<Rpp8u*>(dstPtr),
            alpha, RPPI_CHN_PACKED, 3);

    return RPP_SUCCESS;
}

// ----------------------------------------
// Host Noise functions  calls 
// ----------------------------------------

RppStatus
rppi_noiseAdd_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                            RppiNoise noiseType, void* noiseParameter)
{
    if(noiseType==GAUSSIAN)
    {
        noiseAdd_gaussian_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                                    noiseType, (RppiGaussParameter *)noiseParameter,
                                    RPPI_CHN_PLANAR, 1);
    }
    else if(noiseType==SNP)
    {
        noiseAdd_snp_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                                    noiseType, (Rpp32f *)noiseParameter,
                                    RPPI_CHN_PLANAR, 1);
    }
}

RppStatus
rppi_noiseAdd_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                            RppiNoise noiseType, void* noiseParameter)
{
    if(noiseType==GAUSSIAN)
    {
        noiseAdd_gaussian_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                            noiseType, (RppiGaussParameter *)noiseParameter,
                            RPPI_CHN_PLANAR, 3);
    }
    else if(noiseType==SNP)
    {
        noiseAdd_snp_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                                    noiseType, (Rpp32f *)noiseParameter,
                                    RPPI_CHN_PLANAR, 3);        
    }
}

RppStatus
rppi_noiseAdd_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                            RppiNoise noiseType, void* noiseParameter)
{
    if(noiseType==GAUSSIAN)
    {
        noiseAdd_gaussian_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                            noiseType, (RppiGaussParameter *)noiseParameter,
                            RPPI_CHN_PACKED, 3);
    }
    else if(noiseType==SNP)
    {
        noiseAdd_snp_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                                    noiseType, (Rpp32f *)noiseParameter,
                                    RPPI_CHN_PACKED, 3);        
    }
}

// ----------------------------------------
// Host fog functions call 
// ----------------------------------------

RppStatus
rppi_fog_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32f fogValue)
{
 	validate_image_size(srcSize);
    Rpp32f stdDev=fogValue*50;
 	validate_float_min(0, stdDev);
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


 	validate_image_size(srcSize);
    Rpp32f stdDev=fogValue*50;
 	validate_float_min(0, stdDev);
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
 	validate_image_size(srcSize);
    Rpp32f stdDev=fogValue*10;
 	validate_float_min(0, stdDev);
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
// Rain fog functions call 
// ----------------------------------------

RppStatus
rppi_rain_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32f rainValue, Rpp32u rainWidth, Rpp32u rainHeight)
{
    validate_image_size(srcSize);
    rain_host(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			rainValue, rainWidth, rainHeight,
			RPPI_CHN_PLANAR, 1);
    return RPP_SUCCESS;
}

RppStatus
rppi_rain_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32f rainValue, Rpp32u rainWidth, Rpp32u rainHeight)
{
    validate_image_size(srcSize);
    rain_host(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			rainValue, rainWidth, rainHeight,
			RPPI_CHN_PLANAR, 3);
        return RPP_SUCCESS;
}

RppStatus
rppi_rain_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32f rainValue, Rpp32u rainWidth, Rpp32u rainHeight)
{
    validate_image_size(srcSize);
    rain_host(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			rainValue, rainWidth, rainHeight,
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
 	 validate_float_min(0, stdDev);
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
 	 validate_float_min(0, stdDev);
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
 	 validate_float_min(0, stdDev);
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
 	 validate_int_max(newMax, newMin);

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
 	 validate_int_max(newMax, newMin);

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
 	 validate_int_max(newMax, newMin);

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
 	 validate_float_range( 0, 2, alpha);
 	 validate_float_range( 0, 255, beta);

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
 	 validate_float_range( 0, 2, alpha);
 	 validate_float_range( 0, 255, beta);

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
 	 validate_float_range( 0, 2, alpha);
 	 validate_float_range( 0, 255, beta);

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
 	 validate_float_min(0, gamma);

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
 	 validate_float_min(0, gamma);

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
 	 validate_float_min(0, gamma);

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
// GPU blend functions  calls 
// ----------------------------------------

RppStatus
rppi_blend_u8_pln1_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, RppHandle_t rppHandle)
{
   	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 blend_cl(static_cast<cl_mem>(srcPtr1), static_cast<cl_mem>(srcPtr2), 
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
rppi_blend_u8_pln3_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, RppHandle_t rppHandle)
{
   	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 blend_cl(static_cast<cl_mem>(srcPtr1), static_cast<cl_mem>(srcPtr2), 
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
rppi_blend_u8_pkd3_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, RppHandle_t rppHandle)
{
   	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 blend_cl(static_cast<cl_mem>(srcPtr1), static_cast<cl_mem>(srcPtr2), 
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
// GPU ADD NOISE functions  calls 
// ----------------------------------------

RppStatus
rppi_noiseAdd_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiNoise noiseType, void * noiseParameter, RppHandle_t rppHandle)
{
   	validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	{
 	    if(noiseType==GAUSSIAN)
            noise_add_gaussian_cl(static_cast<cl_mem>(srcPtr),
                srcSize,
                static_cast<cl_mem>(dstPtr), 
                noiseType,(RppiGaussParameter *)noiseParameter,
                RPPI_CHN_PLANAR, 1,
                static_cast<cl_command_queue>(rppHandle));
        else if(noiseType==SNP)
            noise_add_snp_cl(static_cast<cl_mem>(srcPtr), 
                srcSize,
                static_cast<cl_mem>(dstPtr), 
                noiseType,(Rpp32f *)noiseParameter,
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
rppi_noiseAdd_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiNoise noiseType, void * noiseParameter, RppHandle_t rppHandle)
{
   	validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	{
 	    if(noiseType==GAUSSIAN)
            noise_add_gaussian_cl(static_cast<cl_mem>(srcPtr),
                srcSize,
                static_cast<cl_mem>(dstPtr), 
                noiseType,(RppiGaussParameter *)noiseParameter,
                RPPI_CHN_PLANAR, 3,
                static_cast<cl_command_queue>(rppHandle));
        else if(noiseType==SNP)
            noise_add_snp_cl(static_cast<cl_mem>(srcPtr), 
                srcSize,
                static_cast<cl_mem>(dstPtr), 
                noiseType,(Rpp32f *)noiseParameter,
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
rppi_noiseAdd_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiNoise noiseType, void * noiseParameter, RppHandle_t rppHandle)
{
   	validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	{
 	    if(noiseType==GAUSSIAN)
            noise_add_gaussian_cl(static_cast<cl_mem>(srcPtr),
                srcSize,
                static_cast<cl_mem>(dstPtr), 
                noiseType,(RppiGaussParameter *)noiseParameter,
                RPPI_CHN_PACKED, 3,
                static_cast<cl_command_queue>(rppHandle));
        else if(noiseType==SNP)
            noise_add_snp_cl(static_cast<cl_mem>(srcPtr), 
                srcSize,
                static_cast<cl_mem>(dstPtr), 
                noiseType,(Rpp32f *)noiseParameter,
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
// Exposure modification functions  calls 
// ----------------------------------------

RppStatus
rppi_exposure_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f exposureValue, RppHandle_t rppHandle)
{
   	validate_image_size(srcSize);

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
rppi_exposure_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f exposureValue, RppHandle_t rppHandle)
{
   	validate_image_size(srcSize);

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
rppi_exposure_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f exposureValue, RppHandle_t rppHandle)
{
   	validate_image_size(srcSize);

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

RppStatus
rppi_occlusionAdd_u8_pln1_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize1, RppiSize srcSize2, RppPtr_t dstPtr, 
                               Rpp32u src1x1, Rpp32u src1y1, Rpp32u src1x2, Rpp32u src1y2, 
                               Rpp32u src2x1, Rpp32u src2y1, Rpp32u src2x2, Rpp32u src2y2, RppHandle_t rppHandle){

#ifdef OCL_COMPILE
 	{
        occlusion_cl( srcPtr1, srcPtr2,
                  srcSize1,  srcSize2, dstPtr, 
                 RPPI_CHN_PLANAR,src1x1, src1y1,
                src1x2, src1y2, src2x1, src2y1, src2x2, src2y2
                 1,
                 cl_command_queue theQueue)
       
 	} 
#elif defined (HIP_COMPILE) 
 	{ 
 	} 
#endif //BACKEND 
	}

RppStatus
rppi_occlusionAdd_u8_pln3_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize1, RppiSize srcSize2, RppPtr_t dstPtr, 
                               Rpp32u src1x1, Rpp32u src1y1, Rpp32u src1x2, Rpp32u src1y2, 
                               Rpp32u src2x1, Rpp32u src2y1, Rpp32u src2x2, Rpp32u src2y2, RppHandle_t rppHandle){
#ifdef OCL_COMPILE
 	{
        occlusion_cl( srcPtr1, srcPtr2,
                  srcSize1,  srcSize2, dstPtr, 
                 RPPI_CHN_PLANAR,src1x1, src1y1,
                src1x2, src1y2, src2x1, src2y1, src2x2, src2y2
                 3,
                 cl_command_queue theQueue)
 	} 
#elif defined (HIP_COMPILE) 
 	{ 
 	} 
#endif //BACKEND 
}

RppStatus
rppi_occlusionAdd_u8_pkd3_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize1, RppiSize srcSize2, RppPtr_t dstPtr, 
                               Rpp32u src1x1, Rpp32u src1y1, Rpp32u src1x2, Rpp32u src1y2, 
                               Rpp32u src2x1, Rpp32u src2y1, Rpp32u src2x2, Rpp32u src2y2, RppHandle_t rppHandle){
#ifdef OCL_COMPILE
 	{
       occlusion_cl( srcPtr1, srcPtr2,
                  srcSize1,  srcSize2, dstPtr, 
                 RPPI_CHN_PACKED,src1x1, src1y1,
                src1x2, src1y2, src2x1, src2y1, src2x2, src2y2
                 3,
                 cl_command_queue theQueue)
 	} 
#elif defined (HIP_COMPILE) 
 	{ 
 	} 
#endif //BACKEND 
	return RPP_SUCCESS;
}

/*RppStatus
rppi_histogram_balance_u8_pln1_gpu(RppPtr_t srcPtr, RppPtr_t dstPtr, 
                                    RppiSize srcSize, RppHandle_t rppHandle){
#ifdef OCL_COMPILE
 	{
       
 	} 
#elif defined (HIP_COMPILE) 
 	{ 
 	} 
#endif //BACKEND 
}

RppStatus
rppi_histogram_balance_u8_pln3_gpu(RppPtr_t srcPtr, RppPtr_t dstPtr, 
                                    RppiSize srcSize, RppHandle_t rppHandle){
#ifdef OCL_COMPILE
 	{
       
 	} 
#elif defined (HIP_COMPILE) 
 	{ 
 	} 
#endif //BACKEND 
}

RppStatus
rppi_histogram_balance_u8_pkd3_gpu(RppPtr_t srcPtr, RppPtr_t dstPtr, 
                                    RppiSize srcSize, RppHandle_t rppHandle){
#ifdef OCL_COMPILE
 	{
       
 	} 
#elif defined (HIP_COMPILE) 
 	{ 
 	} 
#endif //BACKEND 

}*/
