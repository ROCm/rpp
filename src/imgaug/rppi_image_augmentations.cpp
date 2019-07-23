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



// // ----------------------------------------
// Host jitter functions calls 
// ----------------------------------------


RppStatus
rppi_jitter_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             unsigned int maxJitterX, unsigned int maxJitterY)
{
    jitter_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     maxJitterX, maxJitterY, 
                     RPPI_CHN_PLANAR, 1);
    return RPP_SUCCESS;
}

RppStatus
rppi_jitter_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             unsigned int maxJitterX, unsigned int maxJitterY)
{
    jitter_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     maxJitterX, maxJitterY, 
                     RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_jitter_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             unsigned int maxJitterX, unsigned int maxJitterY)
{
    jitter_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     maxJitterX, maxJitterY, 
                     RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}
 



// ----------------------------------------
// Host Noise functions  calls 
// ----------------------------------------

// RppStatus
// rppi_noise_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
//                             RppiNoise noiseType, void* noiseParameter)
// {
//     if(noiseType==GAUSSIAN)
//     {
//         noise_gaussian_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
//                                     noiseType, (RppiGaussParameter *)noiseParameter,
//                                     RPPI_CHN_PLANAR, 1);
//     }
//     else if(noiseType==SNP)
//     {
//         noise_snp_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
//                                     noiseType, (Rpp32f *)noiseParameter,
//                                     RPPI_CHN_PLANAR, 1);
//     }
// }

// RppStatus
// rppi_noise_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
//                             RppiNoise noiseType, void* noiseParameter)
// {
//     if(noiseType==GAUSSIAN)
//     {
        // noise_gaussian_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
        //                     noiseType, (RppiGaussParameter *)noiseParameter,
        //                     RPPI_CHN_PLANAR, 3);
//     }
//     else if(noiseType==SNP)
//     {
//         noise_snp_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
//                                     noiseType, (Rpp32f *)noiseParameter,
//                                     RPPI_CHN_PLANAR, 3);        
//     }
// }

// RppStatus
// rppi_noise_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
//                             RppiNoise noiseType, void* noiseParameter)
// {
//     if(noiseType==GAUSSIAN)
//     {
//         noise_gaussian_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
//                             noiseType, (RppiGaussParameter *)noiseParameter,
//                             RPPI_CHN_PACKED, 3);
//     }
//     else if(noiseType==SNP)
//     {
//         noise_snp_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
//                                     noiseType, (Rpp32f *)noiseParameter,
//                                     RPPI_CHN_PACKED, 3);        
//     }
// }

RppStatus
rppi_snpNoise_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f noiseProbability)
{
    noise_snp_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                        noiseProbability,
                        RPPI_CHN_PLANAR, 1);
}

RppStatus
rppi_snpNoise_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f noiseProbability)
{
    noise_snp_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                        noiseProbability,
                        RPPI_CHN_PLANAR, 3);
}

RppStatus
rppi_snpNoise_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f noiseProbability)
{
    noise_snp_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                        noiseProbability,
                        RPPI_CHN_PACKED, 3);
}

RppStatus
rppi_gaussianNoise_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f mean, Rpp32f sigma)
{
    noise_gaussian_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                        mean, sigma, 
                        RPPI_CHN_PLANAR, 1);
}

RppStatus
rppi_gaussianNoise_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f mean, Rpp32f sigma)
{
    noise_gaussian_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                        mean, sigma, 
                        RPPI_CHN_PLANAR, 3);
}

RppStatus
rppi_gaussianNoise_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f mean, Rpp32f sigma)
{
    noise_gaussian_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                        mean, sigma, 
                        RPPI_CHN_PACKED, 3);
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
// Host rain functions calls 
// ----------------------------------------


RppStatus
rppi_rain_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f rainPercentage,Rpp32u rainWidth,Rpp32u rainHeight, Rpp32f transparency)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, rainPercentage);
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
 	 validate_float_range( 0, 1, rainPercentage);
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
 	 validate_float_range( 0, 1, rainPercentage);
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
 	 validate_float_range( 0, 1, snowValue);
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
 	 validate_float_range( 0, 1, snowValue);
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
 	 validate_float_range( 0, 1, snowValue);
	 snow_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			snowValue,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host random_shadow functions calls 
// ----------------------------------------


RppStatus
rppi_random_shadow_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1 ,Rpp32u y1,Rpp32u x2,Rpp32u y2,Rpp32u numberOfShadows,Rpp32u maxSizeX,Rpp32u maxSizeY)
{

 	 validate_image_size(srcSize);
 	 validate_int_range( 0, srcSize.width - 1, x1);
 	 validate_int_range( 0, srcSize.height - 1, y1);
 	 validate_int_range( 0, srcSize.width - 1, x2);
 	 validate_int_range( 0, srcSize.height - 1, y2);
 	 validate_int_min(1, numberOfShadows);
	 random_shadow_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			x1,
			y1,
			x2,
			y2,
			numberOfShadows,
			maxSizeX,
			maxSizeY,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_random_shadow_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1 ,Rpp32u y1,Rpp32u x2,Rpp32u y2,Rpp32u numberOfShadows,Rpp32u maxSizeX,Rpp32u maxSizeY)
{

 	 validate_image_size(srcSize);
 	 validate_int_range( 0, srcSize.width - 1, x1);
 	 validate_int_range( 0, srcSize.height - 1, y1);
 	 validate_int_range( 0, srcSize.width - 1, x2);
 	 validate_int_range( 0, srcSize.height - 1, y2);
 	 validate_int_min(1, numberOfShadows);
	 random_shadow_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			x1,
			y1,
			x2,
			y2,
			numberOfShadows,
			maxSizeX,
			maxSizeY,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_random_shadow_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2,Rpp32u numberOfShadows,Rpp32u maxSizeX,Rpp32u maxSizeY)
{

 	 validate_image_size(srcSize);
 	 validate_int_range( 0, srcSize.width - 1, x1);
 	 validate_int_range( 0, srcSize.height - 1, y1);
 	 validate_int_range( 0, srcSize.width - 1, x2);
 	 validate_int_range( 0, srcSize.height - 1, y2);
 	 validate_int_min(1, numberOfShadows);
	 random_shadow_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			x1,
			y1,
			x2,
			y2,
			numberOfShadows,
			maxSizeX,
			maxSizeY,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host blend functions calls 
// ----------------------------------------


RppStatus
rppi_blend_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, alpha);
	 blend_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), 
			static_cast<Rpp8u*>(srcPtr2), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			alpha,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_blend_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, alpha);
	 blend_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), 
			static_cast<Rpp8u*>(srcPtr2), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			alpha,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_blend_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, alpha);
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
rppi_pixelate_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2 )
{

 	 validate_image_size(srcSize);
 	 validate_int_range( 0, srcSize.width - 1, x1);
 	 validate_int_range( 0, srcSize.height - 1, y1);
 	 validate_int_range( 0, srcSize.width - 1, x2);
 	 validate_int_range( 0, srcSize.height - 1, y2);
	 Rpp32u kernelSize = 3;
	 pixelate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr),
			kernelSize,
			x1,
			y1,
			x2,
			y2,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_pixelate_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1 ,Rpp32u y1,Rpp32u x2,Rpp32u y2 )
{

 	 validate_image_size(srcSize);
 	 validate_int_range( 0, srcSize.width - 1, x1);
 	 validate_int_range( 0, srcSize.height - 1, y1);
 	 validate_int_range( 0, srcSize.width - 1, x2);
 	 validate_int_range( 0, srcSize.height - 1, y2);
	 Rpp32u kernelSize = 3;
	 pixelate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			kernelSize,
			x1,
			y1,
			x2,
			y2,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_pixelate_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1 ,Rpp32u y1,Rpp32u x2,Rpp32u y2 )
{

 	 validate_image_size(srcSize);
 	 validate_int_range( 0, srcSize.width - 1, x1);
 	 validate_int_range( 0, srcSize.height - 1, y1);
 	 validate_int_range( 0, srcSize.width - 1, x2);
 	 validate_int_range( 0, srcSize.height - 1, y2);
	 Rpp32u kernelSize = 3;
	 pixelate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			kernelSize,
			x1,
			y1,
			x2,
			y2,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host random_crop_letterbox functions calls 
// ----------------------------------------


RppStatus
rppi_random_crop_letterbox_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2)
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
 	 validate_int_range( 0, srcSize.width - 1, x1);
 	 validate_int_range( 0, srcSize.height - 1, y1);
 	 validate_int_range( 0, srcSize.width - 1, x2);
 	 validate_int_range( 0, srcSize.height - 1, y2);
	 random_crop_letterbox_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			dstSize,
			x1,
			y1,
			x2,
			y2,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_random_crop_letterbox_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2)
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
 	 validate_int_range( 0, srcSize.width - 1, x1);
 	 validate_int_range( 0, srcSize.height - 1, y1);
 	 validate_int_range( 0, srcSize.width - 1, x2);
 	 validate_int_range( 0, srcSize.height - 1, y2);
	 random_crop_letterbox_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			dstSize,
			x1,
			y1,
			x2,
			y2,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_random_crop_letterbox_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2)
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
 	 validate_int_range( 0, srcSize.width - 1, x1);
 	 validate_int_range( 0, srcSize.height - 1, y1);
 	 validate_int_range( 0, srcSize.width - 1, x2);
 	 validate_int_range( 0, srcSize.height - 1, y2);
	 random_crop_letterbox_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			dstSize,
			x1,
			y1,
			x2,
			y2,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host occlusion functions calls 
// ----------------------------------------


RppStatus
rppi_occlusion_u8_pln1_host(RppPtr_t srcPtr1,RppiSize srcSize1,RppPtr_t srcPtr2,RppiSize srcSize2,RppPtr_t dstPtr,Rpp32u src1x1,Rpp32u src1y1,Rpp32u src1x2,Rpp32u src1y2,Rpp32u src2x1,Rpp32u src2y1,Rpp32u src2x2,Rpp32u src2y2)
{

 	 validate_image_size(srcSize1);
 	 validate_image_size(srcSize2);
 	 validate_int_range( 0, srcSize1.width - 1, src1x1);
 	 validate_int_range( 0, srcSize1.height - 1, src1y1);
 	 validate_int_range( 0, srcSize1.width - 1, src1x2);
 	 validate_int_range( 0, srcSize1.height - 1, src1y2);
 	 validate_int_range( 0, srcSize1.width - 1, src2x1);
 	 validate_int_range( 0, srcSize1.height - 1, src2y1);
 	 validate_int_range( 0, srcSize1.width - 1, src2x2);
 	 validate_int_range( 0, srcSize1.height - 1, src2y2);
	 occlusion_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), 
			srcSize1,
			static_cast<Rpp8u*>(srcPtr2), 
			srcSize2,
			static_cast<Rpp8u*>(dstPtr), 
			src1x1,
			src1y1,
			src1x2,
			src1y2,
			src2x1,
			src2y1,
			src2x2,
			src2y2,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_occlusion_u8_pln3_host(RppPtr_t srcPtr1,RppiSize srcSize1,RppPtr_t srcPtr2,RppiSize srcSize2,RppPtr_t dstPtr,Rpp32u src1x1,Rpp32u src1y1,Rpp32u src1x2,Rpp32u src1y2,Rpp32u src2x1,Rpp32u src2y1,Rpp32u src2x2,Rpp32u src2y2)
{

 	 validate_image_size(srcSize1);
 	 validate_image_size(srcSize2);
 	 validate_int_range( 0, srcSize1.width - 1, src1x1);
 	 validate_int_range( 0, srcSize1.height - 1, src1y1);
 	 validate_int_range( 0, srcSize1.width - 1, src1x2);
 	 validate_int_range( 0, srcSize1.height - 1, src1y2);
 	 validate_int_range( 0, srcSize1.width - 1, src2x1);
 	 validate_int_range( 0, srcSize1.height - 1, src2y1);
 	 validate_int_range( 0, srcSize1.width - 1, src2x2);
 	 validate_int_range( 0, srcSize1.height - 1, src2y2);
	 occlusion_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), 
			srcSize1,
			static_cast<Rpp8u*>(srcPtr2), 
			srcSize2,
			static_cast<Rpp8u*>(dstPtr), 
			src1x1,
			src1y1,
			src1x2,
			src1y2,
			src2x1,
			src2y1,
			src2x2,
			src2y2,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_occlusion_u8_pkd3_host(RppPtr_t srcPtr1,RppiSize srcSize1,RppPtr_t srcPtr2,RppiSize srcSize2,RppPtr_t dstPtr,Rpp32u src1x1,Rpp32u src1y1,Rpp32u src1x2,Rpp32u src1y2,Rpp32u src2x1,Rpp32u src2y1,Rpp32u src2x2,Rpp32u src2y2)
{

 	 validate_image_size(srcSize1);
 	 validate_image_size(srcSize2);
 	 validate_int_range( 0, srcSize1.width - 1, src1x1);
 	 validate_int_range( 0, srcSize1.height - 1, src1y1);
 	 validate_int_range( 0, srcSize1.width - 1, src1x2);
 	 validate_int_range( 0, srcSize1.height - 1, src1y2);
 	 validate_int_range( 0, srcSize1.width - 1, src2x1);
 	 validate_int_range( 0, srcSize1.height - 1, src2y1);
 	 validate_int_range( 0, srcSize1.width - 1, src2x2);
 	 validate_int_range( 0, srcSize1.height - 1, src2y2);
	 occlusion_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), 
			srcSize1,
			static_cast<Rpp8u*>(srcPtr2), 
			srcSize2,
			static_cast<Rpp8u*>(dstPtr), 
			src1x1,
			src1y1,
			src1x2,
			src1y2,
			src2x1,
			src2y1,
			src2x2,
			src2y2,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host exposure functions calls 
// ----------------------------------------

RppStatus
rppi_exposure_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f exposureValue)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( -4, 4, exposureValue);
	 exposure_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			exposureValue,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_exposure_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f exposureValue)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( -4, 4, exposureValue);
	 exposure_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			exposureValue,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_exposure_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f exposureValue)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( -4, 4, exposureValue);
	 exposure_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			exposureValue,
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
//----------------------------------------
//GPU fog functions  calls 
//----------------------------------------


// RppStatus
// rppi_fog_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f fogValue, RppHandle_t rppHandle) 
// {

//  	 validate_image_size(srcSize);
//  	 validate_float_range( 0, 1, fogValue);

// #ifdef OCL_COMPILE
//  	 {
//  	 fog_cl(static_cast<cl_mem>(srcPtr), 
// 			srcSize,
// 			static_cast<cl_mem>(dstPtr), 
// 			fogValue,
// 			RPPI_CHN_PLANAR, 1,
// 			static_cast<cl_command_queue>(rppHandle));
//  	 } 
// #elif defined (HIP_COMPILE) 
//  	 { 
//  	 } 
// #endif //BACKEND 
// 		return RPP_SUCCESS;
// }

// RppStatus
// rppi_fog_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f fogValue, RppHandle_t rppHandle) 
// {

//  	 validate_image_size(srcSize);
//  	 validate_float_range( 0, 1, fogValue);

// #ifdef OCL_COMPILE
//  	 {
//  	 fog_cl(static_cast<cl_mem>(srcPtr), 
// 			srcSize,
// 			static_cast<cl_mem>(dstPtr), 
// 			fogValue,
// 			RPPI_CHN_PLANAR, 3,
// 			static_cast<cl_command_queue>(rppHandle));
//  	 } 
// #elif defined (HIP_COMPILE) 
//  	 { 
//  	 } 
// #endif //BACKEND 
// 		return RPP_SUCCESS;
// }

// RppStatus
// rppi_fog_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f fogValue, RppHandle_t rppHandle) 
// {

//  	 validate_image_size(srcSize);
//  	 validate_float_range( 0, 1, fogValue);

// #ifdef OCL_COMPILE
//  	 {
//  	 fog_cl(static_cast<cl_mem>(srcPtr), 
// 			srcSize,
// 			static_cast<cl_mem>(dstPtr), 
// 			fogValue,
// 			RPPI_CHN_PACKED, 3,
// 			static_cast<cl_command_queue>(rppHandle));
//  	 } 
// #elif defined (HIP_COMPILE) 
//  	 { 
//  	 } 
// #endif //BACKEND 
// 		return RPP_SUCCESS;
// }
 
// ----------------------------------------
// GPU rain functions  calls 
// ----------------------------------------


RppStatus
rppi_rain_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f rainPercentage,Rpp32u rainWidth,Rpp32u rainHeight, Rpp32f transparency, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, rainPercentage);

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
 	 validate_float_range( 0, 1, rainPercentage);

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
 	 validate_float_range( 0, 1, rainPercentage);

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
 	 validate_float_range( 0, 1, snowValue);
#ifdef OCL_COMPILE
 	 {

		cl_context theContext;
		cl_int err;
		size_t bytes1C = sizeof(unsigned char)*srcSize.width * srcSize.height;
		size_t bytes3C = sizeof(unsigned char)*srcSize.width * srcSize.height * 3;
		clGetCommandQueueInfo(  static_cast<cl_command_queue>(rppHandle),
								CL_QUEUE_CONTEXT,
								sizeof(cl_context), &theContext, NULL);
		cl_mem src3C = clCreateBuffer(theContext, CL_MEM_READ_ONLY,
										sizeof(unsigned char)*srcSize.width * srcSize.height * 3, NULL, NULL);
		cl_mem dst3C = clCreateBuffer(theContext, CL_MEM_READ_WRITE,
										sizeof(unsigned char)*srcSize.width * srcSize.height * 3, NULL, NULL);
		err = clEnqueueCopyBuffer(static_cast<cl_command_queue>(rppHandle), static_cast<cl_mem>(srcPtr), src3C, 0, 0,
									sizeof(unsigned char)*srcSize.width*srcSize.height,
							        0, NULL, NULL);
		err = clEnqueueCopyBuffer(static_cast<cl_command_queue>(rppHandle), static_cast<cl_mem>(srcPtr), src3C,
								 0, sizeof(unsigned char) * srcSize.width*srcSize.height,
								 sizeof(unsigned char)*srcSize.width*srcSize.height,
							     0, NULL, NULL);
		err = clEnqueueCopyBuffer(static_cast<cl_command_queue>(rppHandle), static_cast<cl_mem>(srcPtr), src3C,
								  0, sizeof(unsigned char) * srcSize.width*srcSize.height * 2,
								  sizeof(unsigned char)*srcSize.width*srcSize.height,
							      0, NULL, NULL);
		snow_cl(src3C, 
				srcSize,
				dst3C, 
				snowValue,
				RPPI_CHN_PLANAR, 1,
				static_cast<cl_command_queue>(rppHandle));
		
		err = clEnqueueCopyBuffer(static_cast<cl_command_queue>(rppHandle), dst3C, static_cast<cl_mem>(dstPtr),  0, 0,
									sizeof(unsigned char)*srcSize.width*srcSize.height,
							        0, NULL, NULL);
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
 	 validate_float_range( 0, 1, snowValue);

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
 	 validate_float_range( 0, 1, snowValue);

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
// GPU random_shadow functions  calls 
// ----------------------------------------


RppStatus
rppi_random_shadow_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1 ,Rpp32u y1,Rpp32u x2,Rpp32u y2,Rpp32u numberOfShadows,Rpp32u maxSizeX,Rpp32u maxSizeY, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_range( 0, srcSize.width - 1, x1);
 	 validate_int_range( 0, srcSize.height - 1, y1);
 	 validate_int_range( 0, srcSize.width - 1, x2);
 	 validate_int_range( 0, srcSize.height - 1, y2);
 	 validate_int_min(1, numberOfShadows);

#ifdef OCL_COMPILE
 	 {
 	 random_shadow_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			x1,
			y1,
			x2,
			y2,
			numberOfShadows,
			maxSizeX,
			maxSizeY,
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
rppi_random_shadow_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1 ,Rpp32u y1,Rpp32u x2,Rpp32u y2,Rpp32u numberOfShadows,Rpp32u maxSizeX,Rpp32u maxSizeY, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_range( 0, srcSize.width - 1, x1);
 	 validate_int_range( 0, srcSize.height - 1, y1);
 	 validate_int_range( 0, srcSize.width - 1, x2);
 	 validate_int_range( 0, srcSize.height - 1, y2);
 	 validate_int_min(1, numberOfShadows);

#ifdef OCL_COMPILE
 	 {
 	 random_shadow_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			x1,
			y1,
			x2,
			y2,
			numberOfShadows,
			maxSizeX,
			maxSizeY,
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
rppi_random_shadow_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1 ,Rpp32u y1,Rpp32u x2,Rpp32u y2,Rpp32u numberOfShadows,Rpp32u maxSizeX,Rpp32u maxSizeY, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_range( 0, srcSize.width - 1, x1);
 	 validate_int_range( 0, srcSize.height - 1, y1);
 	 validate_int_range( 0, srcSize.width - 1, x2);
 	 validate_int_range( 0, srcSize.height - 1, y2);
 	 validate_int_min(1, numberOfShadows);

#ifdef OCL_COMPILE
 	 {
 	 random_shadow_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			x1,
			y1,
			x2,
			y2,
			numberOfShadows,
			maxSizeX,
			maxSizeY,
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
 	 validate_float_range( 0, 1, alpha);

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
 	 validate_float_range( 0, 1, alpha);

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
 	 validate_float_range( 0, 1, alpha);

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
rppi_pixelate_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1 ,Rpp32u y1,Rpp32u x2,Rpp32u y2 , RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_range( 0, srcSize.width - 1, x1);
 	 validate_int_range( 0, srcSize.height - 1, y1);
 	 validate_int_range( 0, srcSize.width - 1, x2);
 	 validate_int_range( 0, srcSize.height - 1, y2);
	 Rpp32u kernelSize = 3;
#ifdef OCL_COMPILE
 	 {
 	 pixelate_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize, 
			x1,
			y1,
			x2,
			y2,
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
rppi_pixelate_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1 ,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_range( 0, srcSize.width - 1, x1);
 	 validate_int_range( 0, srcSize.height - 1, y1);
 	 validate_int_range( 0, srcSize.width - 1, x2);
 	 validate_int_range( 0, srcSize.height - 1, y2);
	 Rpp32u kernelSize = 3;

#ifdef OCL_COMPILE
 	 {
 	 pixelate_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			kernelSize,
			x1,
			y1,
			x2,
			y2,
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
rppi_pixelate_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1 ,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_range( 0, srcSize.width - 1, x1);
 	 validate_int_range( 0, srcSize.height - 1, y1);
 	 validate_int_range( 0, srcSize.width - 1, x2);
 	 validate_int_range( 0, srcSize.height - 1, y2);
	 Rpp32u kernelSize = 3;

#ifdef OCL_COMPILE
 	 {
 	 pixelate_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr),
			kernelSize, 
			x1,
			y1,
			x2,
			y2,
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
// GPU random_crop_letterbox functions  calls 
// ----------------------------------------


RppStatus
rppi_random_crop_letterbox_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
 	 validate_int_range( 0, srcSize.width - 1, x1);
 	 validate_int_range( 0, srcSize.height - 1, y1);
 	 validate_int_range( 0, srcSize.width - 1, x2);
 	 validate_int_range( 0, srcSize.height - 1, y2);

#ifdef OCL_COMPILE
 	 {
 	 random_crop_letterbox_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			dstSize,
			x1,
			y1,
			x2,
			y2,
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
rppi_random_crop_letterbox_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
 	 validate_int_range( 0, srcSize.width - 1, x1);
 	 validate_int_range( 0, srcSize.height - 1, y1);
 	 validate_int_range( 0, srcSize.width - 1, x2);
 	 validate_int_range( 0, srcSize.height - 1, y2);

#ifdef OCL_COMPILE
 	 {
 	 random_crop_letterbox_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			dstSize,
			x1,
			y1,
			x2,
			y2,
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
rppi_random_crop_letterbox_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
 	 validate_int_range( 0, srcSize.width - 1, x1);
 	 validate_int_range( 0, srcSize.height - 1, y1);
 	 validate_int_range( 0, srcSize.width - 1, x2);
 	 validate_int_range( 0, srcSize.height - 1, y2);

#ifdef OCL_COMPILE
 	 {
 	 random_crop_letterbox_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			dstSize,
			x1,
			y1,
			x2,
			y2,
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
// GPU occlusion functions  calls 
// ----------------------------------------


RppStatus
rppi_occlusion_u8_pln1_gpu(RppPtr_t srcPtr1,RppiSize srcSize1,RppPtr_t srcPtr2,RppiSize srcSize2,RppPtr_t dstPtr,Rpp32u src1x1,Rpp32u src1y1,Rpp32u src1x2,Rpp32u src1y2,Rpp32u src2x1,Rpp32u src2y1,Rpp32u src2x2,Rpp32u src2y2, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize1);
 	 validate_image_size(srcSize2);
 	 validate_int_range( 0, srcSize1.width - 1, src1x1);
 	 validate_int_range( 0, srcSize1.height - 1, src1y1);
 	 validate_int_range( 0, srcSize1.width - 1, src1x2);
 	 validate_int_range( 0, srcSize1.height - 1, src1y2);
 	 validate_int_range( 0, srcSize1.width - 1, src2x1);
 	 validate_int_range( 0, srcSize1.height - 1, src2y1);
 	 validate_int_range( 0, srcSize1.width - 1, src2x2);
 	 validate_int_range( 0, srcSize1.height - 1, src2y2);

#ifdef OCL_COMPILE
 	 {
 	 occlusion_cl(static_cast<cl_mem>(srcPtr1), 
			srcSize1,
			static_cast<cl_mem>(srcPtr2), 
			srcSize2,
			static_cast<cl_mem>(dstPtr), 
			src1x1,
			src1y1,
			src1x2,
			src1y2,
			src2x1,
			src2y1,
			src2x2,
			src2y2,
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
rppi_occlusion_u8_pln3_gpu(RppPtr_t srcPtr1,RppiSize srcSize1,RppPtr_t srcPtr2,RppiSize srcSize2,RppPtr_t dstPtr,
			Rpp32u src1x1,Rpp32u src1y1,Rpp32u src1x2,Rpp32u src1y2,Rpp32u src2x1,Rpp32u src2y1,
			Rpp32u src2x2,Rpp32u src2y2, RppHandle_t rppHandle) 
{

 	 /*validate_image_size(srcSize1);
 	 validate_image_size(srcSize2);
 	 validate_int_range( 0, srcSize1.width - 1, src1x1);
 	 validate_int_range( 0, srcSize1.height - 1, src1y1);
 	 validate_int_range( 0, srcSize1.width - 1, src1x2);
 	 validate_int_range( 0, srcSize1.height - 1, src1y2);
 	 validate_int_range( 0, srcSize1.width - 1, src2x1);
 	 validate_int_range( 0, srcSize1.height - 1, src2y1);
 	 validate_int_range( 0, srcSize1.width - 1, src2x2);
 	 validate_int_range( 0, srcSize1.height - 1, src2y2);*/

#ifdef OCL_COMPILE
 	 {
 	 occlusion_cl(static_cast<cl_mem>(srcPtr1), 
			srcSize1,
			static_cast<cl_mem>(srcPtr2), 
			srcSize2,
			static_cast<cl_mem>(dstPtr), 
			src1x1,
			src1y1,
			src1x2,
			src1y2,
			src2x1,
			src2y1,
			src2x2,
			src2y2,
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
rppi_occlusion_u8_pkd3_gpu(RppPtr_t srcPtr1,RppiSize srcSize1,RppPtr_t srcPtr2,RppiSize srcSize2,RppPtr_t dstPtr,Rpp32u src1x1,Rpp32u src1y1,Rpp32u src1x2,Rpp32u src1y2,Rpp32u src2x1,Rpp32u src2y1,Rpp32u src2x2,Rpp32u src2y2, RppHandle_t rppHandle) 
{

 	 /*validate_image_size(srcSize1);
 	 validate_image_size(srcSize2);
 	 validate_int_range( 0, srcSize1.width - 1, src1x1);
 	 validate_int_range( 0, srcSize1.height - 1, src1y1);
 	 validate_int_range( 0, srcSize1.width - 1, src1x2);
 	 validate_int_range( 0, srcSize1.height - 1, src1y2);
 	 validate_int_range( 0, srcSize1.width - 1, src2x1);
 	 validate_int_range( 0, srcSize1.height - 1, src2y1);
 	 validate_int_range( 0, srcSize1.width - 1, src2x2);
 	 validate_int_range( 0, srcSize1.height - 1, src2y2);*/

#ifdef OCL_COMPILE
 	 {
	
 	 occlusion_cl(static_cast<cl_mem>(srcPtr1), 
			srcSize1,
			static_cast<cl_mem>(srcPtr2), 
			srcSize2,
			static_cast<cl_mem>(dstPtr), 
			src1x1,
			src1y1,
			src1x2,
			src1y2,
			src2x1,
			src2y1,
			src2x2,
			src2y2,
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
// GPU exposure functions  calls 
// ----------------------------------------


RppStatus
rppi_exposure_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f exposureValue, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_float_range( -4, 4, exposureValue);

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
 	 validate_float_range( -4, 4, exposureValue);

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
 	 validate_float_range( -4, 4, exposureValue);

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

// ----------------------------------------
// GPU jitter functions  calls 
// ----------------------------------------
RppStatus
rppi_jitter_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u minJitter, Rpp32u maxJitter, RppHandle_t rppHandle)
{
   	 validate_image_size(srcSize);
	 validate_int_range(0,255,minJitter);
	 validate_int_range(0,255,maxJitter);
	 validate_int_min(minJitter, maxJitter);

#ifdef OCL_COMPILE
 	 {
 	 jitter_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			minJitter,
			maxJitter,
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
rppi_jitter_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u minJitter, Rpp32u maxJitter, RppHandle_t rppHandle)
{
   	 validate_image_size(srcSize);
	 validate_int_range(0,255,minJitter);
	 validate_int_range(0,255,maxJitter);
	 validate_int_min(minJitter, maxJitter);
#ifdef OCL_COMPILE
 	 {
 	 jitter_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			minJitter,
			maxJitter,
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
rppi_jitter_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u minJitter, Rpp32u maxJitter, RppHandle_t rppHandle)
{
   	 validate_image_size(srcSize);
	 validate_int_range(0,255,minJitter);
	 validate_int_range(0,255,maxJitter);
	 validate_int_min(minJitter, maxJitter);
#ifdef OCL_COMPILE
 	 {
 	 jitter_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			minJitter,
			maxJitter,
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

// RppStatus
// rppi_noise_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiNoise noiseType, void * noiseParameter, RppHandle_t rppHandle)
// {
//    	validate_image_size(srcSize);

// #ifdef OCL_COMPILE
//  	{
//  	    if(noiseType==GAUSSIAN)
//             noise_add_gaussian_cl(static_cast<cl_mem>(srcPtr),
//                 srcSize,
//                 static_cast<cl_mem>(dstPtr), 
//                 noiseType,(RppiGaussParameter *)noiseParameter,
//                 RPPI_CHN_PLANAR, 1,
//                 static_cast<cl_command_queue>(rppHandle));
//         else if(noiseType==SNP)
//             noise_add_snp_cl(static_cast<cl_mem>(srcPtr), 
//                 srcSize,
//                 static_cast<cl_mem>(dstPtr), 
//                 noiseType,(Rpp32f *)noiseParameter,
//                 RPPI_CHN_PLANAR, 1,
//                 static_cast<cl_command_queue>(rppHandle));
//  	} 
// #elif defined (HIP_COMPILE) 
//  	{ 
//  	} 
// #endif //BACKEND 
// 	return RPP_SUCCESS;
// }

// RppStatus
// rppi_noise_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiNoise noiseType, void * noiseParameter, RppHandle_t rppHandle)
// {
//    	validate_image_size(srcSize);

// #ifdef OCL_COMPILE
//  	{
//  	    if(noiseType==GAUSSIAN)
//             noise_add_gaussian_cl(static_cast<cl_mem>(srcPtr),
//                 srcSize,
//                 static_cast<cl_mem>(dstPtr), 
//                 noiseType,(RppiGaussParameter *)noiseParameter,
//                 RPPI_CHN_PLANAR, 3,
//                 static_cast<cl_command_queue>(rppHandle));
//         else if(noiseType==SNP)
//             noise_add_snp_cl(static_cast<cl_mem>(srcPtr), 
//                 srcSize,
//                 static_cast<cl_mem>(dstPtr), 
//                 noiseType,(Rpp32f *)noiseParameter,
//                 RPPI_CHN_PLANAR, 3,
//                 static_cast<cl_command_queue>(rppHandle));
//  	} 
// #elif defined (HIP_COMPILE) 
//  	{ 
//  	} 
// #endif //BACKEND 
// 	return RPP_SUCCESS;
// }

// RppStatus
// rppi_noise_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiNoise noiseType, void * noiseParameter, RppHandle_t rppHandle)
// {
//    	validate_image_size(srcSize);

// #ifdef OCL_COMPILE
//  	{
//  	    if(noiseType==GAUSSIAN)
//             noise_add_gaussian_cl(static_cast<cl_mem>(srcPtr),
//                 srcSize,
//                 static_cast<cl_mem>(dstPtr), 
//                 noiseType,(RppiGaussParameter *)noiseParameter,
//                 RPPI_CHN_PACKED, 3,
//                 static_cast<cl_command_queue>(rppHandle));
//         else if(noiseType==SNP)
//             noise_add_snp_cl(static_cast<cl_mem>(srcPtr), 
//                 srcSize,
//                 static_cast<cl_mem>(dstPtr), 
//                 noiseType,(Rpp32f *)noiseParameter,
//                 RPPI_CHN_PACKED, 3,
//                 static_cast<cl_command_queue>(rppHandle));
//  	} 
// #elif defined (HIP_COMPILE) 
//  	{ 
//  	} 
// #endif //BACKEND 
// 	return RPP_SUCCESS;
// }


RppStatus
rppi_snpNoise_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f noiseProbability, RppHandle_t rppHandle)
{
   	validate_image_size(srcSize);

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
RppStatus
rppi_gaussianNoise_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f mean, Rpp32f sigma, RppHandle_t rppHandle)
{
   	validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	{
            gaussianNoise_cl(static_cast<cl_mem>(srcPtr),
                srcSize,
                static_cast<cl_mem>(dstPtr), 
                mean, sigma,
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
rppi_gaussianNoise_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f mean, Rpp32f sigma, RppHandle_t rppHandle)
{
   	validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	{
            gaussianNoise_cl(static_cast<cl_mem>(srcPtr),
                srcSize,
                static_cast<cl_mem>(dstPtr), 
                mean, sigma,
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
rppi_gaussianNoise_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f mean, Rpp32f sigma, RppHandle_t rppHandle)
{
   	validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	{
            gaussianNoise_cl(static_cast<cl_mem>(srcPtr),
                srcSize,
                static_cast<cl_mem>(dstPtr), 
                mean, sigma,
                RPPI_CHN_PACKED, 3,
                static_cast<cl_command_queue>(rppHandle));
 	} 
#elif defined (HIP_COMPILE) 
 	{ 
 	} 
#endif //BACKEND 
	return RPP_SUCCESS;
}




// // /******* Color Temperature ********/
// RppStatus
// rppi_occlusionAdd_u8_pln1_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize1, RppiSize srcSize2, RppPtr_t dstPtr, 
//                                Rpp32u src1x1, Rpp32u src1y1, Rpp32u src1x2, Rpp32u src1y2, 
//                                Rpp32u src2x1, Rpp32u src2y1, Rpp32u src2x2, Rpp32u src2y2, RppHandle_t rppHandle){

// #ifdef OCL_COMPILE
//  	{
//         occlusion_cl( static_cast<cl_mem>(srcPtr1), static_cast<cl_mem>(srcPtr2),
//                   srcSize1,  srcSize2, static_cast<cl_mem>(dstPtr), 
//                  RPPI_CHN_PLANAR,src1x1, src1y1,
//                 src1x2, src1y2, src2x1, src2y1, src2x2, src2y2,
//                  1,
//                  static_cast<cl_command_queue>(rppHandle));
       
//  	} 
// #elif defined (HIP_COMPILE) 
//  	{ 
//  	} 
// #endif //BACKEND 
// }

// RppStatus
// rppi_occlusionAdd_u8_pln3_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize1, RppiSize srcSize2, RppPtr_t dstPtr, 
//                                Rpp32u src1x1, Rpp32u src1y1, Rpp32u src1x2, Rpp32u src1y2, 
//                                Rpp32u src2x1, Rpp32u src2y1, Rpp32u src2x2, Rpp32u src2y2, RppHandle_t rppHandle){
// #ifdef OCL_COMPILE
//  	{
//         occlusion_cl( static_cast<cl_mem>(srcPtr1), static_cast<cl_mem>(srcPtr2),
//                   srcSize1,  srcSize2, static_cast<cl_mem>(dstPtr),  
//                  RPPI_CHN_PLANAR,src1x1, src1y1,
//                 src1x2, src1y2, src2x1, src2y1, src2x2, src2y2,
//                  3,
//                  static_cast<cl_command_queue>(rppHandle));
//  	} 
// #elif defined (HIP_COMPILE) 
//  	{ 
//  	} 
// #endif //BACKEND 
// }

// RppStatus
// rppi_occlusionAdd_u8_pkd3_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize1, RppiSize srcSize2, RppPtr_t dstPtr, 
//                                Rpp32u src1x1, Rpp32u src1y1, Rpp32u src1x2, Rpp32u src1y2, 
//                                Rpp32u src2x1, Rpp32u src2y1, Rpp32u src2x2, Rpp32u src2y2, RppHandle_t rppHandle){
// #ifdef OCL_COMPILE
//  	{
//        occlusion_cl( static_cast<cl_mem>(srcPtr1), static_cast<cl_mem>(srcPtr2),
//                   srcSize1,  srcSize2, static_cast<cl_mem>(dstPtr), 
//                  RPPI_CHN_PACKED,src1x1, src1y1,
//                 src1x2, src1y2, src2x1, src2y1, src2x2, src2y2,
//                  3,
//                  static_cast<cl_command_queue>(rppHandle));
//  	} 
// #elif defined (HIP_COMPILE) 
//  	{ 
//  	} 
// #endif //BACKEND 
// 	return RPP_SUCCESS;
// }

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
	return RPP_SUCCESS;
}

}*/

RppStatus
rppi_fog_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32f fogValue,RppHandle_t rppHandle)
{
 	Rpp32f stdDev=fogValue*50;
    validate_float_min(0, stdDev);
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
    fog_cl(static_cast<cl_mem>(dstPtr), 
			srcSize, 
			fogValue,
			RPPI_CHN_PLANAR, 1, 
            static_cast<cl_command_queue>(rppHandle) );
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
 	Rpp32f stdDev=fogValue*50;
    validate_float_min(0, stdDev);
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
    fog_cl(static_cast<cl_mem>(dstPtr), 
			srcSize, 
			fogValue,
			RPPI_CHN_PLANAR, 3, 
            static_cast<cl_command_queue>(rppHandle) );
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
 	Rpp32f stdDev=fogValue*50;
    validate_float_min(0, stdDev);
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
    fog_cl(static_cast<cl_mem>(dstPtr), 
			srcSize, 
			fogValue,
			RPPI_CHN_PACKED, 3, 
            static_cast<cl_command_queue>(rppHandle) );
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}