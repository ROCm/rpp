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
// Host color_temperature functions calls 
// ----------------------------------------


RppStatus
rppi_color_temperature_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp8s adjustmentValue)
{
    color_temperature_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     adjustmentValue,
                     RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_color_temperature_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp8s adjustmentValue)
{
    color_temperature_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     adjustmentValue,
                     RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host pixelate functions calls 
// ----------------------------------------


RppStatus
rppi_pixelate_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             Rpp32u kernelSize, unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2)
{
    pixelate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     kernelSize, x1, y1, x2, y2, 
                     RPPI_CHN_PLANAR, 1);
    return RPP_SUCCESS;
}

RppStatus
rppi_pixelate_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             Rpp32u kernelSize, unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2)
{
    pixelate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     kernelSize, x1, y1, x2, y2, 
                     RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_pixelate_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             Rpp32u kernelSize, unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2)
{
    pixelate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     kernelSize, x1, y1, x2, y2, 
                     RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host jitterAdd functions calls 
// ----------------------------------------


RppStatus
rppi_jitterAdd_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             unsigned int maxJitterX, unsigned int maxJitterY)
{
    jitterAdd_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     maxJitterX, maxJitterY, 
                     RPPI_CHN_PLANAR, 1);
    return RPP_SUCCESS;
}

RppStatus
rppi_jitterAdd_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             unsigned int maxJitterX, unsigned int maxJitterY)
{
    jitterAdd_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     maxJitterX, maxJitterY, 
                     RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_jitterAdd_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             unsigned int maxJitterX, unsigned int maxJitterY)
{
    jitterAdd_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     maxJitterX, maxJitterY, 
                     RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host vignette functions calls 
// ----------------------------------------


RppStatus
rppi_vignette_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp32f stdDev)
{
    vignette_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     stdDev,
                     RPPI_CHN_PLANAR, 1);
    return RPP_SUCCESS;
}

RppStatus
rppi_vignette_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp32f stdDev)
{
    vignette_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     stdDev,
                     RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_vignette_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp32f stdDev)
{
    vignette_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     stdDev,
                     RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host fish_eye_effect functions calls 
// ----------------------------------------


RppStatus
rppi_fish_eye_effect_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{
    fish_eye_effect_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                                RPPI_CHN_PLANAR, 1);
    return RPP_SUCCESS;
}

RppStatus
rppi_fish_eye_effect_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{
    fish_eye_effect_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                                RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_fish_eye_effect_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{
    fish_eye_effect_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                                RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host lens_correction functions calls 
// ----------------------------------------


RppStatus
rppi_lens_correction_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom)
{
    lens_correction_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                                strength, zoom, 
                                RPPI_CHN_PLANAR, 1);
    return RPP_SUCCESS;
}

RppStatus
rppi_lens_correction_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom)
{
    lens_correction_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                                strength, zoom, 
                                RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_lens_correction_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom)
{
    lens_correction_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                                strength, zoom, 
                                RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host occlusionAdd functions calls 
// ----------------------------------------


RppStatus
rppi_occlusionAdd_u8_pln1_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize1, RppiSize srcSize2, RppPtr_t dstPtr, 
                               Rpp32u src1x1, Rpp32u src1y1, Rpp32u src1x2, Rpp32u src1y2, 
                               Rpp32u src2x1, Rpp32u src2y1, Rpp32u src2x2, Rpp32u src2y2)
{
    occlusionAdd_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize1, srcSize2, static_cast<Rpp8u*>(dstPtr), 
                             src1x1, src1y1, src1x2, src1y2, src2x1, src2y1, src2x2, src2y2, 
                             RPPI_CHN_PLANAR, 1);

    return RPP_SUCCESS;

}

RppStatus
rppi_occlusionAdd_u8_pln3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize1, RppiSize srcSize2, RppPtr_t dstPtr, 
                               Rpp32u src1x1, Rpp32u src1y1, Rpp32u src1x2, Rpp32u src1y2, 
                               Rpp32u src2x1, Rpp32u src2y1, Rpp32u src2x2, Rpp32u src2y2)
{
    occlusionAdd_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize1, srcSize2, static_cast<Rpp8u*>(dstPtr), 
                             src1x1, src1y1, src1x2, src1y2, src2x1, src2y1, src2x2, src2y2, 
                             RPPI_CHN_PLANAR, 3);

    return RPP_SUCCESS;

}

RppStatus
rppi_occlusionAdd_u8_pkd3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize1, RppiSize srcSize2, RppPtr_t dstPtr, 
                               Rpp32u src1x1, Rpp32u src1y1, Rpp32u src1x2, Rpp32u src1y2, 
                               Rpp32u src2x1, Rpp32u src2y1, Rpp32u src2x2, Rpp32u src2y2)
{
    occlusionAdd_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize1, srcSize2, static_cast<Rpp8u*>(dstPtr), 
                             src1x1, src1y1, src1x2, src1y2, src2x1, src2y1, src2x2, src2y2, 
                             RPPI_CHN_PACKED, 3);

    return RPP_SUCCESS;

}
 
// ----------------------------------------
// Host snowy functions calls 
// ----------------------------------------


RppStatus
rppi_snowyRGB_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f strength)
{

    snowy_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                           strength,
                           RPPI_CHN_PLANAR, 3, RGB);
    return RPP_SUCCESS;
}

RppStatus
rppi_snowyRGB_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f strength)
{

    snowy_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                           strength,
                           RPPI_CHN_PACKED, 3, RGB);
    return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host random_shadow functions calls 
// ----------------------------------------


RppStatus
rppi_random_shadow_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                                Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, 
                                Rpp32u numberOfShadows, Rpp32u maxSizeX, Rpp32u maxSizeY)
{
    random_shadow_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                              x1, y1, x2, y2, numberOfShadows, maxSizeX, maxSizeY, 
                              RPPI_CHN_PLANAR, 1);
    return RPP_SUCCESS;
}

RppStatus
rppi_random_shadow_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                                Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, 
                                Rpp32u numberOfShadows, Rpp32u maxSizeX, Rpp32u maxSizeY)
{
    random_shadow_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                              x1, y1, x2, y2, numberOfShadows, maxSizeX, maxSizeY, 
                              RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_random_shadow_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                                Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, 
                                Rpp32u numberOfShadows, Rpp32u maxSizeX, Rpp32u maxSizeY)
{
    random_shadow_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                              x1, y1, x2, y2, numberOfShadows, maxSizeX, maxSizeY, 
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
// GPU pixelate functions  calls 
// ----------------------------------------
RppStatus
rppi_pixelate_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_range( 0, srcSize.height, x1);
 	 validate_int_range( 0, srcSize.width, y1);
 	 validate_int_range( 0, srcSize.height, x2);
 	 validate_int_range( 0, srcSize.width, y2);

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
rppi_pixelate_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_range( 0, srcSize.height, x1);
 	 validate_int_range( 0, srcSize.width, y1);
 	 validate_int_range( 0, srcSize.height, x2);
 	 validate_int_range( 0, srcSize.width, y2);

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
rppi_pixelate_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_range( 0, srcSize.height, x1);
 	 validate_int_range( 0, srcSize.width, y1);
 	 validate_int_range( 0, srcSize.height, x2);
 	 validate_int_range( 0, srcSize.width, y2);

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
// GPU jitter functions  calls 
// ----------------------------------------
RppStatus
rppi_jitter_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u minJitter, Rpp32u maxJitter, RppHandle_t rppHandle)
{
   	 validate_image_size(srcSize);

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
// GPU snow functions  calls 
// ----------------------------------------
RppStatus
rppi_snow_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f snowCoefficient, RppHandle_t rppHandle)
{
   	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 snow_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			snowCoefficient,
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
rppi_snow_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f snowCoefficient, RppHandle_t rppHandle)
{
   	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 snow_cl(static_cast<cl_mem>(srcPtr),
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			snowCoefficient,
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
rppi_snow_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f snowCoefficient, RppHandle_t rppHandle)
{
   	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 snow_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr),
			snowCoefficient,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}
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
	return RPP_SUCCESS;
}


/******* Pixelate ********/
// GPU calls for Pixelate function
// Host calls for Pixelate function
RppStatus
rppi_pixelate_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
							Rpp32u kernelSize, unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2)
{
    pixelate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     kernelSize, x1, y1, x2, y2, 
                     RPPI_CHN_PLANAR, 1);
    return RPP_SUCCESS;
}

RppStatus
rppi_pixelate_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             Rpp32u kernelSize, unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2)
{
    pixelate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     kernelSize, x1, y1, x2, y2, 
                     RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_pixelate_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             Rpp32u kernelSize, unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2)
{
    pixelate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     kernelSize, x1, y1, x2, y2, 
                     RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}




/******* Jitter Add ********/

// GPU calls for JitterAdd function

// Host calls for JitterAdd function

RppStatus
rppi_jitterAdd_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             unsigned int maxJitterX, unsigned int maxJitterY)
{
    jitterAdd_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     maxJitterX, maxJitterY, 
                     RPPI_CHN_PLANAR, 1);
    return RPP_SUCCESS;
}

RppStatus
rppi_jitterAdd_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             unsigned int maxJitterX, unsigned int maxJitterY)
{
    jitterAdd_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     maxJitterX, maxJitterY, 
                     RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_jitterAdd_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             unsigned int maxJitterX, unsigned int maxJitterY)
{
    jitterAdd_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     maxJitterX, maxJitterY, 
                     RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}




/******* Vignette ********/

// GPU calls for Vignette function

// Host calls for Vignette function

RppStatus
rppi_vignette_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp32f stdDev)
{
    vignette_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     stdDev,
                     RPPI_CHN_PLANAR, 1);
    return RPP_SUCCESS;
}

RppStatus
rppi_vignette_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp32f stdDev)
{
    vignette_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     stdDev,
                     RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_vignette_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp32f stdDev)
{
    vignette_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     stdDev,
                     RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}




/******* Color Temperature ********/

// GPU calls for Color Temperature function

// Host calls for Color Temperature function

RppStatus
rppi_color_temperature_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp8s adjustmentValue)
{
    color_temperature_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     adjustmentValue,
                     RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_color_temperature_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp8s adjustmentValue)
{
    color_temperature_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     adjustmentValue,
                     RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}




/******* Fish Eye Effect ********/

// GPU calls for Fish Eye Effect function

// Host calls for Fish Eye Effect function

RppStatus
rppi_fish_eye_effect_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{
    fish_eye_effect_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                                RPPI_CHN_PLANAR, 1);
    return RPP_SUCCESS;
}

RppStatus
rppi_fish_eye_effect_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{
    fish_eye_effect_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                                RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_fish_eye_effect_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{
    fish_eye_effect_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                                RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}




/******* Lens Correction ********/

// GPU calls for Lens Correction function

// Host calls for Lens Correction function

RppStatus
rppi_lens_correction_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom)
{
    lens_correction_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                                strength, zoom, 
                                RPPI_CHN_PLANAR, 1);
    return RPP_SUCCESS;
}

RppStatus
rppi_lens_correction_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom)
{
    lens_correction_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                                strength, zoom, 
                                RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_lens_correction_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom)
{
    lens_correction_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                                strength, zoom, 
                                RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}




/******* Occlusion Add ********/

// GPU calls for Occlusion Add function

// Host calls for Occlusion Add function

RppStatus
rppi_occlusionAdd_u8_pln1_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize1, RppiSize srcSize2, RppPtr_t dstPtr, 
                               Rpp32u src1x1, Rpp32u src1y1, Rpp32u src1x2, Rpp32u src1y2, 
                               Rpp32u src2x1, Rpp32u src2y1, Rpp32u src2x2, Rpp32u src2y2)
{
    occlusionAdd_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize1, srcSize2, static_cast<Rpp8u*>(dstPtr), 
                             src1x1, src1y1, src1x2, src1y2, src2x1, src2y1, src2x2, src2y2, 
                             RPPI_CHN_PLANAR, 1);

    return RPP_SUCCESS;

}

RppStatus
rppi_occlusionAdd_u8_pln3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize1, RppiSize srcSize2, RppPtr_t dstPtr, 
                               Rpp32u src1x1, Rpp32u src1y1, Rpp32u src1x2, Rpp32u src1y2, 
                               Rpp32u src2x1, Rpp32u src2y1, Rpp32u src2x2, Rpp32u src2y2)
{
    occlusionAdd_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize1, srcSize2, static_cast<Rpp8u*>(dstPtr), 
                             src1x1, src1y1, src1x2, src1y2, src2x1, src2y1, src2x2, src2y2, 
                             RPPI_CHN_PLANAR, 3);

    return RPP_SUCCESS;

}

RppStatus
rppi_occlusionAdd_u8_pkd3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize1, RppiSize srcSize2, RppPtr_t dstPtr, 
                               Rpp32u src1x1, Rpp32u src1y1, Rpp32u src1x2, Rpp32u src1y2, 
                               Rpp32u src2x1, Rpp32u src2y1, Rpp32u src2x2, Rpp32u src2y2)
{
    occlusionAdd_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize1, srcSize2, static_cast<Rpp8u*>(dstPtr), 
                             src1x1, src1y1, src1x2, src1y2, src2x1, src2y1, src2x2, src2y2, 
                             RPPI_CHN_PACKED, 3);

    return RPP_SUCCESS;

}




/******* Snowy ********/

// GPU calls for Snowy function

// Host calls for Snowy function

RppStatus
rppi_snowyRGB_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f strength)
{

    snowy_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                           strength,
                           RPPI_CHN_PLANAR, 3, RGB);
    return RPP_SUCCESS;
}

RppStatus
rppi_snowyRGB_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f strength)
{

    snowy_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                           strength,
                           RPPI_CHN_PACKED, 3, RGB);
    return RPP_SUCCESS;
}




/******* Random Shadow ********/

// GPU calls for Random Shadow function

// Host calls for Random Shadow function

RppStatus
rppi_random_shadow_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                                Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, 
                                Rpp32u numberOfShadows, Rpp32u maxSizeX, Rpp32u maxSizeY)
{
    random_shadow_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                              x1, y1, x2, y2, numberOfShadows, maxSizeX, maxSizeY, 
                              RPPI_CHN_PLANAR, 1);
    return RPP_SUCCESS;
}

RppStatus
rppi_random_shadow_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                                Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, 
                                Rpp32u numberOfShadows, Rpp32u maxSizeX, Rpp32u maxSizeY)
{
    random_shadow_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                              x1, y1, x2, y2, numberOfShadows, maxSizeX, maxSizeY, 
                              RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_random_shadow_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                                Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, 
                                Rpp32u numberOfShadows, Rpp32u maxSizeX, Rpp32u maxSizeY)
{
    random_shadow_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                              x1, y1, x2, y2, numberOfShadows, maxSizeX, maxSizeY, 
                              RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}
// ----------------------------------------
// Random Shadow functions  calls 
// ----------------------------------------

RppStatus
rppi_random_shadow_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, Rpp32u numberOfShadows, Rpp32u maxSizeX, Rpp32u maxSizeY, RppHandle_t rppHandle)
{
   	validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	{
        random_shadow_cl(static_cast<cl_mem>(srcPtr),
            srcSize,
            static_cast<cl_mem>(dstPtr), 
            x1, y1, x2, y2, numberOfShadows, maxSizeX, maxSizeY,
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
rppi_random_shadow_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, Rpp32u numberOfShadows, Rpp32u maxSizeX, Rpp32u maxSizeY, RppHandle_t rppHandle)
{
   	validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	{
        random_shadow_cl(static_cast<cl_mem>(srcPtr),
            srcSize,
            static_cast<cl_mem>(dstPtr), 
            x1, y1, x2, y2, numberOfShadows, maxSizeX, maxSizeY,
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
rppi_random_shadow_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, Rpp32u numberOfShadows, Rpp32u maxSizeX, Rpp32u maxSizeY, RppHandle_t rppHandle)
{
   	validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	{
        random_shadow_cl(static_cast<cl_mem>(srcPtr),
            srcSize,
            static_cast<cl_mem>(dstPtr), 
            x1, y1, x2, y2, numberOfShadows, maxSizeX, maxSizeY,
            RPPI_CHN_PACKED, 3,
            static_cast<cl_command_queue>(rppHandle));
 	} 
#elif defined (HIP_COMPILE) 
 	{ 
 	} 
#endif //BACKEND 
	return RPP_SUCCESS;
}