#include <rppdefs.h>
#include <rppi_image_augumentation_functions.h>

#include "cpu/host_color_model_conversions.hpp"

#include <iostream>
#ifdef HIP_COMPILE
#include <hip/rpp_hip_common.hpp>

#elif defined(OCL_COMPILE)
#include <cl/rpp_cl_common.hpp>
#include "cl/cl_declarations.hpp"
#endif //backend




/******* RGB 2 HSV ********/

// GPU calls for RGB 2 HSV function

RppStatus
rppi_rgb2hsv_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,  RppHandle_t rppHandle)
{
    #ifdef HIP_COMPILE
   /*Still needs to be implemented*/

    #elif defined (OCL_COMPILE)

    convert_rgb2hsv_cl(   static_cast<cl_mem>(srcPtr), srcSize,
                            static_cast<cl_mem>(dstPtr),
                            RPPI_CHN_PLANAR, 3 /*Channel*/,
                            static_cast<cl_command_queue>(rppHandle));

    #endif //backend

    return RPP_SUCCESS;

}

RppStatus
rppi_rgb2hsv_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,  RppHandle_t rppHandle)
{
    #ifdef HIP_COMPILE
   /*Still needs to be implemented*/

    #elif defined (OCL_COMPILE)

    convert_rgb2hsv_cl(   static_cast<cl_mem>(srcPtr), srcSize,
                            static_cast<cl_mem>(dstPtr),
                            RPPI_CHN_PACKED, 3 /*Channel*/,
                            static_cast<cl_command_queue>(rppHandle));

    #endif //backend

    return RPP_SUCCESS;

}

// Host calls for RGB 2 HSV function

RppStatus
rppi_rgb2hsv_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{

    rgb2hsv_host<Rpp8u, Rpp32f>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp32f*>(dstPtr),
                         RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;

}

RppStatus
rppi_rgb2hsv_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{

    rgb2hsv_host<Rpp8u, Rpp32f>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp32f*>(dstPtr),
                         RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;

}




/******* HSV 2 RGB ********/

// GPU calls for HSV 2 RGB function

RppStatus
rppi_hsv2rgb_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,  RppHandle_t rppHandle)
{
    #ifdef HIP_COMPILE
   /*Still needs to be implemented*/

    #elif defined (OCL_COMPILE)

    convert_hsv2rgb_cl(   static_cast<cl_mem>(srcPtr), srcSize,
                            static_cast<cl_mem>(dstPtr),
                            RPPI_CHN_PLANAR, 3 /*Channel*/,
                            static_cast<cl_command_queue>(rppHandle));

    #endif //backend

    return RPP_SUCCESS;

}

RppStatus
rppi_hsv2rgb_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,  RppHandle_t rppHandle)
{
    #ifdef HIP_COMPILE
   /*Still needs to be implemented*/

    #elif defined (OCL_COMPILE)

    convert_hsv2rgb_cl(   static_cast<cl_mem>(srcPtr), srcSize,
                            static_cast<cl_mem>(dstPtr),
                            RPPI_CHN_PACKED, 3 /*Channel*/,
                            static_cast<cl_command_queue>(rppHandle));

    #endif //backend

    return RPP_SUCCESS;

}

// Host calls for HSV 2 RGB function

RppStatus
rppi_hsv2rgb_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{

    hsv2rgb_host<Rpp32f, Rpp8u>(static_cast<Rpp32f*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                         RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;

}

RppStatus
rppi_hsv2rgb_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{

    hsv2rgb_host<Rpp32f, Rpp8u>(static_cast<Rpp32f*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                         RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;

}




/******* HUE ********/

// GPU calls for HUE function

RppStatus
rppi_hueRGB_u8_pln3_gpu (RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                        Rpp32f hueShift,  RppHandle_t rppHandle){
    #ifdef HIP_COMPILE
    /*Still needs to be implemented*/

    #elif defined (OCL_COMPILE)
    hue_saturation_rgb_cl (    static_cast<cl_mem>(srcPtr), srcSize,
                            static_cast<cl_mem>(dstPtr), hueShift, 0.0/*Saturation*/,
                            RPPI_CHN_PLANAR, 3 /*Channel*/,
                            static_cast<cl_command_queue>(rppHandle));

    #endif //backend

    return RPP_SUCCESS;

}

RppStatus
rppi_hueRGB_u8_pkd3_gpu (RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                        Rpp32f hueShift,  RppHandle_t rppHandle){
    #ifdef HIP_COMPILE
    /*Still needs to be implemented*/

    #elif defined (OCL_COMPILE)
    hue_saturation_rgb_cl (    static_cast<cl_mem>(srcPtr), srcSize,
                            static_cast<cl_mem>(dstPtr), hueShift, 0.0/*Saturation*/,
                            RPPI_CHN_PACKED, 3 /*Channel*/,
                            static_cast<cl_command_queue>(rppHandle));

    #endif //backend

    return RPP_SUCCESS;

}

RppStatus
rppi_hueHSV_u8_pln3_gpu (RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                        Rpp32f hueShift,  RppHandle_t rppHandle){
    #ifdef HIP_COMPILE
    /*Still needs to be implemented*/

    #elif defined (OCL_COMPILE)
    hue_saturation_hsv_cl (    static_cast<cl_mem>(srcPtr), srcSize,
                            static_cast<cl_mem>(dstPtr), hueShift, 0.0/*Saturation*/,
                            RPPI_CHN_PLANAR, 3 /*Channel*/,
                            static_cast<cl_command_queue>(rppHandle));

    #endif //backend

    return RPP_SUCCESS;

}

RppStatus
rppi_hueHSV_u8_pkd3_gpu (RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                        Rpp32f hueShift,  RppHandle_t rppHandle){
    #ifdef HIP_COMPILE
    /*Still needs to be implemented*/

    #elif defined (OCL_COMPILE)
    hue_saturation_hsv_cl (    static_cast<cl_mem>(srcPtr), srcSize,
                            static_cast<cl_mem>(dstPtr), hueShift, 0.0/*Saturation*/,
                            RPPI_CHN_PACKED, 3 /*Channel*/,
                            static_cast<cl_command_queue>(rppHandle));

    #endif //backend

    return RPP_SUCCESS;

}

// Host calls for HUE function

RppStatus
rppi_hueRGB_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f hueShift)
{

    hue_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                           hueShift,
                           RPPI_CHN_PLANAR, 3, RGB);
    return RPP_SUCCESS;
}

RppStatus
rppi_hueRGB_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f hueShift)
{

    hue_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                           hueShift,
                           RPPI_CHN_PACKED, 3, RGB);
    return RPP_SUCCESS;
}

RppStatus
rppi_hueHSV_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f hueShift)
{

    hue_host<Rpp32f>(static_cast<Rpp32f*>(srcPtr), srcSize, static_cast<Rpp32f*>(dstPtr),
                           hueShift,
                           RPPI_CHN_PLANAR, 3, HSV);
    return RPP_SUCCESS;
}

RppStatus
rppi_hueHSV_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f hueShift)
{

    hue_host<Rpp32f>(static_cast<Rpp32f*>(srcPtr), srcSize, static_cast<Rpp32f*>(dstPtr),
                           hueShift,
                           RPPI_CHN_PACKED, 3, HSV);
    return RPP_SUCCESS;
}




/******* Saturation ********/

// GPU calls for Saturation function

RppStatus
rppi_saturationRGB_u8_pln3_gpu (RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                        Rpp32f saturationFactor,  RppHandle_t rppHandle){
    #ifdef HIP_COMPILE
    /*Still needs to be implemented*/

    #elif defined (OCL_COMPILE)
    hue_saturation_rgb_cl (    static_cast<cl_mem>(srcPtr), srcSize,
                            static_cast<cl_mem>(dstPtr), 0.0/*hue*/, saturationFactor,
                            RPPI_CHN_PLANAR, 3 /*Channel*/,
                            static_cast<cl_command_queue>(rppHandle));

    #endif //backend

    return RPP_SUCCESS;

}

RppStatus
rppi_saturationRGB_u8_pkd3_gpu (RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                        Rpp32f saturationFactor,  RppHandle_t rppHandle){
    #ifdef HIP_COMPILE
    /*Still needs to be implemented*/

    #elif defined (OCL_COMPILE)
    hue_saturation_rgb_cl (    static_cast<cl_mem>(srcPtr), srcSize,
                            static_cast<cl_mem>(dstPtr), 0.0/*hue*/, saturationFactor,
                            RPPI_CHN_PACKED, 3 /*Channel*/,
                            static_cast<cl_command_queue>(rppHandle));

    #endif //backend

    return RPP_SUCCESS;

}

RppStatus
rppi_saturationHSV_u8_pln3_gpu (RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                        Rpp32f saturationFactor,  RppHandle_t rppHandle){
    #ifdef HIP_COMPILE
    /*Still needs to be implemented*/

    #elif defined (OCL_COMPILE)
    hue_saturation_hsv_cl (    static_cast<cl_mem>(srcPtr), srcSize,
                            static_cast<cl_mem>(dstPtr), 0.0/*hue*/, saturationFactor,
                            RPPI_CHN_PLANAR, 3 /*Channel*/,
                            static_cast<cl_command_queue>(rppHandle));

    #endif //backend

    return RPP_SUCCESS;

}

RppStatus
rppi_saturationHSV_u8_pkd3_gpu (RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                        Rpp32f saturationFactor,  RppHandle_t rppHandle){
    #ifdef HIP_COMPILE
    /*Still needs to be implemented*/

    #elif defined (OCL_COMPILE)
    hue_saturation_hsv_cl (    static_cast<cl_mem>(srcPtr), srcSize,
                            static_cast<cl_mem>(dstPtr), 0.0/*hue*/, saturationFactor,
                            RPPI_CHN_PACKED, 3 /*Channel*/,
                            static_cast<cl_command_queue>(rppHandle));

    #endif //backend

    return RPP_SUCCESS;

}

// Host calls for Saturation function

RppStatus
rppi_saturationRGB_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f saturationFactor)
{

    saturation_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                           saturationFactor,
                           RPPI_CHN_PLANAR, 3, RGB);
    return RPP_SUCCESS;
}

RppStatus
rppi_saturationRGB_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f saturationFactor)
{

    saturation_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                           saturationFactor,
                           RPPI_CHN_PACKED, 3, RGB);
    return RPP_SUCCESS;
}

RppStatus
rppi_saturationHSV_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f saturationFactor)
{

    saturation_host<Rpp32f>(static_cast<Rpp32f*>(srcPtr), srcSize, static_cast<Rpp32f*>(dstPtr),
                           saturationFactor,
                           RPPI_CHN_PLANAR, 3, HSV);
    return RPP_SUCCESS;
}

RppStatus
rppi_saturationHSV_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f saturationFactor)
{

    saturation_host<Rpp32f>(static_cast<Rpp32f*>(srcPtr), srcSize, static_cast<Rpp32f*>(dstPtr),
                           saturationFactor,
                           RPPI_CHN_PACKED, 3, HSV);
    return RPP_SUCCESS;
}




/******* Gamma Correction ********/

// GPU calls for Gamma Correction function

// Host calls for Gamma Correction function

RppStatus
rppi_gamma_correction_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                             Rpp32f gamma)
{
    gamma_correction_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                                    gamma,
                                    1);

    return RPP_SUCCESS;

}

RppStatus
rppi_gamma_correction_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                             Rpp32f gamma)
{
    gamma_correction_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                                    gamma,
                                    3);

    return RPP_SUCCESS;

}

RppStatus
rppi_gamma_correction_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                             Rpp32f gamma)
{
    gamma_correction_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                                    gamma,
                                    3);

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