#include <rppdefs.h>
#include <rppi_image_augumentation_functions.h>

#include "cpu/host_hsv_functions.hpp"

#include <iostream>
#ifdef HIP_COMPILE
#include <hip/rpp_hip_common.hpp>

#elif defined(OCL_COMPILE)
#include <cl/rpp_cl_common.hpp>
#include "cl/cl_declarations.hpp"
#endif //backend

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



///////////////HUE RELATED//////////////////////////
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

///////////////
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


RppStatus
rppi_saturationRGB_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f saturationFactor)
{

    saturationRGB_pln_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                            static_cast<Rpp8u*>(dstPtr), saturationFactor);
    return RPP_SUCCESS;

}

RppStatus
rppi_saturationRGB_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f saturationFactor)
{

    saturationRGB_pkd_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                            static_cast<Rpp8u*>(dstPtr), saturationFactor);
    return RPP_SUCCESS;

}

RppStatus
rppi_saturationHSV_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f saturationFactor)
{

    saturationHSV_pln_host<Rpp32f>(static_cast<Rpp32f*>(srcPtr), srcSize,
                            static_cast<Rpp32f*>(dstPtr), saturationFactor);
    return RPP_SUCCESS;

}

RppStatus
rppi_saturationHSV_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f saturationFactor)
{

    saturationHSV_pkd_host<Rpp32f>(static_cast<Rpp32f*>(srcPtr), srcSize,
                            static_cast<Rpp32f*>(dstPtr), saturationFactor);
    return RPP_SUCCESS;

}

RppStatus
rppi_rgb2hsv_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{

    rgb2hsv_pln_host<Rpp32f>(static_cast<Rpp32f*>(srcPtr), srcSize,
                                    static_cast<Rpp32f*>(dstPtr));
    return RPP_SUCCESS;

}

RppStatus
rppi_rgb2hsv_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{

    rgb2hsv_pkd_host<Rpp32f>(static_cast<Rpp32f*>(srcPtr), srcSize,
                                    static_cast<Rpp32f*>(dstPtr));
    return RPP_SUCCESS;

}

RppStatus
rppi_hueRGB_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f hueShift)
{

    hueRGB_pln_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                            static_cast<Rpp8u*>(dstPtr), hueShift);
    return RPP_SUCCESS;

}

RppStatus
rppi_hueRGB_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f hueShift)
{

    hueRGB_pkd_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                            static_cast<Rpp8u*>(dstPtr), hueShift);
    return RPP_SUCCESS;

}

RppStatus
rppi_hueHSV_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f hueShift)
{

    hueHSV_pln_host<Rpp32f>(static_cast<Rpp32f*>(srcPtr), srcSize,
                            static_cast<Rpp32f*>(dstPtr), hueShift);
    return RPP_SUCCESS;

}

RppStatus
rppi_hueHSV_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f hueShift)
{

    hueHSV_pkd_host<Rpp32f>(static_cast<Rpp32f*>(srcPtr), srcSize,
                            static_cast<Rpp32f*>(dstPtr), hueShift);
    return RPP_SUCCESS;

}

RppStatus
rppi_hsv2rgb_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{

    hsv2rgb_pln_host<Rpp32f>(static_cast<Rpp32f*>(srcPtr), srcSize,
                                    static_cast<Rpp32f*>(dstPtr));
    return RPP_SUCCESS;

}

RppStatus
rppi_hsv2rgb_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{

    hsv2rgb_pkd_host<Rpp32f>(static_cast<Rpp32f*>(srcPtr), srcSize,
                                    static_cast<Rpp32f*>(dstPtr));
    return RPP_SUCCESS;

}
