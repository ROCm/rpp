#include <rppdefs.h>
#include <rppi_image_augumentation_functions.h>

//#include "cpu/host_rgb2hsv.hpp"

#include <iostream>
#ifdef HIP_COMPILE
#include <hip/rpp_hip_common.hpp>
#include "hip/hip_brightness_contrast.hpp"
#elif defined(OCL_COMPILE)
#include <cl/rpp_cl_common.hpp>
#include "cl/cl_declarations.hpp"
#endif //backend

/*RppStatus
rppi_rgb2hsv_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{

    host_rgb2hsv<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                                    static_cast<Rpp8u*>(dstPtr));
    return RPP_SUCCESS;

}*/

RppStatus
rppi_rgb2hsv_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,  RppHandle_t rppHandle)
{
    #ifdef HIP_COMPILE
   /*Still needs to be implemented*/

    #elif defined (OCL_COMPILE)

    cl_convert_rgb2hsv(   static_cast<cl_mem>(srcPtr), srcSize,
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

    cl_convert_rgb2hsv(   static_cast<cl_mem>(srcPtr), srcSize,
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

    cl_convert_hsv2rgb(   static_cast<cl_mem>(srcPtr), srcSize,
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

    cl_convert_hsv2rgb(   static_cast<cl_mem>(srcPtr), srcSize,
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
    cl_hue_saturation_rgb (    static_cast<cl_mem>(srcPtr), srcSize,
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
    cl_hue_saturation_rgb (    static_cast<cl_mem>(srcPtr), srcSize,
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
    cl_hue_saturation_rgb (    static_cast<cl_mem>(srcPtr), srcSize,
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
    cl_hue_saturation_rgb (    static_cast<cl_mem>(srcPtr), srcSize,
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
    cl_hue_saturation_hsv (    static_cast<cl_mem>(srcPtr), srcSize,
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
    cl_hue_saturation_hsv (    static_cast<cl_mem>(srcPtr), srcSize,
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
    cl_hue_saturation_hsv (    static_cast<cl_mem>(srcPtr), srcSize,
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
    cl_hue_saturation_hsv (    static_cast<cl_mem>(srcPtr), srcSize,
                            static_cast<cl_mem>(dstPtr), 0.0/*hue*/, saturationFactor,
                            RPPI_CHN_PACKED, 3 /*Channel*/,
                            static_cast<cl_command_queue>(rppHandle));
    
    #endif //backend

    return RPP_SUCCESS;
    
}


RppStatus
rppi_saturationRGB_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f saturationFactor)
{

    host_saturationRGB_pln<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                            static_cast<Rpp8u*>(dstPtr), saturationFactor);
    return RPP_SUCCESS;

}

RppStatus
rppi_saturationRGB_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f saturationFactor)
{

    host_saturationRGB_pkd<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                            static_cast<Rpp8u*>(dstPtr), saturationFactor);
    return RPP_SUCCESS;

}

RppStatus
rppi_saturationHSV_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f saturationFactor)
{

    host_saturationHSV_pln<Rpp32f>(static_cast<Rpp32f*>(srcPtr), srcSize,
                            static_cast<Rpp32f*>(dstPtr), saturationFactor);
    return RPP_SUCCESS;

}

RppStatus
rppi_saturationHSV_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f saturationFactor)
{

    host_saturationHSV_pkd<Rpp32f>(static_cast<Rpp32f*>(srcPtr), srcSize,
                            static_cast<Rpp32f*>(dstPtr), saturationFactor);
    return RPP_SUCCESS;

}

RppStatus
rppi_rgb2hsv_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{

    host_rgb2hsv_pln<Rpp32f>(static_cast<Rpp32f*>(srcPtr), srcSize,
                                    static_cast<Rpp32f*>(dstPtr));
    return RPP_SUCCESS;

}

RppStatus
rppi_rgb2hsv_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{

    host_rgb2hsv_pkd<Rpp32f>(static_cast<Rpp32f*>(srcPtr), srcSize,
                                    static_cast<Rpp32f*>(dstPtr));
    return RPP_SUCCESS;

}

RppStatus
rppi_hueRGB_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f hueShift)
{

    host_hueRGB_pln<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                            static_cast<Rpp8u*>(dstPtr), hueShift);
    return RPP_SUCCESS;

}

RppStatus
rppi_hueRGB_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f hueShift)
{

    host_hueRGB_pkd<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                            static_cast<Rpp8u*>(dstPtr), hueShift);
    return RPP_SUCCESS;

}

RppStatus
rppi_hueHSV_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f hueShift)
{

    host_hueHSV_pln<Rpp32f>(static_cast<Rpp32f*>(srcPtr), srcSize,
                            static_cast<Rpp32f*>(dstPtr), hueShift);
    return RPP_SUCCESS;

}

RppStatus
rppi_hueHSV_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f hueShift)
{

    host_hueHSV_pkd<Rpp32f>(static_cast<Rpp32f*>(srcPtr), srcSize,
                            static_cast<Rpp32f*>(dstPtr), hueShift);
    return RPP_SUCCESS;

}

RppStatus
rppi_hsv2rgb_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{

    host_hsv2rgb_pln<Rpp32f>(static_cast<Rpp32f*>(srcPtr), srcSize,
                                    static_cast<Rpp32f*>(dstPtr));
    return RPP_SUCCESS;

}

RppStatus
rppi_hsv2rgb_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{

    host_hsv2rgb_pkd<Rpp32f>(static_cast<Rpp32f*>(srcPtr), srcSize,
                                    static_cast<Rpp32f*>(dstPtr));
    return RPP_SUCCESS;

}
