#include <rppdefs.h>
#include <rppi_image_augumentation_functions.h>

#include "cpu/host_hue.hpp"

#include <iostream>

#ifdef HIP_COMPILE
#include <hip/rpp_hip_common.hpp>
#include "hip/hip_brightness_contrast.hpp"
#elif defined(OCL_COMPILE)
#include <cl/rpp_cl_common.hpp>
#include "cl/cl_declarations.hpp"
#endif //backend

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

// GPU functions 

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