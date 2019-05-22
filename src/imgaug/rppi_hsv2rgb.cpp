#include <rppdefs.h>
#include <rppi_image_augumentation_functions.h>

#include "cpu/host_hsv2rgb.hpp"

#include <iostream>

#ifdef HIP_COMPILE
#include <hip/rpp_hip_common.hpp>
#include "hip/hip_brightness_contrast.hpp"
#elif defined(OCL_COMPILE)
#include <cl/rpp_cl_common.hpp>
#include "cl/cl_declarations.hpp"
#endif //backend


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