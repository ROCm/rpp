#include <rppdefs.h>
#include <rppi_image_augumentation_functions.h>

#include "cpu/host_contrast.hpp"

#include <iostream>
#ifdef HIP_COMPILE
#include <hip/rpp_hip_common.hpp>
#include "hip/hip_brightness_contrast.hpp"
#elif defined(OCL_COMPILE)
#include <cl/rpp_cl_common.hpp>
#include "cl/cl_declarations.hpp"
#endif //backend


RppStatus
rppi_contrast_1C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                            Rpp32u new_min, Rpp32u new_max)
{

    host_contrast<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                                    static_cast<Rpp8u*>(dstPtr), new_min ,new_max );
    return RPP_SUCCESS;

}

RppStatus
rppi_contrast_1C8U_pln(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                            Rpp32u new_min, Rpp32u new_max, RppHandle_t rppHandle)
{

    #ifdef HIP_COMPILE
   /*Still needs to be implemented*/

    #elif defined (OCL_COMPILE)

    cl_contrast_streach (   static_cast<cl_mem>(srcPtr), srcSize,
                            static_cast<cl_mem>(dstPtr),
                            new_min, new_max,
                            RPPI_CHN_PLANAR, 1 /*Channel*/,
                            static_cast<cl_command_queue>(rppHandle));


    #endif //backend

    return RPP_SUCCESS;

}

