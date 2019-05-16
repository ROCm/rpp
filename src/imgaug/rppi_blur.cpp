#include <rppdefs.h>
#include <rppi_image_augumentation_functions.h>
#include "cpu/host_blur.hpp"

#ifdef HIP_COMPILE
#include <hip/rpp_hip_common.hpp>

#elif defined(OCL_COMPILE)
#include <cl/rpp_cl_common.hpp>
#include "cl/cl_declarations.hpp"
#endif //backend
#include <stdio.h>
#include <iostream>

RppStatus
rppi_blur3x3_1C8U_pln(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppHandle_t rppHandle)
{
#ifdef OCL_COMPILE

    cl_gaussian_blur(static_cast<cl_mem>(srcPtr), srcSize,
                        static_cast<cl_mem>(dstPtr),
                        3 /*Filter width*/,
                        RPPI_CHN_PLANAR, 1 /*Channel*/,
                        static_cast<cl_command_queue>(rppHandle) );


#endif //backend

    return RPP_SUCCESS;

}

RppStatus
rppi_blur3x3_3C8U_pln(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppHandle_t rppHandle)
{
#ifdef OCL_COMPILE

    cl_gaussian_blur(static_cast<cl_mem>(srcPtr), srcSize,
                        static_cast<cl_mem>(dstPtr),
                        3 /*Filter width*/,
                        RPPI_CHN_PLANAR, 3 /*Channel*/,
                        static_cast<cl_command_queue>(rppHandle) );

#endif //backend

    return RPP_SUCCESS;

}

RppStatus
rppi_blur3x3_1C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{

    host_blur<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                                    static_cast<Rpp8u*>(dstPtr), 1);
    return RPP_SUCCESS;

}

RppStatus
rppi_blur3x3_3C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{

    host_blur<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                                    static_cast<Rpp8u*>(dstPtr), 3);
    return RPP_SUCCESS;

}
