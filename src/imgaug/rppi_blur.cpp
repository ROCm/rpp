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
rppi_blur3x3_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppHandle_t rppHandle)
{
#ifdef OCL_COMPILE

    gaussian_blur_cl(static_cast<cl_mem>(srcPtr), srcSize,
                        static_cast<cl_mem>(dstPtr),
                        3 /*Filter width*/,
                        RPPI_CHN_PLANAR, 1 /*Channel*/,
                        static_cast<cl_command_queue>(rppHandle) );


#endif //backend

    return RPP_SUCCESS;

}

RppStatus
rppi_blur3x3_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppHandle_t rppHandle)
{
#ifdef OCL_COMPILE

    gaussian_blur_cl(static_cast<cl_mem>(srcPtr), srcSize,
                        static_cast<cl_mem>(dstPtr),
                        3 /*Filter width*/,
                        RPPI_CHN_PLANAR, 3 /*Channel*/,
                        static_cast<cl_command_queue>(rppHandle) );

#endif //backend

    return RPP_SUCCESS;

}


RppStatus
rppi_blur3x3_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppHandle_t rppHandle)
{
#ifdef OCL_COMPILE

    gaussian_blur_cl(static_cast<cl_mem>(srcPtr), srcSize,
                        static_cast<cl_mem>(dstPtr),
                        3 /*Filter width*/,
                        RPPI_CHN_PACKED, 3 /*Channel*/,
                        static_cast<cl_command_queue>(rppHandle) );

#endif //backend

    return RPP_SUCCESS;

}

RppStatus
rppi_blur3x3_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{

    blur_pln_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                                    static_cast<Rpp8u*>(dstPtr), 1);
    return RPP_SUCCESS;

}

RppStatus
rppi_blur3x3_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{

    blur_pln_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                                    static_cast<Rpp8u*>(dstPtr), 3);
    return RPP_SUCCESS;

}

RppStatus
rppi_blur3x3_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{

    blur_pkd_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                                    static_cast<Rpp8u*>(dstPtr), 3);
    return RPP_SUCCESS;

}
