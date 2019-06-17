#include <rppdefs.h>
#include <rppi_arithmetic_and_logical_functions.h>

#include "cpu/host_filter_functions.hpp"

#ifdef HIP_COMPILE
#include <hip/rpp_hip_common.hpp>

#elif defined(OCL_COMPILE)
#include <cl/rpp_cl_common.hpp>
#include "cl/cl_declarations.hpp"
#endif //backend
#include <stdio.h>
#include <iostream>




/******* Bilateral Filter ********/

// GPU calls for Bilateral Filter function

RppStatus
rppi_bilateral_filter_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize,
                                  RppPtr_t dstPtr, Rpp32u filterSize,
                                  Rpp64f sigmaI, Rpp64f sigmaS,
                                  RppHandle_t rppHandle)
{
#ifdef OCL_COMPILE

    bilateral_filter_cl(static_cast<cl_mem>(srcPtr), srcSize,
                        static_cast<cl_mem>(dstPtr),
                        filterSize, sigmaI, sigmaS,
                        RPPI_CHN_PLANAR, 1 /*Channel*/,
                        static_cast<cl_command_queue>(rppHandle) );


#endif //backend

    return RPP_SUCCESS;

}

RppStatus
rppi_bilateral_filter_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize,
                                  RppPtr_t dstPtr, Rpp32u filterSize,
                                  Rpp64f sigmaI, Rpp64f sigmaS,
                                  RppHandle_t rppHandle)
{
#ifdef OCL_COMPILE

    bilateral_filter_cl(static_cast<cl_mem>(srcPtr), srcSize,
                        static_cast<cl_mem>(dstPtr),
                        filterSize, sigmaI, sigmaS,
                        RPPI_CHN_PLANAR, 3 /*Channel*/,
                        static_cast<cl_command_queue>(rppHandle) );

#endif //backend

    return RPP_SUCCESS;

}

RppStatus
rppi_bilateral_filter_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize,
                                  RppPtr_t dstPtr, Rpp32u filterSize,
                                  Rpp64f sigmaI, Rpp64f sigmaS,
                                  RppHandle_t rppHandle)
{
#ifdef OCL_COMPILE

    bilateral_filter_cl(static_cast<cl_mem>(srcPtr), srcSize,
                        static_cast<cl_mem>(dstPtr),
                        filterSize, sigmaI, sigmaS,
                        RPPI_CHN_PACKED, 3 /*Channel*/,
                        static_cast<cl_command_queue>(rppHandle) );

#endif //backend

    return RPP_SUCCESS;

}

// Host calls for Bilateral Filter function

RppStatus
rppi_bilateral_filter_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                             Rpp32u diameter, Rpp64f sigmaI, Rpp64f sigmaS)
{
    bilateral_filter_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                                    diameter, sigmaI, sigmaS,
                                    RPPI_CHN_PLANAR, 1);

    return RPP_SUCCESS;

}

RppStatus
rppi_bilateral_filter_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                             Rpp32u diameter, Rpp64f sigmaI, Rpp64f sigmaS)
{
    bilateral_filter_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                                    diameter, sigmaI, sigmaS,
                                    RPPI_CHN_PLANAR, 3);

    return RPP_SUCCESS;

}

RppStatus
rppi_bilateral_filter_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                             Rpp32u diameter, Rpp64f sigmaI, Rpp64f sigmaS)
{
    bilateral_filter_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                                    diameter, sigmaI, sigmaS,
                                    RPPI_CHN_PACKED, 3);

    return RPP_SUCCESS;

}

// Box Filter Related Functions


RppStatus
rppi_box_filter_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppHandle_t rppHandle)
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
rppi_box_filter_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppHandle_t rppHandle)
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
rppi_box_filter_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppHandle_t rppHandle)
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
