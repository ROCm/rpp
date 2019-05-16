#include <rppdefs.h>
#include <rppi_image_augumentation_functions.h>
#include "cpu/host_flip.hpp"

#ifdef HIP_COMPILE
#include <hip/rpp_hip_common.hpp>

#elif defined(OCL_COMPILE)
#include <cl/rpp_cl_common.hpp>
#include "cl/cl_declarations.hpp"
#endif //backend

RppStatus
rppi_flip_1C8U_pln(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                    RppiAxis flipAxis, RppHandle_t rppHandle)
{

#ifdef OCL_COMPILE

    cl_flip(static_cast<cl_mem>(srcPtr), srcSize,
            static_cast<cl_mem>(dstPtr),
            flipAxis,
            RPPI_CHN_PLANAR, 1 /*Channel*/,
            static_cast<cl_command_queue>(rppHandle) );

#endif //backend

    return RPP_SUCCESS;
}

RppStatus
rppi_flip_3C8U_pln(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                    RppiAxis flipAxis, RppHandle_t rppHandle)
{

#ifdef OCL_COMPILE

    cl_flip(static_cast<cl_mem>(srcPtr), srcSize,
            static_cast<cl_mem>(dstPtr),
            flipAxis,
            RPPI_CHN_PLANAR, 3 /*Channel*/,
            static_cast<cl_command_queue>(rppHandle) );

#endif //backend

    return RPP_SUCCESS;

}

RppStatus
rppi_flip_u8_pkd3(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                    RppiAxis flipAxis, RppHandle_t rppHandle)
{

#ifdef OCL_COMPILE

    cl_flip(static_cast<cl_mem>(srcPtr), srcSize,
            static_cast<cl_mem>(dstPtr),
            flipAxis,
            RPPI_CHN_PACKED, 3 /*Channel*/,
            static_cast<cl_command_queue>(rppHandle) );

#endif //backend

    return RPP_SUCCESS;
}

 // host function call for single channel input
RppStatus rppi_flip_1C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                                   RppiAxis flipAxis)
{
    host_flip<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                     static_cast<Rpp8u*>(dstPtr),
                     flipAxis, 1);
    return RPP_SUCCESS;
}

RppStatus rppi_flip_3C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                                   RppiAxis flipAxis)
{
    host_flip<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                     static_cast<Rpp8u*>(dstPtr),
                     flipAxis, 3);
    return RPP_SUCCESS;
}
