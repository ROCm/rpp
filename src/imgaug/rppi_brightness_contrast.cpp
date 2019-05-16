#include <rppdefs.h>
#include <rppi_image_augumentation_functions.h>
#include "cpu/host_brightness_contrast.hpp"

#ifdef HIP_COMPILE
#include <hip/rpp_hip_common.hpp>
#include "hip/hip_brightness_contrast.hpp"
#elif defined(OCL_COMPILE)
#include <cl/rpp_cl_common.hpp>
#include "cl/cl_declarations.hpp"
#endif //backend

RppStatus
rppi_brighten_1C8U_pln( RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr,
                        Rpp32f alpha, Rpp32s beta,
                        RppHandle_t rppHandle )
{

#ifdef HIP_COMPILE
    hip_brightness_contrast<Rpp8u>( static_cast<Rpp8u*>(srcPtr), srcSize,
                                    static_cast<Rpp8u*>(dstPtr),
                                    alpha, beta,
                                    RPPI_CHN_PLANAR,
                                    (hipStream_t)rppHandle);

#elif defined (OCL_COMPILE)

    cl_brightness_contrast (    static_cast<cl_mem>(srcPtr), srcSize,
                                static_cast<cl_mem>(dstPtr),
                                alpha, beta,
                                RPPI_CHN_PLANAR, 1 /*Channel*/,
                                static_cast<cl_command_queue>(rppHandle) );


#endif //backend

    return RPP_SUCCESS;
}


RppStatus
rppi_brighten_3C8U_pln( RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr,
                        Rpp32f alpha, Rpp32s beta,
                        RppHandle_t rppHandle )
{


#ifdef OCL_COMPILE

    cl_brightness_contrast (    static_cast<cl_mem>(srcPtr), srcSize,
                                static_cast<cl_mem>(dstPtr),
                                alpha, beta,
                                RPPI_CHN_PLANAR, 3 /*Channel*/,
                                static_cast<cl_command_queue>(rppHandle) );


#endif //backend

    return RPP_SUCCESS;
}



RppStatus
rppi_brighten_3C8U_pkd( RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr,
                        Rpp32f alpha, Rpp32s beta,
                        RppHandle_t rppHandle )
{


#ifdef OCL_COMPILE

    cl_brightness_contrast (    static_cast<cl_mem>(srcPtr), srcSize,
                                static_cast<cl_mem>(dstPtr),
                                alpha, beta,
                                RPPI_CHN_PACKED, 3 /*Channel*/,
                                static_cast<cl_command_queue>(rppHandle) );


#endif //backend

    return RPP_SUCCESS;
}



RppStatus
rppi_brighten_1C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                            Rpp32f alpha, Rpp32s beta, RppHandle_t rppHandle)
{
    host_brightness_contrast<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                                    static_cast<Rpp8u*>(dstPtr), alpha, beta, 1, RPPI_CHN_PLANAR );

    return RPP_SUCCESS;

}

RppStatus
rppi_brighten_3C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, Rpp32s beta)
{
    host_brightness_contrast<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                                    static_cast<Rpp8u*>(dstPtr), alpha, beta, 3, RPPI_CHN_PLANAR );

    return RPP_SUCCESS;

}
