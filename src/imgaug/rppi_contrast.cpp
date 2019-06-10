#include <rppdefs.h>
#include <rppi_image_augumentation_functions.h>
#include "cpu/host_contrast.hpp"

#ifdef HIP_COMPILE
#include <hip/rpp_hip_common.hpp>

#elif defined(OCL_COMPILE)
#include <cl/rpp_cl_common.hpp>
#include "cl/cl_declarations.hpp"
#endif //backend

RppStatus
rppi_contrast_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                            Rpp32u newMin, Rpp32u newMax)
{
    contrast_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                                    static_cast<Rpp8u*>(dstPtr), newMin , newMax, 1 );
    return RPP_SUCCESS;

}

RppStatus
rppi_contrast_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize,RppPtr_t dstPtr,
                            Rpp32u newMin, Rpp32u newMax)
{
    contrast_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                                    static_cast<Rpp8u*>(dstPtr), newMin ,newMax, 3 );
    return RPP_SUCCESS;
}

RppStatus
rppi_contrast_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize,RppPtr_t dstPtr,
                            Rpp32u newMin, Rpp32u newMax)
{
    contrast_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                                    static_cast<Rpp8u*>(dstPtr), newMin ,newMax, 3 );
    return RPP_SUCCESS;
}

RppStatus
rppi_contrast_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                            Rpp32u newMin, Rpp32u newMax, RppHandle_t rppHandle)
{

    #ifdef HIP_COMPILE
   /*Still needs to be implemented*/

    #elif defined (OCL_COMPILE)

    contrast_stretch_cl (   static_cast<cl_mem>(srcPtr), srcSize,
                            static_cast<cl_mem>(dstPtr),
                            newMin, newMax,
                            RPPI_CHN_PLANAR, 1 /*Channel*/,
                            static_cast<cl_command_queue>(rppHandle));


    #endif //backend

    return RPP_SUCCESS;

}

RppStatus
rppi_contrast_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                            Rpp32u newMin, Rpp32u newMax, RppHandle_t rppHandle)
{

    #ifdef HIP_COMPILE
   /*Still needs to be implemented*/

    #elif defined (OCL_COMPILE)

    contrast_stretch_cl (   static_cast<cl_mem>(srcPtr), srcSize,
                            static_cast<cl_mem>(dstPtr),
                            newMin, newMax,
                            RPPI_CHN_PLANAR, 3 /*Channel*/,
                            static_cast<cl_command_queue>(rppHandle));


    #endif //backend

    return RPP_SUCCESS;

}



RppStatus
rppi_contrast_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                            Rpp32u newMin, Rpp32u newMax, RppHandle_t rppHandle)
{

    #ifdef HIP_COMPILE
   /*Still needs to be implemented*/

    #elif defined (OCL_COMPILE)

    contrast_stretch_cl (   static_cast<cl_mem>(srcPtr), srcSize,
                            static_cast<cl_mem>(dstPtr),
                            newMin, newMax,
                            RPPI_CHN_PACKED, 3 /*Channel*/,
                            static_cast<cl_command_queue>(rppHandle));


    #endif //backend

    return RPP_SUCCESS;

}
