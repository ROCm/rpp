#include <rppdefs.h>
#include <rppi_image_augumentation_functions.h>

#include "cpu/host_image_augmentations.hpp"

#ifdef HIP_COMPILE
#include <hip/rpp_hip_common.hpp>
#include "hipoc/hipoc_declarations.hpp"
#elif defined(OCL_COMPILE)
#include <cl/rpp_cl_common.hpp>
#include "cl/cl_declarations.hpp"
#endif //backend
#include <stdio.h>
#include <iostream>




/******* Blur ********/

// GPU calls for Blur function

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

// Host calls for Blur function

RppStatus
rppi_blur_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp32f stdDev, Rpp32u kernelSize)
{
    blur_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     stdDev, kernelSize,
                     RPPI_CHN_PLANAR, 1);
    return RPP_SUCCESS;
}

RppStatus
rppi_blur_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp32f stdDev, Rpp32u kernelSize)
{
    blur_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     stdDev, kernelSize,
                     RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_blur_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp32f stdDev, Rpp32u kernelSize)
{
    blur_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     stdDev, kernelSize,
                     RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}




/******* Brightness ********/

// GPU calls for Brightness function

RppStatus
rppi_brightness_u8_pln1_gpu( RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr,
                        Rpp32f alpha, Rpp32s beta,
                        RppHandle_t rppHandle )
{

#ifdef HIP_COMPILE

    hipoc_brightness_contrast(  srcPtr, srcSize,
                                dstPtr,
                                alpha, beta,
                                RPPI_CHN_PLANAR, 1 /*Channel*/,
                                static_cast<hipStream_t>(rppHandle) );

#elif defined (OCL_COMPILE)

    brightness_contrast_cl (    static_cast<cl_mem>(srcPtr), srcSize,
                                static_cast<cl_mem>(dstPtr),
                                alpha, beta,
                                RPPI_CHN_PLANAR, 1 /*Channel*/,
                                static_cast<cl_command_queue>(rppHandle) );


#endif //backend

    return RPP_SUCCESS;
}

RppStatus
rppi_brightness_u8_pln3_gpu( RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr,
                        Rpp32f alpha, Rpp32s beta,
                        RppHandle_t rppHandle )
{

#ifdef HIP_COMPILE

    hipoc_brightness_contrast(  srcPtr, srcSize,
                                dstPtr,
                                alpha, beta,
                                RPPI_CHN_PLANAR, 3 /*Channel*/,
                                static_cast<hipStream_t>(rppHandle) );

#elif defined (OCL_COMPILE)

    brightness_contrast_cl (    static_cast<cl_mem>(srcPtr), srcSize,
                                static_cast<cl_mem>(dstPtr),
                                alpha, beta,
                                RPPI_CHN_PLANAR, 3 /*Channel*/,
                                static_cast<cl_command_queue>(rppHandle) );


#endif //backend

    return RPP_SUCCESS;
}

RppStatus
rppi_brightness_u8_pkd3_gpu( RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr,
                        Rpp32f alpha, Rpp32s beta,
                        RppHandle_t rppHandle )
{


#ifdef HIP_COMPILE

    hipoc_brightness_contrast(  srcPtr, srcSize,
                                dstPtr,
                                alpha, beta,
                                RPPI_CHN_PLANAR, 3 /*Channel*/,
                                static_cast<hipStream_t>(rppHandle) );

#elif defined (OCL_COMPILE)

    brightness_contrast_cl (    static_cast<cl_mem>(srcPtr), srcSize,
                                static_cast<cl_mem>(dstPtr),
                                alpha, beta,
                                RPPI_CHN_PACKED, 3 /*Channel*/,
                                static_cast<cl_command_queue>(rppHandle) );


#endif //backend

    return RPP_SUCCESS;
}

// Host calls for Brightness function

RppStatus
rppi_brightness_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                             Rpp32f alpha, Rpp32s beta)
{
    brightness_contrast_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                                    alpha, beta,
                                    1);

    return RPP_SUCCESS;

}

RppStatus
rppi_brightness_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                             Rpp32f alpha, Rpp32s beta)
{
    brightness_contrast_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                                    alpha, beta,
                                    3);

    return RPP_SUCCESS;

}

RppStatus
rppi_brightness_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                             Rpp32f alpha, Rpp32s beta)
{
    brightness_contrast_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                                    alpha, beta,
                                    3);

    return RPP_SUCCESS;

}




/******* Contrast ********/

// GPU calls for Contrast function

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

// Host calls for Contrast function

RppStatus
rppi_contrast_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                           Rpp32u newMin, Rpp32u newMax)
{
    contrast_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                         newMin, newMax,
                         RPPI_CHN_PLANAR, 1);
    return RPP_SUCCESS;

}

RppStatus
rppi_contrast_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize,RppPtr_t dstPtr,
                           Rpp32u newMin, Rpp32u newMax)
{
    contrast_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                         newMin, newMax,
                         RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_contrast_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize,RppPtr_t dstPtr,
                           Rpp32u newMin, Rpp32u newMax)
{
    contrast_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                         newMin, newMax,
                         RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}




/******* Pixelate ********/

// GPU calls for Pixelate function

// Host calls for Pixelate function

RppStatus
rppi_pixelate_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             Rpp32u kernelSize, unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2)
{
    pixelate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     kernelSize, x1, y1, x2, y2, 
                     RPPI_CHN_PLANAR, 1);
    return RPP_SUCCESS;
}

RppStatus
rppi_pixelate_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             Rpp32u kernelSize, unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2)
{
    pixelate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     kernelSize, x1, y1, x2, y2, 
                     RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_pixelate_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             Rpp32u kernelSize, unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2)
{
    pixelate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     kernelSize, x1, y1, x2, y2, 
                     RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}




/******* Jitter Add ********/

// GPU calls for Pixelate function

// Host calls for Pixelate function

RppStatus
rppi_jitterAdd_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             unsigned int maxJitterX, unsigned int maxJitterY)
{
    jitterAdd_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     maxJitterX, maxJitterY, 
                     RPPI_CHN_PLANAR, 1);
    return RPP_SUCCESS;
}

RppStatus
rppi_jitterAdd_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             unsigned int maxJitterX, unsigned int maxJitterY)
{
    jitterAdd_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     maxJitterX, maxJitterY, 
                     RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_jitterAdd_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             unsigned int maxJitterX, unsigned int maxJitterY)
{
    jitterAdd_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     maxJitterX, maxJitterY, 
                     RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}