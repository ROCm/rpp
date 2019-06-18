#include <rppdefs.h>
#include <rppi_geometric_functions.h>

#include "cpu/host_geometry_transforms.hpp"

#ifdef HIP_COMPILE
#include <hip/rpp_hip_common.hpp>

#elif defined(OCL_COMPILE)
#include <cl/rpp_cl_common.hpp>
#include "cl/cl_declarations.hpp"
#endif //backend




/******* Flip ********/

// GPU calls for Flip function

RppStatus
rppi_flip_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                    RppiAxis flipAxis, RppHandle_t rppHandle)
{

#ifdef OCL_COMPILE

    flip_cl(static_cast<cl_mem>(srcPtr), srcSize,
            static_cast<cl_mem>(dstPtr),
            flipAxis,
            RPPI_CHN_PLANAR, 1 /*Channel*/,
            static_cast<cl_command_queue>(rppHandle) );

#endif //backend

    return RPP_SUCCESS;
}

RppStatus
rppi_flip_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                    RppiAxis flipAxis, RppHandle_t rppHandle)
{

#ifdef OCL_COMPILE

    flip_cl(static_cast<cl_mem>(srcPtr), srcSize,
            static_cast<cl_mem>(dstPtr),
            flipAxis,
            RPPI_CHN_PLANAR, 3 /*Channel*/,
            static_cast<cl_command_queue>(rppHandle) );

#endif //backend

    return RPP_SUCCESS;

}

RppStatus
rppi_flip_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                    RppiAxis flipAxis, RppHandle_t rppHandle)
{

#ifdef OCL_COMPILE

    flip_cl(static_cast<cl_mem>(srcPtr), srcSize,
            static_cast<cl_mem>(dstPtr),
            flipAxis,
            RPPI_CHN_PACKED, 3 /*Channel*/,
            static_cast<cl_command_queue>(rppHandle) );

#endif //backend

    return RPP_SUCCESS;
}

// Host calls for Flip function

RppStatus 
rppi_flip_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                       RppiAxis flipAxis)
{
    flip_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     flipAxis,
                     RPPI_CHN_PLANAR, 1);
    return RPP_SUCCESS;
}

RppStatus
rppi_flip_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                       RppiAxis flipAxis)
{
    flip_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     flipAxis,
                     RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_flip_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                       RppiAxis flipAxis)
{
    flip_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     flipAxis,
                     RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}




/******* Warp Affine ********/

// GPU calls for Warp Affine function

// Host calls for Warp Affine function

RppStatus
rppi_warp_affine_output_size_host(RppiSize srcSize, RppiSize *dstSizePtr,
                                  RppPtr_t affine)
{
    warp_affine_output_size_host<Rpp32f>(srcSize, dstSizePtr,
                                         static_cast<Rpp32f*>(affine));

    return RPP_SUCCESS;

}

RppStatus
rppi_warp_affine_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                              RppPtr_t affine)
{
    warp_affine_host<Rpp8u, Rpp32f>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), dstSize,
                            static_cast<Rpp32f*>(affine),
                            RPPI_CHN_PLANAR, 1);

    return RPP_SUCCESS;

}

RppStatus
rppi_warp_affine_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                              RppPtr_t affine)
{
    warp_affine_host<Rpp8u, Rpp32f>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), dstSize,
                            static_cast<Rpp32f*>(affine),
                            RPPI_CHN_PLANAR, 3);

    return RPP_SUCCESS;

}

RppStatus
rppi_warp_affine_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                              RppPtr_t affine)
{
    warp_affine_host<Rpp8u, Rpp32f>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), dstSize,
                            static_cast<Rpp32f*>(affine),
                            RPPI_CHN_PACKED, 3);

    return RPP_SUCCESS;

}





/******* Rotate ********/

// GPU calls for Rotate function

// Host calls for Rotate function

RppStatus
rppi_rotate_output_size_host(RppiSize srcSize, RppiSize *dstSizePtr,
                             Rpp32f angleDeg)
{
    rotate_output_size_host(srcSize, dstSizePtr,
                            angleDeg);

    return RPP_SUCCESS;

}

RppStatus
rppi_rotate_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f angleDeg)
{
    rotate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), dstSize,
                            angleDeg,
                            RPPI_CHN_PLANAR, 1);

    return RPP_SUCCESS;

}

RppStatus
rppi_rotate_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f angleDeg)
{
    rotate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), dstSize,
                            angleDeg,
                            RPPI_CHN_PLANAR, 3);

    return RPP_SUCCESS;

}

RppStatus
rppi_rotate_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f angleDeg)
{
    rotate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), dstSize,
                            angleDeg,
                            RPPI_CHN_PACKED, 3);

    return RPP_SUCCESS;

}




/******* Scale ********/

// GPU calls for Scale function

// Host calls for Scale function

RppStatus
rppi_scale_output_size_host(RppiSize srcSize, RppiSize *dstSizePtr,
                             Rpp32f percentage)
{
    scale_output_size_host(srcSize, dstSizePtr,
                            percentage);

    return RPP_SUCCESS;

}

RppStatus
rppi_scale_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f percentage)
{
    scale_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), dstSize,
                            percentage,
                            RPPI_CHN_PLANAR, 1);

    return RPP_SUCCESS;

}

RppStatus
rppi_scale_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f percentage)
{
    scale_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), dstSize,
                            percentage,
                            RPPI_CHN_PLANAR, 3);

    return RPP_SUCCESS;

}

RppStatus
rppi_scale_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f percentage)
{
    scale_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), dstSize,
                            percentage,
                            RPPI_CHN_PACKED, 3);

    return RPP_SUCCESS;

}




/******* Resize ********/

// GPU calls for Resize function

// Host calls for Resize function

RppStatus
rppi_resize_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize)
{
    resize_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), dstSize,
                            RPPI_CHN_PLANAR, 1);

    return RPP_SUCCESS;

}

RppStatus
rppi_resize_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize)
{
    resize_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), dstSize,
                            RPPI_CHN_PLANAR, 3);

    return RPP_SUCCESS;

}

RppStatus
rppi_resize_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize)
{
    resize_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), dstSize,
                            RPPI_CHN_PACKED, 3);

    return RPP_SUCCESS;

}




/******* Resize Crop ********/

// GPU calls for Resize Crop function

// Host calls for Resize Crop function

RppStatus
rppi_resizeCrop_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2)
{
    resizeCrop_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), dstSize,
                            x1, y1, x2, y2,
                            RPPI_CHN_PLANAR, 1);

    return RPP_SUCCESS;

}

RppStatus
rppi_resizeCrop_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2)
{
    resizeCrop_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), dstSize,
                            x1, y1, x2, y2,
                            RPPI_CHN_PLANAR, 3);

    return RPP_SUCCESS;

}

RppStatus
rppi_resizeCrop_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2)
{
    resizeCrop_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), dstSize,
                            x1, y1, x2, y2,
                            RPPI_CHN_PACKED, 3);

    return RPP_SUCCESS;

}
