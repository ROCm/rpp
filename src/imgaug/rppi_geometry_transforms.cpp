#include <rppdefs.h>
#include <rppi_geometric_functions.h>
#include<rppi_support_functions.h>
#include<iostream>
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



//Resize -------------------------------
//GPU----------
RppStatus
rppi_resize_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         RppHandle_t rppHandle)
{
   
    #ifdef OCL_COMPILE
        #if DEBUG
        std::cout << "Inside rppi_resize_u8_pln1_gpu function\n " <<endl;
        std::cout << dstSize.height << dstSize.width << std::endl;
        #endif //For DEBUG
    resize_cl(static_cast<cl_mem>(srcPtr), srcSize,
            static_cast<cl_mem>(dstPtr), dstSize,  RPPI_CHN_PLANAR, 1 /* Channel */,
            static_cast<cl_command_queue>(rppHandle) );

    return RPP_SUCCESS;
    #endif
}

RppStatus
rppi_resize_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         RppHandle_t rppHandle)
{
   
    #ifdef OCL_COMPILE

    resize_cl(static_cast<cl_mem>(srcPtr), srcSize,
            static_cast<cl_mem>(dstPtr), dstSize,  RPPI_CHN_PLANAR, 3 /* Channel */,
            static_cast<cl_command_queue>(rppHandle) );

    return RPP_SUCCESS;
    #endif
}

RppStatus
rppi_resize_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         RppHandle_t rppHandle)
{
   
    #ifdef OCL_COMPILE

    resize_cl(static_cast<cl_mem>(srcPtr), srcSize,
            static_cast<cl_mem>(dstPtr), dstSize, RPPI_CHN_PACKED, 3 /* Channel */,
            static_cast<cl_command_queue>(rppHandle) );

    return RPP_SUCCESS;
    #endif
}
//----------GPU
//--------------------------------Resize


//Resize-crop------------------------------------------------
//GPU ---------
RppStatus
rppi_resize_crop_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2,  RppHandle_t rppHandle)
{
   
    #ifdef OCL_COMPILE

    resize_crop_cl(static_cast<cl_mem>(srcPtr), srcSize,
            static_cast<cl_mem>(dstPtr), dstSize, x1, y1, x2, y2, RPPI_CHN_PLANAR, 1 /* Channel */,
            static_cast<cl_command_queue>(rppHandle) );

    return RPP_SUCCESS;
    #endif
}

RppStatus
rppi_resize_crop_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2,  RppHandle_t rppHandle)
{
   
    #ifdef OCL_COMPILE

    resize_crop_cl(static_cast<cl_mem>(srcPtr), srcSize,
            static_cast<cl_mem>(dstPtr), dstSize, x1, y1, x2, y2, RPPI_CHN_PLANAR, 3 /* Channel */,
            static_cast<cl_command_queue>(rppHandle) );

    return RPP_SUCCESS;
    #endif
}

RppStatus
rppi_resize_crop_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2,  RppHandle_t rppHandle)
{
   
    #ifdef OCL_COMPILE

    resize_crop_cl(static_cast<cl_mem>(srcPtr), srcSize,
            static_cast<cl_mem>(dstPtr), dstSize, x1, y1, x2, y2, RPPI_CHN_PACKED, 3 /* Channel */,
            static_cast<cl_command_queue>(rppHandle) );

    return RPP_SUCCESS; 
    #endif
}

//----------GPU
//-------------------------------------------------Resize-crop

//Rotate----------------
//GPU------
RppStatus
rppi_rotate_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f angleDeg, RppHandle_t rppHandle)
{
    
    #ifdef OCL_COMPILE

    rotate_cl(static_cast<cl_mem>(srcPtr), srcSize,
            static_cast<cl_mem>(dstPtr), dstSize, angleDeg, RPPI_CHN_PLANAR, 1 /* Channel */,
            static_cast<cl_command_queue>(rppHandle) );

    return RPP_SUCCESS;
    #endif

}

RppStatus
rppi_rotate_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f angleDeg, RppHandle_t rppHandle)
{
    
    #ifdef OCL_COMPILE

    rotate_cl(static_cast<cl_mem>(srcPtr), srcSize,
            static_cast<cl_mem>(dstPtr), dstSize, angleDeg, RPPI_CHN_PLANAR, 3 /* Channel */,
            static_cast<cl_command_queue>(rppHandle) );

    return RPP_SUCCESS;
    #endif
}

RppStatus
rppi_rotate_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f angleDeg, RppHandle_t rppHandle)
{
    
    #ifdef OCL_COMPILE

    rotate_cl(static_cast<cl_mem>(srcPtr), srcSize,
            static_cast<cl_mem>(dstPtr), dstSize, angleDeg, RPPI_CHN_PACKED, 3 /* Channel */,
            static_cast<cl_command_queue>(rppHandle) );

    return RPP_SUCCESS;
    #endif
}


//------GPU
//----------------Rotate