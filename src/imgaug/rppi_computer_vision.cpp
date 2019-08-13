#include <rppi_computer_vision.h>
#include <rppdefs.h>
#include "rppi_validate.hpp"

#ifdef HIP_COMPILE
#include <hip/rpp_hip_common.hpp>

#elif defined(OCL_COMPILE)
#include <cl/rpp_cl_common.hpp>
#include "cl/cl_declarations.hpp"
#endif //backend
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include "cpu/host_computer_vision.hpp" 
using namespace std::chrono; 
 
RppStatus
rppi_local_binary_pattern_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppHandle_t rppHandle)
{
#ifdef OCL_COMPILE
    {
        local_binary_pattern_cl(static_cast<cl_mem>(srcPtr), 
                            srcSize,
                            static_cast<cl_mem>(dstPtr),
                            RPPI_CHN_PLANAR, 1,
                            static_cast<cl_command_queue>(rppHandle));
 	} 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_local_binary_pattern_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppHandle_t rppHandle)
{
#ifdef OCL_COMPILE
    {
        local_binary_pattern_cl(static_cast<cl_mem>(srcPtr), 
                            srcSize,
                            static_cast<cl_mem>(dstPtr),
                            RPPI_CHN_PLANAR, 3,
                            static_cast<cl_command_queue>(rppHandle));
 	} 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_local_binary_pattern_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppHandle_t rppHandle)
{
#ifdef OCL_COMPILE
    {
        local_binary_pattern_cl(static_cast<cl_mem>(srcPtr), 
                            srcSize,
                            static_cast<cl_mem>(dstPtr),
                            RPPI_CHN_PACKED, 3,
                            static_cast<cl_command_queue>(rppHandle));
 	} 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_data_object_copy_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle)
{
 	 validate_image_size(srcSize);
#ifdef OCL_COMPILE
 	 {
 	 data_object_copy_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr),
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}
RppStatus
rppi_data_object_copy_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle)
{
 	 validate_image_size(srcSize);
#ifdef OCL_COMPILE
 	 {
 	 data_object_copy_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr),
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_data_object_copy_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle)
{
 	 validate_image_size(srcSize);
#ifdef OCL_COMPILE
 	 {
 	 data_object_copy_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr),
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}


RppStatus
rppi_gaussian_image_pyramid_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppHandle_t rppHandle)
{
#ifdef OCL_COMPILE
    {
        gaussian_image_pyramid_cl(static_cast<cl_mem>(srcPtr), 
                            srcSize,
                            static_cast<cl_mem>(dstPtr),
                            stdDev, kernelSize,
                            RPPI_CHN_PLANAR, 1,
                            static_cast<cl_command_queue>(rppHandle));
 	} 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_gaussian_image_pyramid_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppHandle_t rppHandle)
{
#ifdef OCL_COMPILE
    {
        gaussian_image_pyramid_cl(static_cast<cl_mem>(srcPtr), 
                            srcSize,
                            static_cast<cl_mem>(dstPtr),
                            stdDev, kernelSize,
                            RPPI_CHN_PLANAR, 3,
                            static_cast<cl_command_queue>(rppHandle));
 	} 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_gaussian_image_pyramid_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppHandle_t rppHandle)
{
#ifdef OCL_COMPILE
    {
        gaussian_image_pyramid_cl(static_cast<cl_mem>(srcPtr), 
                            srcSize,
                            static_cast<cl_mem>(dstPtr),
                            stdDev, kernelSize,
                            RPPI_CHN_PACKED, 3,
                            static_cast<cl_command_queue>(rppHandle));
 	} 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

// ----------------------------------------
// GPU laplacian_image_pyramid functions declaration 
// ----------------------------------------

RppStatus
rppi_laplacian_image_pyramid_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppHandle_t rppHandle)
{
#ifdef OCL_COMPILE
    {
        laplacian_image_pyramid_cl(static_cast<cl_mem>(srcPtr), 
                            srcSize,
                            static_cast<cl_mem>(dstPtr),
                            stdDev, kernelSize,
                            RPPI_CHN_PLANAR, 1,
                            static_cast<cl_command_queue>(rppHandle));
 	} 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_laplacian_image_pyramid_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppHandle_t rppHandle)
{
#ifdef OCL_COMPILE
    {
        laplacian_image_pyramid_cl(static_cast<cl_mem>(srcPtr), 
                            srcSize,
                            static_cast<cl_mem>(dstPtr),
                            stdDev, kernelSize,
                            RPPI_CHN_PLANAR, 3,
                            static_cast<cl_command_queue>(rppHandle));
 	} 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_laplacian_image_pyramid_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppHandle_t rppHandle)
{
#ifdef OCL_COMPILE
    {
        laplacian_image_pyramid_cl(static_cast<cl_mem>(srcPtr), 
                            srcSize,
                            static_cast<cl_mem>(dstPtr),
                            stdDev, kernelSize,
                            RPPI_CHN_PACKED, 3,
                            static_cast<cl_command_queue>(rppHandle));
 	} 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

 
// ----------------------------------------
// Host local_binary_pattern functions calls 
// ----------------------------------------


RppStatus
rppi_local_binary_pattern_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{
    local_binary_pattern_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     RPPI_CHN_PLANAR, 1);
    return RPP_SUCCESS;
}

RppStatus
rppi_local_binary_pattern_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{
    local_binary_pattern_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_local_binary_pattern_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{
    local_binary_pattern_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}

// ----------------------------------------
// Host data_object_copy functions calls 
// ----------------------------------------


RppStatus
rppi_data_object_copy_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{
    data_object_copy_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     RPPI_CHN_PLANAR, 1);
    return RPP_SUCCESS;
}

RppStatus
rppi_data_object_copy_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{
    data_object_copy_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_data_object_copy_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{
    data_object_copy_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}

// ----------------------------------------
// Host gaussian_image_pyramid functions calls 
// ----------------------------------------


RppStatus
rppi_gaussian_image_pyramid_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp32f stdDev, Rpp32u kernelSize)
{
    gaussian_image_pyramid_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     stdDev, kernelSize,
                     RPPI_CHN_PLANAR, 1);
    return RPP_SUCCESS;
}

RppStatus
rppi_gaussian_image_pyramid_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp32f stdDev, Rpp32u kernelSize)
{
    gaussian_image_pyramid_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     stdDev, kernelSize,
                     RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_gaussian_image_pyramid_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp32f stdDev, Rpp32u kernelSize)
{
    gaussian_image_pyramid_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     stdDev, kernelSize,
                     RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}

// ----------------------------------------
// Host laplacian_image_pyramid functions calls 
// ----------------------------------------


RppStatus
rppi_laplacian_image_pyramid_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp32f stdDev, Rpp32u kernelSize)
{
    laplacian_image_pyramid_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     stdDev, kernelSize,
                     RPPI_CHN_PLANAR, 1);
    return RPP_SUCCESS;
}

RppStatus
rppi_laplacian_image_pyramid_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp32f stdDev, Rpp32u kernelSize)
{
    laplacian_image_pyramid_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     stdDev, kernelSize,
                     RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_laplacian_image_pyramid_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp32f stdDev, Rpp32u kernelSize)
{
    laplacian_image_pyramid_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     stdDev, kernelSize,
                     RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}