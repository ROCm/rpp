#include <rppi_morphological_transforms.h>
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
using namespace std::chrono; 

RppStatus
rppi_dilate_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize, RppHandle_t rppHandle)
{
#ifdef OCL_COMPILE
    {
        dilate_cl(static_cast<cl_mem>(srcPtr), 
                            srcSize,
                            static_cast<cl_mem>(dstPtr), 
                            kernelSize,
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
rppi_dilate_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize, RppHandle_t rppHandle)
{
#ifdef OCL_COMPILE
    {
        dilate_cl(static_cast<cl_mem>(srcPtr), 
                            srcSize,
                            static_cast<cl_mem>(dstPtr), 
                            kernelSize,
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
rppi_dilate_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize, RppHandle_t rppHandle)
{
#ifdef OCL_COMPILE
    {
        dilate_cl(static_cast<cl_mem>(srcPtr), 
                            srcSize,
                            static_cast<cl_mem>(dstPtr), 
                            kernelSize,
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
rppi_erode_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize, RppHandle_t rppHandle)
{
#ifdef OCL_COMPILE
    {
        erode_cl(static_cast<cl_mem>(srcPtr), 
                            srcSize,
                            static_cast<cl_mem>(dstPtr), 
                            kernelSize,
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
rppi_erode_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize, RppHandle_t rppHandle)
{
#ifdef OCL_COMPILE
    {
        erode_cl(static_cast<cl_mem>(srcPtr), 
                            srcSize,
                            static_cast<cl_mem>(dstPtr), 
                            kernelSize,
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
rppi_erode_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize, RppHandle_t rppHandle)
{
#ifdef OCL_COMPILE
    {
        erode_cl(static_cast<cl_mem>(srcPtr), 
                            srcSize,
                            static_cast<cl_mem>(dstPtr), 
                            kernelSize,
                            RPPI_CHN_PACKED, 3,
                            static_cast<cl_command_queue>(rppHandle));
 	} 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}
