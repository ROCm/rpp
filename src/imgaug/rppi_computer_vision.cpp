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
rppi_channel_extract_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32u extractChannelNumber, RppHandle_t rppHandle)
{

    /*call that offset function */
    #ifdef OCL_COMPILE

    channel_extract_cl(static_cast<cl_mem>(srcPtr), srcSize,
            static_cast<cl_mem>(dstPtr), extractChannelNumber, RPPI_CHN_PLANAR, 1 /* Channel */,
            static_cast<cl_command_queue>(rppHandle) );

    return RPP_SUCCESS;
    #endif

}

RppStatus
rppi_channel_extract_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32u extractChannelNumber, RppHandle_t rppHandle)
{

    /*call that offset function */
    #ifdef OCL_COMPILE

    channel_extract_cl(static_cast<cl_mem>(srcPtr), srcSize,
            static_cast<cl_mem>(dstPtr), extractChannelNumber, RPPI_CHN_PLANAR, 3 /* Channel */,
            static_cast<cl_command_queue>(rppHandle) );

    return RPP_SUCCESS;
    #endif

}

RppStatus
rppi_channel_extract_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32u extractChannelNumber, RppHandle_t rppHandle)
{

    /*call that offset function */
    #ifdef OCL_COMPILE

    channel_extract_cl(static_cast<cl_mem>(srcPtr), srcSize,
            static_cast<cl_mem>(dstPtr), extractChannelNumber, RPPI_CHN_PACKED, 3 /* Channel */,
            static_cast<cl_command_queue>(rppHandle) );

    return RPP_SUCCESS;
    #endif

}

RppStatus
rppi_channel_combine_u8_pln1_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t srcPtr3, RppiSize srcSize, RppPtr_t dstPtr,
                         RppHandle_t rppHandle)
{

    /*call that offset function */
    #ifdef OCL_COMPILE

    channel_combine_cl(static_cast<cl_mem>(srcPtr1), static_cast<cl_mem>(srcPtr2), static_cast<cl_mem>(srcPtr3), srcSize,
            static_cast<cl_mem>(dstPtr), RPPI_CHN_PLANAR, 1 /* Channel */,
            static_cast<cl_command_queue>(rppHandle) );

    return RPP_SUCCESS;
    #endif

}

RppStatus
rppi_channel_combine_u8_pln3_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t srcPtr3, RppiSize srcSize, RppPtr_t dstPtr,
                         RppHandle_t rppHandle)
{

    /*call that offset function */
    #ifdef OCL_COMPILE

    channel_combine_cl(static_cast<cl_mem>(srcPtr1), static_cast<cl_mem>(srcPtr2), static_cast<cl_mem>(srcPtr3), srcSize,
            static_cast<cl_mem>(dstPtr), RPPI_CHN_PLANAR, 3 /* Channel */,
            static_cast<cl_command_queue>(rppHandle) );

    return RPP_SUCCESS;
    #endif

}

RppStatus
rppi_channel_combine_u8_pkd3_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t srcPtr3, RppiSize srcSize, RppPtr_t dstPtr,
                         RppHandle_t rppHandle)
{

    /*call that offset function */
    #ifdef OCL_COMPILE

    channel_combine_cl(static_cast<cl_mem>(srcPtr1), static_cast<cl_mem>(srcPtr2), static_cast<cl_mem>(srcPtr3), srcSize,
            static_cast<cl_mem>(dstPtr), RPPI_CHN_PACKED, 3 /* Channel */,
            static_cast<cl_command_queue>(rppHandle) );

    return RPP_SUCCESS;
    #endif

}
