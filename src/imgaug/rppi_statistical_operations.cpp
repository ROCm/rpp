#include <rppi_statistical_operations.h>
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

#include "cpu/host_statistical_operations.hpp" 
 
// ----------------------------------------
// Host min functions calls 
// ----------------------------------------


RppStatus
rppi_min_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
	 min_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), 
			static_cast<Rpp8u*>(srcPtr2), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_min_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
	 min_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), 
			static_cast<Rpp8u*>(srcPtr2), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_min_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
	 min_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), 
			static_cast<Rpp8u*>(srcPtr2), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host max functions calls 
// ----------------------------------------


RppStatus
rppi_max_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
	 max_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), 
			static_cast<Rpp8u*>(srcPtr2), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_max_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
	 max_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), 
			static_cast<Rpp8u*>(srcPtr2), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_max_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
	 max_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), 
			static_cast<Rpp8u*>(srcPtr2), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host histogram functions calls 
// ----------------------------------------


RppStatus
rppi_histogram_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,Rpp32u* outputHistogram,Rpp32u bins)
{

 	 validate_image_size(srcSize);
 	 validate_unsigned_int_range( 1, 256, &bins);
	 histogram_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			outputHistogram,
			bins,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_histogram_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,Rpp32u* outputHistogram,Rpp32u bins)
{

 	 validate_image_size(srcSize);
 	 validate_unsigned_int_range( 1, 256, &bins);
	 histogram_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			outputHistogram,
			bins,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_histogram_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,Rpp32u* outputHistogram,Rpp32u bins)
{

 	 validate_image_size(srcSize);
 	 validate_unsigned_int_range( 1, 256, &bins);
	 histogram_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			outputHistogram,
			bins,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host min_max_loc functions calls 
// ----------------------------------------


RppStatus
rppi_min_max_loc_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,Rpp8u* min,Rpp8u* max,Rpp32u* minLoc,Rpp32u* maxLoc)
{

 	 validate_image_size(srcSize);
	 min_max_loc_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			min,
			max,
			minLoc,
			maxLoc,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_min_max_loc_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,Rpp8u* min,Rpp8u* max,Rpp32u* minLoc,Rpp32u* maxLoc)
{

 	 validate_image_size(srcSize);
	 min_max_loc_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			min,
			max,
			minLoc,
			maxLoc,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_min_max_loc_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,Rpp8u* min,Rpp8u* max,Rpp32u* minLoc,Rpp32u* maxLoc)
{

 	 validate_image_size(srcSize);
	 min_max_loc_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			min,
			max,
			minLoc,
			maxLoc,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}
 
// ----------------------------------------
// GPU min functions  calls 
// ----------------------------------------


RppStatus
rppi_min_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 min_cl(static_cast<cl_mem>(srcPtr1), 
			static_cast<cl_mem>(srcPtr2), 
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
rppi_min_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 min_cl(static_cast<cl_mem>(srcPtr1), 
			static_cast<cl_mem>(srcPtr2), 
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
rppi_min_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 min_cl(static_cast<cl_mem>(srcPtr1), 
			static_cast<cl_mem>(srcPtr2), 
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
 
// ----------------------------------------
// GPU max functions  calls 
// ----------------------------------------


RppStatus
rppi_max_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 max_cl(static_cast<cl_mem>(srcPtr1), 
			static_cast<cl_mem>(srcPtr2), 
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
rppi_max_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 max_cl(static_cast<cl_mem>(srcPtr1), 
			static_cast<cl_mem>(srcPtr2), 
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
rppi_max_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 max_cl(static_cast<cl_mem>(srcPtr1), 
			static_cast<cl_mem>(srcPtr2), 
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
 
// ----------------------------------------
// GPU histogram functions  calls 
// ----------------------------------------


RppStatus
rppi_histogram_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,Rpp32u* outputHistogram,Rpp32u bins, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_unsigned_int_range( 1, 256, &bins);

#ifdef OCL_COMPILE
 	 {
 	 histogram_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			outputHistogram,
			bins,
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
rppi_histogram_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,Rpp32u* outputHistogram,Rpp32u bins, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_unsigned_int_range( 1, 256, &bins);

#ifdef OCL_COMPILE
 	 {
 	 histogram_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			outputHistogram,
			bins,
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
rppi_histogram_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,Rpp32u* outputHistogram,Rpp32u bins, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_unsigned_int_range( 1, 256, &bins);

#ifdef OCL_COMPILE
 	 {
 	 histogram_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			outputHistogram,
			bins,
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
// GPU min_max_loc functions  calls 
// ----------------------------------------


RppStatus
rppi_min_max_loc_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,Rpp8u* min,Rpp8u* max,Rpp32u* minLoc,Rpp32u* maxLoc, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 min_max_loc_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			min,
			max,
			minLoc,
			maxLoc,
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
rppi_min_max_loc_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,Rpp8u* min,Rpp8u* max,Rpp32u* minLoc,Rpp32u* maxLoc, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 min_max_loc_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			min,
			max,
			minLoc,
			maxLoc,
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
rppi_min_max_loc_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,Rpp8u* min,Rpp8u* max,Rpp32u* minLoc,Rpp32u* maxLoc, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 min_max_loc_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			min,
			max,
			minLoc,
			maxLoc,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}