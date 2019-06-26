#include <rppi_logical_operations.h>
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

#include "cpu/host_logical_operations.hpp" 
 
// ----------------------------------------
// Host bitwise_AND functions calls 
// ----------------------------------------


RppStatus
rppi_bitwise_AND_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 bitwise_AND_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PLANAR, 1); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<milliseconds>(stop - start); 
 	 std::cout << duration.count() << std::endl;
	  std::fstream time_file;
 	 time_file.open ("rpp_time.csv",std::fstream::in | std::fstream::out |std::fstream::app);
 	 time_file<<"rppi_bitwise_AND_u8_pln1_host,"; 
 	 time_file<<duration.count() << std::endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_bitwise_AND_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 bitwise_AND_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PLANAR, 3); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<milliseconds>(stop - start); 
 	 std::cout << duration.count() << std::endl;
	  std::fstream time_file;
 	 time_file.open ("rpp_time.csv",std::fstream::in | std::fstream::out |std::fstream::app);
 	 time_file<<"rppi_bitwise_AND_u8_pln3_host,"; 
 	 time_file<<duration.count() << std::endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_bitwise_AND_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 bitwise_AND_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PACKED, 3); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<milliseconds>(stop - start); 
 	 std::cout << duration.count() << std::endl;
	  std::fstream time_file;
 	 time_file.open ("rpp_time.csv",std::fstream::in | std::fstream::out |std::fstream::app);
 	 time_file<<"rppi_bitwise_AND_u8_pkd3_host,"; 
 	 time_file<<duration.count() << std::endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host bitwise_NOT functions calls 
// ----------------------------------------


RppStatus
rppi_bitwise_NOT_u8_pln1_host(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 bitwise_NOT_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PLANAR, 1); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<milliseconds>(stop - start); 
 	 std::cout << duration.count() << std::endl;
	  std::fstream time_file;
 	 time_file.open ("rpp_time.csv",std::fstream::in | std::fstream::out |std::fstream::app);
 	 time_file<<"rppi_bitwise_NOT_u8_pln1_host,"; 
 	 time_file<<duration.count() << std::endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_bitwise_NOT_u8_pln3_host(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 bitwise_NOT_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PLANAR, 3); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<milliseconds>(stop - start); 
 	 std::cout << duration.count() << std::endl;
	  std::fstream time_file;
 	 time_file.open ("rpp_time.csv",std::fstream::in | std::fstream::out |std::fstream::app);
 	 time_file<<"rppi_bitwise_NOT_u8_pln3_host,"; 
 	 time_file<<duration.count() << std::endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_bitwise_NOT_u8_pkd3_host(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 bitwise_NOT_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PACKED, 3); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<milliseconds>(stop - start); 
 	 std::cout << duration.count() << std::endl;
	  std::fstream time_file;
 	 time_file.open ("rpp_time.csv",std::fstream::in | std::fstream::out |std::fstream::app);
 	 time_file<<"rppi_bitwise_NOT_u8_pkd3_host,"; 
 	 time_file<<duration.count() << std::endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host exclusive_OR functions calls 
// ----------------------------------------


RppStatus
rppi_exclusive_OR_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 exclusive_OR_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PLANAR, 1); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<milliseconds>(stop - start); 
 	 std::cout << duration.count() << std::endl;
	  std::fstream time_file;
 	 time_file.open ("rpp_time.csv",std::fstream::in | std::fstream::out |std::fstream::app);
 	 time_file<<"rppi_exclusive_OR_u8_pln1_host,"; 
 	 time_file<<duration.count() << std::endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_exclusive_OR_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 exclusive_OR_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PLANAR, 3); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<milliseconds>(stop - start); 
 	 std::cout << duration.count() << std::endl;
	  std::fstream time_file;
 	 time_file.open ("rpp_time.csv",std::fstream::in | std::fstream::out |std::fstream::app);
 	 time_file<<"rppi_exclusive_OR_u8_pln3_host,"; 
 	 time_file<<duration.count() << std::endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_exclusive_OR_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 exclusive_OR_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PACKED, 3); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<milliseconds>(stop - start); 
 	 std::cout << duration.count() << std::endl;
	  std::fstream time_file;
 	 time_file.open ("rpp_time.csv",std::fstream::in | std::fstream::out |std::fstream::app);
 	 time_file<<"rppi_exclusive_OR_u8_pkd3_host,"; 
 	 time_file<<duration.count() << std::endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host inclusive_OR functions calls 
// ----------------------------------------


RppStatus
rppi_inclusive_OR_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 inclusive_OR_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PLANAR, 1); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<milliseconds>(stop - start); 
 	 std::cout << duration.count() << std::endl;
	  std::fstream time_file;
 	 time_file.open ("rpp_time.csv",std::fstream::in | std::fstream::out |std::fstream::app);
 	 time_file<<"rppi_inclusive_OR_u8_pln1_host,"; 
 	 time_file<<duration.count() << std::endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_inclusive_OR_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 inclusive_OR_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PLANAR, 3); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<milliseconds>(stop - start); 
 	 std::cout << duration.count() << std::endl;
	  std::fstream time_file;
 	 time_file.open ("rpp_time.csv",std::fstream::in | std::fstream::out |std::fstream::app);
 	 time_file<<"rppi_inclusive_OR_u8_pln3_host,"; 
 	 time_file<<duration.count() << std::endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_inclusive_OR_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 inclusive_OR_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			RPPI_CHN_PACKED, 3); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<milliseconds>(stop - start); 
 	 std::cout << duration.count() << std::endl;
	  std::fstream time_file;
 	 time_file.open ("rpp_time.csv",std::fstream::in | std::fstream::out |std::fstream::app);
 	 time_file<<"rppi_inclusive_OR_u8_pkd3_host,"; 
 	 time_file<<duration.count() << std::endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}
 
// ----------------------------------------
// GPU bitwise_AND functions  calls 
// ----------------------------------------


RppStatus
rppi_bitwise_AND_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 bitwise_AND_cl(static_cast<cl_mem>(srcPtr1),static_cast<cl_mem>(srcPtr2),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<milliseconds>(stop - start); 
 	 std::cout << duration.count() << std::endl; 
	  std::fstream time_file;
 	 time_file.open ("rpp_time.csv",std::fstream::in | std::fstream::out |std::fstream::app); 
 	 time_file<<"rppi_bitwise_AND_u8_pln1_gpu,";
 	 time_file<<duration.count() << std::endl;  
 	 time_file.close();
#endif //TIME_INFO  
 	 }
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_bitwise_AND_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 bitwise_AND_cl(static_cast<cl_mem>(srcPtr1),static_cast<cl_mem>(srcPtr2),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<milliseconds>(stop - start); 
 	 std::cout << duration.count() << std::endl; 
	  std::fstream time_file;
 	 time_file.open ("rpp_time.csv",std::fstream::in | std::fstream::out |std::fstream::app); 
 	 time_file<<"rppi_bitwise_AND_u8_pln3_gpu,";
 	 time_file<<duration.count() << std::endl;  
 	 time_file.close();
#endif //TIME_INFO  
 	 }
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_bitwise_AND_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 bitwise_AND_cl(static_cast<cl_mem>(srcPtr1),static_cast<cl_mem>(srcPtr2),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<milliseconds>(stop - start); 
 	 std::cout << duration.count() << std::endl; 
	  std::fstream time_file;
 	 time_file.open ("rpp_time.csv",std::fstream::in | std::fstream::out |std::fstream::app); 
 	 time_file<<"rppi_bitwise_AND_u8_pkd3_gpu,";
 	 time_file<<duration.count() << std::endl;  
 	 time_file.close();
#endif //TIME_INFO  
 	 }
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}
 
// ----------------------------------------
// GPU bitwise_NOT functions  calls 
// ----------------------------------------


RppStatus
rppi_bitwise_NOT_u8_pln1_gpu(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 bitwise_NOT_cl(static_cast<cl_mem>(srcPtr1),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<milliseconds>(stop - start); 
 	 std::cout << duration.count() << std::endl; 
	  std::fstream time_file;
 	 time_file.open ("rpp_time.csv",std::fstream::in | std::fstream::out |std::fstream::app); 
 	 time_file<<"rppi_bitwise_NOT_u8_pln1_gpu,";
 	 time_file<<duration.count() << std::endl;  
 	 time_file.close();
#endif //TIME_INFO  
 	 }
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_bitwise_NOT_u8_pln3_gpu(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 bitwise_NOT_cl(static_cast<cl_mem>(srcPtr1),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<milliseconds>(stop - start); 
 	 std::cout << duration.count() << std::endl; 
	  std::fstream time_file;
 	 time_file.open ("rpp_time.csv",std::fstream::in | std::fstream::out |std::fstream::app); 
 	 time_file<<"rppi_bitwise_NOT_u8_pln3_gpu,";
 	 time_file<<duration.count() << std::endl;  
 	 time_file.close();
#endif //TIME_INFO  
 	 }
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_bitwise_NOT_u8_pkd3_gpu(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 bitwise_NOT_cl(static_cast<cl_mem>(srcPtr1),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<milliseconds>(stop - start); 
 	 std::cout << duration.count() << std::endl; 
	  std::fstream time_file;
 	 time_file.open ("rpp_time.csv",std::fstream::in | std::fstream::out |std::fstream::app); 
 	 time_file<<"rppi_bitwise_NOT_u8_pkd3_gpu,";
 	 time_file<<duration.count() << std::endl;  
 	 time_file.close();
#endif //TIME_INFO 
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}
 
// ----------------------------------------
// GPU exclusive_OR functions  calls 
// ----------------------------------------


RppStatus
rppi_exclusive_OR_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 exclusive_OR_cl(static_cast<cl_mem>(srcPtr1),static_cast<cl_mem>(srcPtr2),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<milliseconds>(stop - start); 
 	 std::cout << duration.count() << std::endl; 
	  std::fstream time_file;
 	 time_file.open ("rpp_time.csv",std::fstream::in | std::fstream::out |std::fstream::app); 
 	 time_file<<"rppi_exclusive_OR_u8_pln1_gpu,";
 	 time_file<<duration.count() << std::endl;  
 	 time_file.close();
#endif //TIME_INFO  
 	 }
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_exclusive_OR_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 exclusive_OR_cl(static_cast<cl_mem>(srcPtr1),static_cast<cl_mem>(srcPtr2),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<milliseconds>(stop - start); 
 	 std::cout << duration.count() << std::endl; 
	  std::fstream time_file;
 	 time_file.open ("rpp_time.csv",std::fstream::in | std::fstream::out |std::fstream::app); 
 	 time_file<<"rppi_exclusive_OR_u8_pln3_gpu,";
 	 time_file<<duration.count() << std::endl;  
 	 time_file.close();
#endif //TIME_INFO  
 	 }
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_exclusive_OR_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 exclusive_OR_cl(static_cast<cl_mem>(srcPtr1),static_cast<cl_mem>(srcPtr2),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<milliseconds>(stop - start); 
 	 std::cout << duration.count() << std::endl; 
	  std::fstream time_file;
 	 time_file.open ("rpp_time.csv",std::fstream::in | std::fstream::out |std::fstream::app); 
 	 time_file<<"rppi_exclusive_OR_u8_pkd3_gpu,";
 	 time_file<<duration.count() << std::endl;  
 	 time_file.close();
#endif //TIME_INFO  
 	 }
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}
 
// ----------------------------------------
// GPU inclusive_OR functions  calls 
// ----------------------------------------


RppStatus
rppi_inclusive_OR_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 inclusive_OR_cl(static_cast<cl_mem>(srcPtr1),static_cast<cl_mem>(srcPtr2),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<milliseconds>(stop - start); 
 	 std::cout << duration.count() << std::endl; 
	  std::fstream time_file;
 	 time_file.open ("rpp_time.csv",std::fstream::in | std::fstream::out |std::fstream::app); 
 	 time_file<<"rppi_inclusive_OR_u8_pln1_gpu,";
 	 time_file<<duration.count() << std::endl;  
 	 time_file.close();
#endif //TIME_INFO  
 	 }
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_inclusive_OR_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 inclusive_OR_cl(static_cast<cl_mem>(srcPtr1),static_cast<cl_mem>(srcPtr2),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<milliseconds>(stop - start); 
 	 std::cout << duration.count() << std::endl; 
	  std::fstream time_file;
 	 time_file.open ("rpp_time.csv",std::fstream::in | std::fstream::out |std::fstream::app); 
 	 time_file<<"rppi_inclusive_OR_u8_pln3_gpu,";
 	 time_file<<duration.count() << std::endl;  
 	 time_file.close();
#endif //TIME_INFO  
 	 }
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_inclusive_OR_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 inclusive_OR_cl(static_cast<cl_mem>(srcPtr1),static_cast<cl_mem>(srcPtr2),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<milliseconds>(stop - start); 
 	 std::cout << duration.count() << std::endl; 
	  std::fstream time_file;
 	 time_file.open ("rpp_time.csv",std::fstream::in | std::fstream::out |std::fstream::app); 
 	 time_file<<"rppi_inclusive_OR_u8_pkd3_gpu,";
 	 time_file<<duration.count() << std::endl;  
 	 time_file.close();
#endif //TIME_INFO  
 	 }
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}