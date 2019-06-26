#include <rppi_arithmetic_operations.h>
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

#include "cpu/host_arithmetic_operations.hpp" 
 
// ----------------------------------------
// Host accumulate_weighted functions calls 
// ----------------------------------------


RppStatus
rppi_accumulate_weighted_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,Rpp64f alpha)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, alpha);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 
	 accumulate_weighted_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
			alpha, 
			1, RPPI_CHN_PLANAR); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_accumulate_weighted_u8_pln1_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_accumulate_weighted_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,Rpp64f alpha)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, alpha);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 accumulate_weighted_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
			alpha, 
			3, RPPI_CHN_PLANAR); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_accumulate_weighted_u8_pln3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_accumulate_weighted_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,Rpp64f alpha)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, alpha);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 accumulate_weighted_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
			alpha, 
			3, RPPI_CHN_PACKED); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_accumulate_weighted_u8_pkd3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host absolute_difference functions calls 
// ----------------------------------------


RppStatus
rppi_absolute_difference_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 absolute_difference_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			1, RPPI_CHN_PLANAR); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_absolute_difference_u8_pln1_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_absolute_difference_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 absolute_difference_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			3, RPPI_CHN_PLANAR); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_absolute_difference_u8_pln3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_absolute_difference_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 absolute_difference_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			3, RPPI_CHN_PACKED); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_absolute_difference_u8_pkd3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host add functions calls 
// ----------------------------------------


RppStatus
rppi_add_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 add_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			1, RPPI_CHN_PLANAR); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_add_u8_pln1_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_add_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 add_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			3, RPPI_CHN_PLANAR); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_add_u8_pln3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_add_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 add_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			3, RPPI_CHN_PACKED); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_add_u8_pkd3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host subtract functions calls 
// ----------------------------------------


RppStatus
rppi_subtract_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 subtract_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			1, RPPI_CHN_PLANAR); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_subtract_u8_pln1_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_subtract_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 subtract_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			3, RPPI_CHN_PLANAR); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_subtract_u8_pln3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_subtract_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 subtract_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			3, RPPI_CHN_PACKED); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_subtract_u8_pkd3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host accumulate functions calls 
// ----------------------------------------


RppStatus
rppi_accumulate_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 accumulate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
			1, RPPI_CHN_PLANAR); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_accumulate_u8_pln1_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_accumulate_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 accumulate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
			3, RPPI_CHN_PLANAR); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_accumulate_u8_pln3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_accumulate_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 accumulate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize,
			3, RPPI_CHN_PACKED); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_accumulate_u8_pkd3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}
 
// ----------------------------------------
// GPU accumulate_weighted functions  calls 
// ----------------------------------------


RppStatus
rppi_accumulate_weighted_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,Rpp64f alpha, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, alpha);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 accumulate_weighted_cl(static_cast<cl_mem>(srcPtr1),static_cast<cl_mem>(srcPtr2),srcSize, 
			alpha, 
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_accumulate_weighted_u8_pln1_gpu,';
 	 time_file<<duration.count() << endl;  
 	 time_file.close();
 	 }
#endif //TIME_INFO  
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_accumulate_weighted_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,Rpp64f alpha, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, alpha);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 accumulate_weighted_cl(static_cast<cl_mem>(srcPtr1),static_cast<cl_mem>(srcPtr2),srcSize, 
			alpha, 
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_accumulate_weighted_u8_pln3_gpu,';
 	 time_file<<duration.count() << endl;  
 	 time_file.close();
 	 }
#endif //TIME_INFO  
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_accumulate_weighted_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,Rpp64f alpha, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 1, alpha);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 accumulate_weighted_cl(static_cast<cl_mem>(srcPtr1),static_cast<cl_mem>(srcPtr2),srcSize, 
			alpha, 
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_accumulate_weighted_u8_pkd3_gpu,';
 	 time_file<<duration.count() << endl;  
 	 time_file.close();
 	 }
#endif //TIME_INFO  
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}
 
// ----------------------------------------
// GPU absolute_difference functions  calls 
// ----------------------------------------


RppStatus
rppi_absolute_difference_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 absolute_difference_cl(static_cast<cl_mem>(srcPtr1),static_cast<cl_mem>(srcPtr2),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_absolute_difference_u8_pln1_gpu,';
 	 time_file<<duration.count() << endl;  
 	 time_file.close();
 	 }
#endif //TIME_INFO  
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_absolute_difference_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 absolute_difference_cl(static_cast<cl_mem>(srcPtr1),static_cast<cl_mem>(srcPtr2),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_absolute_difference_u8_pln3_gpu,';
 	 time_file<<duration.count() << endl;  
 	 time_file.close();
 	 }
#endif //TIME_INFO  
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_absolute_difference_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 absolute_difference_cl(static_cast<cl_mem>(srcPtr1),static_cast<cl_mem>(srcPtr2),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_absolute_difference_u8_pkd3_gpu,';
 	 time_file<<duration.count() << endl;  
 	 time_file.close();
 	 }
#endif //TIME_INFO  
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}
 
// ----------------------------------------
// GPU add functions  calls 
// ----------------------------------------


RppStatus
rppi_add_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 add_cl(static_cast<cl_mem>(srcPtr1),static_cast<cl_mem>(srcPtr2),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_add_u8_pln1_gpu,';
 	 time_file<<duration.count() << endl;  
 	 time_file.close();
 	 }
#endif //TIME_INFO  
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_add_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 add_cl(static_cast<cl_mem>(srcPtr1),static_cast<cl_mem>(srcPtr2),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_add_u8_pln3_gpu,';
 	 time_file<<duration.count() << endl;  
 	 time_file.close();
 	 }
#endif //TIME_INFO  
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_add_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 add_cl(static_cast<cl_mem>(srcPtr1),static_cast<cl_mem>(srcPtr2),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_add_u8_pkd3_gpu,';
 	 time_file<<duration.count() << endl;  
 	 time_file.close();
 	 }
#endif //TIME_INFO  
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}
 
// ----------------------------------------
// GPU subtract functions  calls 
// ----------------------------------------


RppStatus
rppi_subtract_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 subtract_cl(static_cast<cl_mem>(srcPtr1),static_cast<cl_mem>(srcPtr2),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_subtract_u8_pln1_gpu,';
 	 time_file<<duration.count() << endl;  
 	 time_file.close();
 	 }
#endif //TIME_INFO  
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_subtract_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 subtract_cl(static_cast<cl_mem>(srcPtr1),static_cast<cl_mem>(srcPtr2),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_subtract_u8_pln3_gpu,';
 	 time_file<<duration.count() << endl;  
 	 time_file.close();
 	 }
#endif //TIME_INFO  
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_subtract_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 subtract_cl(static_cast<cl_mem>(srcPtr1),static_cast<cl_mem>(srcPtr2),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_subtract_u8_pkd3_gpu,';
 	 time_file<<duration.count() << endl;  
 	 time_file.close();
 	 }
#endif //TIME_INFO  
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}
 
// ----------------------------------------
// GPU accumulate functions  calls 
// ----------------------------------------


RppStatus
rppi_accumulate_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 accumulate_cl(static_cast<cl_mem>(srcPtr1),static_cast<cl_mem>(srcPtr2),srcSize, 
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_accumulate_u8_pln1_gpu,';
 	 time_file<<duration.count() << endl;  
 	 time_file.close();
 	 }
#endif //TIME_INFO  
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_accumulate_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 accumulate_cl(static_cast<cl_mem>(srcPtr1),static_cast<cl_mem>(srcPtr2),srcSize, 
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_accumulate_u8_pln3_gpu,';
 	 time_file<<duration.count() << endl;  
 	 time_file.close();
 	 }
#endif //TIME_INFO  
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_accumulate_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 accumulate_cl(static_cast<cl_mem>(srcPtr1),static_cast<cl_mem>(srcPtr2),srcSize, 
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_accumulate_u8_pkd3_gpu,';
 	 time_file<<duration.count() << endl;  
 	 time_file.close();
 	 }
#endif //TIME_INFO  
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}