#include <rppi_image_augmentations.h>
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

#include "cpu/host_image_augmentations.hpp" 
 
// ----------------------------------------
// Host blur functions calls 
// ----------------------------------------


RppStatus
rppi_blur_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev)
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, stdDev);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 blur_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			1, RPPI_CHN_PLANAR); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_blur_u8_pln1_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_blur_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev)
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, stdDev);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 blur_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			3, RPPI_CHN_PLANAR); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_blur_u8_pln3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_blur_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev)
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, stdDev);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 blur_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			3, RPPI_CHN_PACKED); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_blur_u8_pkd3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host contrast functions calls 
// ----------------------------------------


RppStatus
rppi_contrast_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u newMin,Rpp32u newMax)
{

 	 validate_image_size(srcSize);
 	 validate_int_max(newMax, newMin);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 contrast_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			1, RPPI_CHN_PLANAR); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_contrast_u8_pln1_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_contrast_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u newMin,Rpp32u newMax)
{

 	 validate_image_size(srcSize);
 	 validate_int_max(newMax, newMin);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 contrast_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			3, RPPI_CHN_PLANAR); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_contrast_u8_pln3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_contrast_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u newMin,Rpp32u newMax)
{

 	 validate_image_size(srcSize);
 	 validate_int_max(newMax, newMin);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 contrast_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			3, RPPI_CHN_PACKED); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_contrast_u8_pkd3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host brightness functions calls 
// ----------------------------------------


RppStatus
rppi_brightness_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha,Rpp32f beta)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 2, alpha);
 	 validate_float_range( 0, 255, beta);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 brightness_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			1, RPPI_CHN_PLANAR); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_brightness_u8_pln1_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_brightness_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha,Rpp32f beta)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 2, alpha);
 	 validate_float_range( 0, 255, beta);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 brightness_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			3, RPPI_CHN_PLANAR); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_brightness_u8_pln3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_brightness_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha,Rpp32f beta)
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 2, alpha);
 	 validate_float_range( 0, 255, beta);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 brightness_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			3, RPPI_CHN_PACKED); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_brightness_u8_pkd3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host gamma_correction functions calls 
// ----------------------------------------


RppStatus
rppi_gamma_correction_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f gamma)
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, gamma);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 gamma_correction_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			1, RPPI_CHN_PLANAR); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_gamma_correction_u8_pln1_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_gamma_correction_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f gamma)
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, gamma);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 gamma_correction_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			3, RPPI_CHN_PLANAR); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_gamma_correction_u8_pln3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_gamma_correction_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f gamma)
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, gamma);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 gamma_correction_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			3, RPPI_CHN_PACKED); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_gamma_correction_u8_pkd3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}
 
// ----------------------------------------
// GPU blur functions  calls 
// ----------------------------------------


RppStatus
rppi_blur_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, stdDev);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 blur_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_blur_u8_pln1_gpu,';
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
rppi_blur_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, stdDev);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 blur_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_blur_u8_pln3_gpu,';
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
rppi_blur_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, stdDev);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 blur_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_blur_u8_pkd3_gpu,';
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
// GPU contrast functions  calls 
// ----------------------------------------


RppStatus
rppi_contrast_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u newMin,Rpp32u newMax, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_max(newMax, newMin);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 contrast_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_contrast_u8_pln1_gpu,';
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
rppi_contrast_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u newMin,Rpp32u newMax, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_max(newMax, newMin);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 contrast_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_contrast_u8_pln3_gpu,';
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
rppi_contrast_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u newMin,Rpp32u newMax, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_max(newMax, newMin);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 contrast_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_contrast_u8_pkd3_gpu,';
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
// GPU brightness functions  calls 
// ----------------------------------------


RppStatus
rppi_brightness_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha,Rpp32f beta, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 2, alpha);
 	 validate_float_range( 0, 255, beta);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 brightness_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_brightness_u8_pln1_gpu,';
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
rppi_brightness_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha,Rpp32f beta, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 2, alpha);
 	 validate_float_range( 0, 255, beta);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 brightness_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_brightness_u8_pln3_gpu,';
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
rppi_brightness_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha,Rpp32f beta, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_float_range( 0, 2, alpha);
 	 validate_float_range( 0, 255, beta);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 brightness_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_brightness_u8_pkd3_gpu,';
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
// GPU gamma_correction functions  calls 
// ----------------------------------------


RppStatus
rppi_gamma_correction_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f gamma, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, gamma);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 gamma_correction_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_gamma_correction_u8_pln1_gpu,';
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
rppi_gamma_correction_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f gamma, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, gamma);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 gamma_correction_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_gamma_correction_u8_pln3_gpu,';
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
rppi_gamma_correction_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f gamma, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, gamma);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 gamma_correction_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_gamma_correction_u8_pkd3_gpu,';
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