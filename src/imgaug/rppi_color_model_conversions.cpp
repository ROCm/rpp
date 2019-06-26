#include <rppi_color_model_conversions.h>
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

#include "cpu/host_color_model_conversions.hpp" 
 
// ----------------------------------------
// Host rgb_to_hsv functions calls 
// ----------------------------------------


RppStatus
rppi_rgb_to_hsv_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 rgb_to_hsv_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			 RPPI_CHN_PLANAR, 1); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_rgb_to_hsv_u8_pln1_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_rgb_to_hsv_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 rgb_to_hsv_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			 RPPI_CHN_PLANAR, 3); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_rgb_to_hsv_u8_pln3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_rgb_to_hsv_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 rgb_to_hsv_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			 RPPI_CHN_PACKED, 3); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_rgb_to_hsv_u8_pkd3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host hsv_to_rgb functions calls 
// ----------------------------------------


RppStatus
rppi_hsv_to_rgb_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 hsv_to_rgb_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			 RPPI_CHN_PLANAR, 1); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_hsv_to_rgb_u8_pln1_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_hsv_to_rgb_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 hsv_to_rgb_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			 RPPI_CHN_PLANAR, 3); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_hsv_to_rgb_u8_pln3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_hsv_to_rgb_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 hsv_to_rgb_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			 RPPI_CHN_PACKED, 3); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_hsv_to_rgb_u8_pkd3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host hueRGB functions calls 
// ----------------------------------------


RppStatus
rppi_hueRGB_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift)
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, hueShift);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 hueRGB_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), hueShift,
			 RPPI_CHN_PLANAR, 1); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_hueRGB_u8_pln1_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_hueRGB_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift)
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, hueShift);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 hueRGB_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), hueShift,
			 RPPI_CHN_PLANAR, 3); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_hueRGB_u8_pln3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_hueRGB_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift)
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, hueShift);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 hueRGB_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), hueShift,
			 RPPI_CHN_PACKED, 3); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_hueRGB_u8_pkd3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host hueHSV functions calls 
// ----------------------------------------


RppStatus
rppi_hueHSV_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift)
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, hueShift);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 hueHSV_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), hueShift,
			 RPPI_CHN_PLANAR, 1); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_hueHSV_u8_pln1_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_hueHSV_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift)
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, hueShift);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 hueHSV_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), hueShift,
			 RPPI_CHN_PLANAR, 3); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_hueHSV_u8_pln3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_hueHSV_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift)
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, hueShift);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 hueHSV_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), hueShift, 
			 RPPI_CHN_PACKED, 3); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_hueHSV_u8_pkd3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host saturationRGB functions calls 
// ----------------------------------------


RppStatus
rppi_saturationRGB_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor)
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, saturationFactor);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 saturationRGB_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), saturationFactor,
			 RPPI_CHN_PLANAR, 1); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_saturationRGB_u8_pln1_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_saturationRGB_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor)
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, saturationFactor);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 saturationRGB_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), saturationFactor,
			 RPPI_CHN_PLANAR, 3); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_saturationRGB_u8_pln3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_saturationRGB_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor)
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, saturationFactor);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 saturationRGB_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), saturationFactor,
			 RPPI_CHN_PACKED, 3); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_saturationRGB_u8_pkd3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host saturationHSV functions calls 
// ----------------------------------------


RppStatus
rppi_saturationHSV_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor)
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, saturationFactor);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 saturationHSV_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), saturationFactor,
			 RPPI_CHN_PLANAR, 1); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_saturationHSV_u8_pln1_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_saturationHSV_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor)
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, saturationFactor);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 saturationHSV_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), saturationFactor,
			 RPPI_CHN_PLANAR, 3); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_saturationHSV_u8_pln3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_saturationHSV_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor)
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, saturationFactor);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 saturationHSV_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), saturationFactor,
			 RPPI_CHN_PACKED, 3); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_saturationHSV_u8_pkd3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}
 
// ----------------------------------------
// GPU rgb_to_hsv functions  calls 
// ----------------------------------------


RppStatus
rppi_rgb_to_hsv_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 rgb_to_hsv_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_rgb_to_hsv_u8_pln1_gpu,';
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
rppi_rgb_to_hsv_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 rgb_to_hsv_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_rgb_to_hsv_u8_pln3_gpu,';
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
rppi_rgb_to_hsv_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 rgb_to_hsv_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_rgb_to_hsv_u8_pkd3_gpu,';
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
// GPU hsv_to_rgb functions  calls 
// ----------------------------------------


RppStatus
rppi_hsv_to_rgb_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 hsv_to_rgb_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_hsv_to_rgb_u8_pln1_gpu,';
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
rppi_hsv_to_rgb_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 hsv_to_rgb_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_hsv_to_rgb_u8_pln3_gpu,';
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
rppi_hsv_to_rgb_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 hsv_to_rgb_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_hsv_to_rgb_u8_pkd3_gpu,';
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
// GPU hueRGB functions  calls 
// ----------------------------------------


RppStatus
rppi_hueRGB_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, hueShift);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 hueRGB_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), hueShift, 0.0/*Saturation*/,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_hueRGB_u8_pln1_gpu,';
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
rppi_hueRGB_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, hueShift);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 hueRGB_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), hueShift, 0.0/*Saturation*/,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_hueRGB_u8_pln3_gpu,';
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
rppi_hueRGB_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, hueShift);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 hueRGB_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), hueShift, 0.0/*Saturation*/,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_hueRGB_u8_pkd3_gpu,';
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
// GPU hueHSV functions  calls 
// ----------------------------------------


RppStatus
rppi_hueHSV_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, hueShift);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 hueHSV_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), hueShift, 0.0/*Saturation*/,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_hueHSV_u8_pln1_gpu,';
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
rppi_hueHSV_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, hueShift);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 hueHSV_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), hueShift, 0.0/*Saturation*/,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_hueHSV_u8_pln3_gpu,';
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
rppi_hueHSV_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, hueShift);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 hueHSV_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), hueShift, 0.0/*Saturation*/,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_hueHSV_u8_pkd3_gpu,';
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
// GPU saturationRGB functions  calls 
// ----------------------------------------


RppStatus
rppi_saturationRGB_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, saturationFactor);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 hueRGB_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 0.0/*hue*/, saturationFactor,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_saturationRGB_u8_pln1_gpu,';
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
rppi_saturationRGB_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, saturationFactor);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 hueRGB_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 0.0/*hue*/, saturationFactor,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_saturationRGB_u8_pln3_gpu,';
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
rppi_saturationRGB_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, saturationFactor);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 hueRGB_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 0.0/*hue*/, saturationFactor,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_saturationRGB_u8_pkd3_gpu,';
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
// GPU saturationHSV functions  calls 
// ----------------------------------------


RppStatus
rppi_saturationHSV_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, saturationFactor);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 hueHSV_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr),  0.0/*hue*/, saturationFactor,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_saturationHSV_u8_pln1_gpu,';
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
rppi_saturationHSV_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, saturationFactor);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 hueHSV_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr),  0.0/*hue*/, saturationFactor,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_saturationHSV_u8_pln3_gpu,';
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
rppi_saturationHSV_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, saturationFactor);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 hueHSV_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr),  0.0/*hue*/, saturationFactor,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_saturationHSV_u8_pkd3_gpu,';
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