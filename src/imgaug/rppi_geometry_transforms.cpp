#include <rppi_geometry_transforms.h>
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

#include "cpu/host_geometry_transforms.hpp" 
 
// ----------------------------------------
// Host flip functions calls 
// ----------------------------------------


RppStatus
rppi_flip_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiAxis flipAxis)
{

 	 validate_image_size(srcSize);
 	 validate_int_range( 0, 1, flipAxis);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 flip_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			1, RPPI_CHN_PLANAR); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_flip_u8_pln1_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_flip_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiAxis flipAxis)
{

 	 validate_image_size(srcSize);
 	 validate_int_range( 0, 1, flipAxis);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 flip_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			3, RPPI_CHN_PLANAR); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_flip_u8_pln3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_flip_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiAxis flipAxis)
{

 	 validate_image_size(srcSize);
 	 validate_int_range( 0, 1, flipAxis);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 flip_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			3, RPPI_CHN_PACKED); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_flip_u8_pkd3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host resize functions calls 
// ----------------------------------------


RppStatus
rppi_resize_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize)
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 resize_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			1, RPPI_CHN_PLANAR); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_resize_u8_pln1_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_resize_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize)
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 resize_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			3, RPPI_CHN_PLANAR); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_resize_u8_pln3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_resize_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize)
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 resize_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			3, RPPI_CHN_PACKED); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_resize_u8_pkd3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host resize_crop functions calls 
// ----------------------------------------


RppStatus
rppi_resize_crop_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2)
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
 	 validate_int_range( 0, srcSize.height, x1);
 	 validate_int_range( 0, srcSize.width, y1);
 	 validate_int_range( 0, srcSize.height, x2);
 	 validate_int_range( 0, srcSize.width, y2);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 resize_crop_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			1, RPPI_CHN_PLANAR); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_resize_crop_u8_pln1_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_resize_crop_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2)
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
 	 validate_int_range( 0, srcSize.height, x1);
 	 validate_int_range( 0, srcSize.width, y1);
 	 validate_int_range( 0, srcSize.height, x2);
 	 validate_int_range( 0, srcSize.width, y2);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 resize_crop_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			3, RPPI_CHN_PLANAR); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_resize_crop_u8_pln3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_resize_crop_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2)
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
 	 validate_int_range( 0, srcSize.height, x1);
 	 validate_int_range( 0, srcSize.width, y1);
 	 validate_int_range( 0, srcSize.height, x2);
 	 validate_int_range( 0, srcSize.width, y2);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 resize_crop_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			3, RPPI_CHN_PACKED); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_resize_crop_u8_pkd3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host rotate functions calls 
// ----------------------------------------


RppStatus
rppi_rotate_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32f angleDeg)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 rotate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			1, RPPI_CHN_PLANAR); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_rotate_u8_pln1_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_rotate_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32f angleDeg)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 rotate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			3, RPPI_CHN_PLANAR); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_rotate_u8_pln3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}

RppStatus
rppi_rotate_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32f angleDeg)
{

 	 validate_image_size(srcSize);
#ifdef TIME_INFO
 	  auto start = high_resolution_clock::now(); 
#endif //TIME_INFO 

	 rotate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			3, RPPI_CHN_PACKED); 
 
#ifdef TIME_INFO  
 	 auto stop = high_resolution_clock::now();
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl;
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv');
 	 time_file<<'rppi_rotate_u8_pkd3_host,'; 
 	 time_file<<duration.count() << endl; 
 	 time_file.close();

#endif //TIME_INFO  

	return RPP_SUCCESS;
}
 
// ----------------------------------------
// GPU flip functions  calls 
// ----------------------------------------


RppStatus
rppi_flip_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiAxis flipAxis, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_range( 0, 1, flipAxis);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 flip_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_flip_u8_pln1_gpu,';
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
rppi_flip_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiAxis flipAxis, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_range( 0, 1, flipAxis);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 flip_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_flip_u8_pln3_gpu,';
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
rppi_flip_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiAxis flipAxis, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_range( 0, 1, flipAxis);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 flip_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_flip_u8_pkd3_gpu,';
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
// GPU resize functions  calls 
// ----------------------------------------


RppStatus
rppi_resize_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 resize_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_resize_u8_pln1_gpu,';
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
rppi_resize_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 resize_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_resize_u8_pln3_gpu,';
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
rppi_resize_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 resize_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_resize_u8_pkd3_gpu,';
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
// GPU resize_crop functions  calls 
// ----------------------------------------


RppStatus
rppi_resize_crop_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
 	 validate_int_range( 0, srcSize.height, x1);
 	 validate_int_range( 0, srcSize.width, y1);
 	 validate_int_range( 0, srcSize.height, x2);
 	 validate_int_range( 0, srcSize.width, y2);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 resize_crop_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_resize_crop_u8_pln1_gpu,';
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
rppi_resize_crop_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
 	 validate_int_range( 0, srcSize.height, x1);
 	 validate_int_range( 0, srcSize.width, y1);
 	 validate_int_range( 0, srcSize.height, x2);
 	 validate_int_range( 0, srcSize.width, y2);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 resize_crop_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_resize_crop_u8_pln3_gpu,';
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
rppi_resize_crop_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
 	 validate_int_range( 0, srcSize.height, x1);
 	 validate_int_range( 0, srcSize.width, y1);
 	 validate_int_range( 0, srcSize.height, x2);
 	 validate_int_range( 0, srcSize.width, y2);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 resize_crop_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_resize_crop_u8_pkd3_gpu,';
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
// GPU rotate functions  calls 
// ----------------------------------------


RppStatus
rppi_rotate_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32f angleDeg, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 rotate_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_rotate_u8_pln1_gpu,';
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
rppi_rotate_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32f angleDeg, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 rotate_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_rotate_u8_pln3_gpu,';
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
rppi_rotate_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32f angleDeg, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE 
 	 {
#ifdef TIME_INFO 
 	 auto start = high_resolution_clock::now(); 
#endif //TIME_INFO  
 	 	 	
 	 rotate_cl(static_cast<cl_mem>(srcPtr),srcSize, 
			static_cast<cl_mem>(dstPtr), 
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle)); 
 
#ifdef TIME_INFO  
 	  auto stop = high_resolution_clock::now();  
 	 auto duration = duration_cast<microseconds>(stop - start); 
 	 cout << duration.count() << endl; 
	 std::ofstream time_file;
 	 time_file.open ('rpp_time.csv'); 
 	 time_file<<'rppi_rotate_u8_pkd3_gpu,';
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