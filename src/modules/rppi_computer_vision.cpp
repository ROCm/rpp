#include <rppi_computer_vision.h>
#include <rppdefs.h>
#include "rppi_validate.hpp"

#ifdef HIP_COMPILE
#include <hip/rpp_hip_common.hpp>
#include "hip/hip_declarations.hpp"

#elif defined(OCL_COMPILE)
#include <cl/rpp_cl_common.hpp>
#include "cl/cl_declarations.hpp"
#endif //backend
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
using namespace std::chrono; 

#include "cpu/host_computer_vision.hpp" 

/*************************************** Data Object Copy ************************************/

RppStatus  
rppi_data_object_copy_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        data_object_copy_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#elif defined (HIP_COMPILE)
    {
        data_object_copy_hip_batch(
            static_cast<Rpp8u*>(srcPtr),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        data_object_copy_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        data_object_copy_hip_batch(
            static_cast<Rpp8u*>(srcPtr),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
    {
        data_object_copy_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        data_object_copy_hip_batch(
            static_cast<Rpp8u*>(srcPtr),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    data_object_copy_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr),
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 1
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    data_object_copy_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr),
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 3
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    data_object_copy_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr),
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PACKED, 3
    );

    return RPP_SUCCESS;
}

/*************************************** Local Binary Pattern ************************************/

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        local_binary_pattern_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#elif defined (HIP_COMPILE)
    {
        local_binary_pattern_hip_batch(
            static_cast<Rpp8u*>(srcPtr),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        local_binary_pattern_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        local_binary_pattern_hip_batch(
            static_cast<Rpp8u*>(srcPtr),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
    {
        local_binary_pattern_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        local_binary_pattern_hip_batch(
            static_cast<Rpp8u*>(srcPtr),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    local_binary_pattern_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr),
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 1
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    local_binary_pattern_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr),
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 3
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    local_binary_pattern_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr),
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PACKED, 3
    );

    return RPP_SUCCESS;
}

/*************************************** Canny Edge Detector ************************************/

RppStatus  
rppi_canny_edge_detector_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
    copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
    copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        canny_edge_detector_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#elif defined (HIP_COMPILE)
    {
        canny_edge_detector_hip_batch(
            static_cast<Rpp8u*>(srcPtr),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
    copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
    copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        canny_edge_detector_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        canny_edge_detector_hip_batch(
            static_cast<Rpp8u*>(srcPtr),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
    copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        canny_edge_detector_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        canny_edge_detector_hip_batch(
            static_cast<Rpp8u*>(srcPtr),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    canny_edge_detector_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr),
        minThreshold,
        maxThreshold,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 1
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    canny_edge_detector_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr),
        minThreshold,
        maxThreshold,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 3
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    canny_edge_detector_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr),
        minThreshold,
        maxThreshold,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PACKED, 3
    );

    return RPP_SUCCESS;
}

/*************************************** Harris Corner Detector ************************************/

RppStatus  
rppi_harris_corner_detector_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
    copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
    copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
    copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
    copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        harris_corner_detector_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#elif defined (HIP_COMPILE)
    {
        harris_corner_detector_hip_batch(
            static_cast<Rpp8u*>(srcPtr),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
    copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
    copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
    copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
    copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        harris_corner_detector_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        harris_corner_detector_hip_batch(
            static_cast<Rpp8u*>(srcPtr),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
    copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
    copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
    copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        harris_corner_detector_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        harris_corner_detector_hip_batch(
            static_cast<Rpp8u*>(srcPtr),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    harris_corner_detector_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr),
        gaussianKernelSize,
        stdDev,
        kernelSize,
        kValue,
        threshold,
        nonmaxKernelSize,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 1
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    harris_corner_detector_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr),
        gaussianKernelSize,
        stdDev,
        kernelSize,
        kValue,
        threshold,
        nonmaxKernelSize,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 3
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    harris_corner_detector_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr),
        gaussianKernelSize,
        stdDev,
        kernelSize,
        kValue,
        threshold,
        nonmaxKernelSize,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PACKED, 3
    );

    return RPP_SUCCESS;
}

/*************************************** Fast Corner Detector ************************************/

RppStatus  
rppi_fast_corner_detector_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
    copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
    copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        fast_corner_detector_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#elif defined (HIP_COMPILE)
    {
        fast_corner_detector_hip_batch(
            static_cast<Rpp8u*>(srcPtr),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
    copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
    copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        fast_corner_detector_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        fast_corner_detector_hip_batch(
            static_cast<Rpp8u*>(srcPtr),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
    copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        fast_corner_detector_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        fast_corner_detector_hip_batch(
            static_cast<Rpp8u*>(srcPtr),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    fast_corner_detector_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr),
        numOfPixels,
        threshold,
        nonmaxKernelSize,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 1
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    fast_corner_detector_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr),
        numOfPixels,
        threshold,
        nonmaxKernelSize,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 3
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    fast_corner_detector_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr),
        numOfPixels,
        threshold,
        nonmaxKernelSize,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PACKED, 3
    );

    return RPP_SUCCESS;
}

/*************************************** Gaussian Image Pyramid ************************************/

RppStatus  
rppi_gaussian_image_pyramid_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
    copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        gaussian_image_pyramid_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#elif defined (HIP_COMPILE)
    {
        gaussian_image_pyramid_hip_batch(
            static_cast<Rpp8u*>(srcPtr),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
    copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        gaussian_image_pyramid_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        gaussian_image_pyramid_hip_batch(
            static_cast<Rpp8u*>(srcPtr),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        gaussian_image_pyramid_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        gaussian_image_pyramid_hip_batch(
            static_cast<Rpp8u*>(srcPtr),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    gaussian_image_pyramid_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr),
        stdDev,
        kernelSize,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 1
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    gaussian_image_pyramid_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr),
        stdDev,
        kernelSize,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 3
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    gaussian_image_pyramid_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr),
        stdDev,
        kernelSize,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PACKED, 3
    );

    return RPP_SUCCESS;
}

/*************************************** Laplacian Image Pyramid ************************************/

RppStatus  
rppi_laplacian_image_pyramid_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
    copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        laplacian_image_pyramid_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#elif defined (HIP_COMPILE)
    {
        laplacian_image_pyramid_hip_batch(
            static_cast<Rpp8u*>(srcPtr),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
    copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        laplacian_image_pyramid_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        laplacian_image_pyramid_hip_batch(
            static_cast<Rpp8u*>(srcPtr),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        laplacian_image_pyramid_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        laplacian_image_pyramid_hip_batch(
            static_cast<Rpp8u*>(srcPtr),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    laplacian_image_pyramid_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr),
        stdDev,
        kernelSize,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 1
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    laplacian_image_pyramid_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr),
        stdDev,
        kernelSize,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 3
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    laplacian_image_pyramid_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr),
        stdDev,
        kernelSize,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PACKED, 3
    );

    return RPP_SUCCESS;
}

/*************************************** Remap ************************************/

RppStatus  
rppi_remap_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        remap_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr), rowRemapTable, colRemapTable,
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#elif defined (HIP_COMPILE)
    {
        remap_hip_batch(
            static_cast<Rpp8u*>(srcPtr),
            static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        remap_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr), rowRemapTable, colRemapTable,
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        remap_hip_batch(
            static_cast<Rpp8u*>(srcPtr),
            static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
    {
        remap_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr), rowRemapTable, colRemapTable,
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        remap_hip_batch(
            static_cast<Rpp8u*>(srcPtr),
            static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    remap_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 1
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    remap_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 3
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    remap_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PACKED, 3
    );

    return RPP_SUCCESS;
}

/*************************************** control_flow ************************************/

RppStatus
    rpp_bool_control_flow(bool num1, bool num2, bool *output, RppOp operation, rppHandle_t rppHandle)
{
    if(operation == RPP_SCALAR_OP_AND)
        *output = num1 & num2;
    if(operation == RPP_SCALAR_OP_OR)
        *output = num1 | num2;
    if(operation == RPP_SCALAR_OP_XOR)
        *output = num1 ^ num2;
    if(operation == RPP_SCALAR_OP_NAND)
        *output = !(num1 & num2);
    return RPP_SUCCESS;
}

RppStatus
    rpp_u8_control_flow(Rpp8u num1, Rpp8u num2, Rpp8u *output, RppOp operation, rppHandle_t rppHandle)
{
    if(operation == RPP_SCALAR_OP_ADD)
        *output = num1 + num2;
    if(operation == RPP_SCALAR_OP_SUBTRACT)
        *output = num1 - num2;
    if(operation == RPP_SCALAR_OP_MULTIPLY)
        *output = num1 * num2;
    if(operation == RPP_SCALAR_OP_MODULUS)
        *output = num1 % num2;
    if(operation == RPP_SCALAR_OP_DIVIDE)
        *output = num1 / num2;
    if(operation == RPP_SCALAR_OP_MIN)
        *output = std::min(num1, num2);
    if(operation == RPP_SCALAR_OP_MAX)
        *output = std::max(num1, num2);
    if(operation == RPP_SCALAR_OP_EQUAL)
        *output = (num1 == num2);
    if(operation == RPP_SCALAR_OP_NOTEQUAL)
        *output = (num1 != num2);
    if(operation == RPP_SCALAR_OP_LESS)
        *output = (num1 < num2);
    if(operation == RPP_SCALAR_OP_LESSEQ)
        *output = (num1 <= num2);
    if(operation == RPP_SCALAR_OP_GREATER)
        *output = (num1 > num2);
    if(operation == RPP_SCALAR_OP_GREATEREQ)
        *output = (num1 >= num2);
    return RPP_SUCCESS;
}

/*************************************** hog ************************************/

RppStatus
rppi_hog_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t binsTensor, Rpp32u *binsTensorLength, RppiSize *kernelSize, RppiSize *windowSize, Rpp32u *windowStride, Rpp32u *numOfBins, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    hog_host_batch<Rpp8u, Rpp32u>(
        static_cast<Rpp8u*>(srcPtr), 
        srcSize, 
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize, 
        static_cast<Rpp32u*>(binsTensor), 
        binsTensorLength, 
        kernelSize, 
        windowSize, 
        windowStride, 
        numOfBins, 
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 1);

    return RPP_SUCCESS;
}

RppStatus
rppi_hog_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t binsTensor, Rpp32u *binsTensorLength, RppiSize *kernelSize, RppiSize *windowSize, Rpp32u *windowStride, Rpp32u *numOfBins, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    hog_host_batch<Rpp8u, Rpp32u>(
        static_cast<Rpp8u*>(srcPtr), 
        srcSize, 
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize, 
        static_cast<Rpp32u*>(binsTensor), 
        binsTensorLength, 
        kernelSize, 
        windowSize, 
        windowStride, 
        numOfBins, 
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 3);

    return RPP_SUCCESS;
}

RppStatus
rppi_hog_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t binsTensor, Rpp32u *binsTensorLength, RppiSize *kernelSize, RppiSize *windowSize, Rpp32u *windowStride, Rpp32u *numOfBins, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    hog_host_batch<Rpp8u, Rpp32u>(
        static_cast<Rpp8u*>(srcPtr), 
        srcSize, 
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize, 
        static_cast<Rpp32u*>(binsTensor), 
        binsTensorLength, 
        kernelSize, 
        windowSize, 
        windowStride, 
        numOfBins, 
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PACKED, 3);

    return RPP_SUCCESS;
}

RppStatus
rppi_hough_lines_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t lines, 
                                    Rpp32f* rho, Rpp32f* theta, Rpp32u *threshold, 
                                    Rpp32u *minLineLength, Rpp32u *maxLineGap, Rpp32u *linesMax, 
                                    Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    hough_lines_host_batch<Rpp8u, Rpp32u>(
        static_cast<Rpp8u*>(srcPtr), 
        srcSize, 
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize, 
        static_cast<Rpp32u*>(lines), 
        rho, 
        theta, 
        threshold, 
        minLineLength, 
        maxLineGap, 
        linesMax,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 1);

    return RPP_SUCCESS;
}

RppStatus
rppi_hough_lines_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t lines, 
                                    Rpp32f* rho, Rpp32f* theta, Rpp32u *threshold, 
                                    Rpp32u *minLineLength, Rpp32u *maxLineGap, Rpp32u *linesMax, 
                                    Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    
    return RPP_SUCCESS;
}


/*************************************** Tensor Transpose ************************************/

RppStatus
rppi_tensor_transpose_u8_host(RppPtr_t srcPtr, RppPtr_t dstPtr, Rpp32u dimension1, Rpp32u dimension2, Rpp32u tensorDimension, Rpp32u *tensorDimensionValues)
{
    tensor_transpose_host<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr), 
        static_cast<Rpp8u*>(dstPtr), 
        dimension1, 
        dimension2, 
        tensorDimension, 
        tensorDimensionValues
    );

    return RPP_SUCCESS;
}

/*************************************** reconstruction_laplacian_image_pyramid ************************************/

RppStatus
rppi_reconstruction_laplacian_image_pyramid_u8_pln1_batchPD_host(
    RppPtr_t srcPtr1, RppiSize *srcSize1, RppiSize maxSrcSize1, 
    RppPtr_t srcPtr2, RppiSize *srcSize2, RppiSize maxSrcSize2, 
    RppPtr_t dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, 
    Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize2, rpp::deref(rppHandle));
    copy_host_maxDstSize(maxSrcSize1, rpp::deref(rppHandle));

    reconstruction_laplacian_image_pyramid_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr1), 
        srcSize1, 
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize, 
        static_cast<Rpp8u*>(srcPtr2), 
        srcSize2, 
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize, 
        static_cast<Rpp8u*>(dstPtr), 
        stdDev, 
        kernelSize, 
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 1);

    return RPP_SUCCESS;
}

RppStatus
rppi_reconstruction_laplacian_image_pyramid_u8_pln3_batchPD_host(
    RppPtr_t srcPtr1, RppiSize *srcSize1, RppiSize maxSrcSize1, 
    RppPtr_t srcPtr2, RppiSize *srcSize2, RppiSize maxSrcSize2, 
    RppPtr_t dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, 
    Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize2, rpp::deref(rppHandle));
    copy_host_maxDstSize(maxSrcSize1, rpp::deref(rppHandle));

    reconstruction_laplacian_image_pyramid_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr1), 
        srcSize1, 
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize, 
        static_cast<Rpp8u*>(srcPtr2), 
        srcSize2, 
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize, 
        static_cast<Rpp8u*>(dstPtr), 
        stdDev, 
        kernelSize, 
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 3);
        
    return RPP_SUCCESS;
}

RppStatus
rppi_reconstruction_laplacian_image_pyramid_u8_pkd3_batchPD_host(
    RppPtr_t srcPtr1, RppiSize *srcSize1, RppiSize maxSrcSize1, 
    RppPtr_t srcPtr2, RppiSize *srcSize2, RppiSize maxSrcSize2, 
    RppPtr_t dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, 
    Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize2, rpp::deref(rppHandle));
    copy_host_maxDstSize(maxSrcSize1, rpp::deref(rppHandle));

    reconstruction_laplacian_image_pyramid_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr1), 
        srcSize1, 
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize, 
        static_cast<Rpp8u*>(srcPtr2), 
        srcSize2, 
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize, 
        static_cast<Rpp8u*>(dstPtr), 
        stdDev, 
        kernelSize, 
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PACKED, 3);
        
    return RPP_SUCCESS;
}

// GPU

RppStatus
rppi_reconstruction_laplacian_image_pyramid_u8_pln1_batchPD_gpu(
    RppPtr_t srcPtr1, RppiSize *srcSize1, RppiSize maxSrcSize1, 
    RppPtr_t srcPtr2, RppiSize *srcSize2, RppiSize maxSrcSize2, 
    RppPtr_t dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, 
    Rpp32u nbatchSize, rppHandle_t rppHandle)
{
  

    return RPP_SUCCESS;
}

RppStatus
rppi_reconstruction_laplacian_image_pyramid_u8_pln3_batchPD_gpu(
    RppPtr_t srcPtr1, RppiSize *srcSize1, RppiSize maxSrcSize1, 
    RppPtr_t srcPtr2, RppiSize *srcSize2, RppiSize maxSrcSize2, 
    RppPtr_t dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, 
    Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize2, rpp::deref(rppHandle));
    copy_host_maxDstSize(maxSrcSize1, rpp::deref(rppHandle));

    
        
    return RPP_SUCCESS;
}

RppStatus
rppi_reconstruction_laplacian_image_pyramid_u8_pkd3_batchPD_gpu(
    RppPtr_t srcPtr1, RppiSize *srcSize1, RppiSize maxSrcSize1, 
    RppPtr_t srcPtr2, RppiSize *srcSize2, RppiSize maxSrcSize2, 
    RppPtr_t dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, 
    Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    
        
    return RPP_SUCCESS;
}

// **************************************** convert bit depth ****************************************

RppStatus
rppi_convert_bit_depth_u8s8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    convert_bit_depth_host_batch<Rpp8u, Rpp8s>(
        static_cast<Rpp8u*>(srcPtr), 
        srcSize, 
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize, 
        static_cast<Rpp8s*>(dstPtr), 
        1, 
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 1);
        
    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8u16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    convert_bit_depth_host_batch<Rpp8u, Rpp16u>(
        static_cast<Rpp8u*>(srcPtr), 
        srcSize, 
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize, 
        static_cast<Rpp16u*>(dstPtr), 
        2, 
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 1);
        
    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8s16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    convert_bit_depth_host_batch<Rpp8u, Rpp16s>(
        static_cast<Rpp8u*>(srcPtr), 
        srcSize, 
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize, 
        static_cast<Rpp16s*>(dstPtr), 
        3, 
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 1);
        
    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8s8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    convert_bit_depth_host_batch<Rpp8u, Rpp8s>(
        static_cast<Rpp8u*>(srcPtr), 
        srcSize, 
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize, 
        static_cast<Rpp8s*>(dstPtr), 
        1, 
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 3);
        
    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8u16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    convert_bit_depth_host_batch<Rpp8u, Rpp16u>(
        static_cast<Rpp8u*>(srcPtr), 
        srcSize, 
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize, 
        static_cast<Rpp16u*>(dstPtr), 
        2, 
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 3);
        
    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8s16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    convert_bit_depth_host_batch<Rpp8u, Rpp16s>(
        static_cast<Rpp8u*>(srcPtr), 
        srcSize, 
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize, 
        static_cast<Rpp16s*>(dstPtr), 
        3, 
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 3);
        
    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8s8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    convert_bit_depth_host_batch<Rpp8u, Rpp8s>(
        static_cast<Rpp8u*>(srcPtr), 
        srcSize, 
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize, 
        static_cast<Rpp8s*>(dstPtr), 
        1, 
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PACKED, 3);
        
    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8u16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    convert_bit_depth_host_batch<Rpp8u, Rpp16u>(
        static_cast<Rpp8u*>(srcPtr), 
        srcSize, 
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize, 
        static_cast<Rpp16u*>(dstPtr), 
        2, 
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PACKED, 3);
        
    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8s16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    convert_bit_depth_host_batch<Rpp8u, Rpp16s>(
        static_cast<Rpp8u*>(srcPtr), 
        srcSize, 
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize, 
        static_cast<Rpp16s*>(dstPtr), 
        3, 
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PACKED, 3);
        
    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8s8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        convert_bit_depth_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            1,
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#elif defined (HIP_COMPILE)
    {
        // to add
    }
#endif //BACKEND
        
    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8u16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        convert_bit_depth_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            2,
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#elif defined (HIP_COMPILE)
    {
        // to add
    }
#endif //BACKEND
        
    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8s16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        convert_bit_depth_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            3,
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#elif defined (HIP_COMPILE)
    {
        // to add
    }
#endif //BACKEND
        
    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8s8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        convert_bit_depth_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            1,
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        // to add
    }
#endif //BACKEND
        
    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8u16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        convert_bit_depth_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            2,
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        // to add
    }
#endif //BACKEND
        
    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8s16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        convert_bit_depth_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            3,
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        // to add
    }
#endif //BACKEND
        
    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8s8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
    {
        convert_bit_depth_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            1,
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        // to add
    }
#endif //BACKEND
        
    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8u16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
    {
        convert_bit_depth_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            2,
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        // to add
    }
#endif //BACKEND
        
    return RPP_SUCCESS;
}

RppStatus
rppi_convert_bit_depth_u8s16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
    {
        convert_bit_depth_cl_batch(
            static_cast<cl_mem>(srcPtr),
            static_cast<cl_mem>(dstPtr),
            3,
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        // to add
    }
#endif //BACKEND
        
    return RPP_SUCCESS;
}

// **************************************** tensor convert bit depth ****************************************

RppStatus
rppi_tensor_convert_bit_depth_u8s8_host(RppPtr_t srcPtr, RppPtr_t dstPtr, 
                                        Rpp32u tensorDimension, RppPtr_t tensorDimensionValues)
{
    tensor_convert_bit_depth_host<Rpp8u, Rpp8s>(static_cast<Rpp8u*>(srcPtr), static_cast<Rpp8s*>(dstPtr), 
                                                1, 
                                                tensorDimension, static_cast<Rpp32u*>(tensorDimensionValues));

    return RPP_SUCCESS;
}

RppStatus
rppi_tensor_convert_bit_depth_u8u16_host(RppPtr_t srcPtr, RppPtr_t dstPtr, 
                                        Rpp32u tensorDimension, RppPtr_t tensorDimensionValues)
{
    tensor_convert_bit_depth_host<Rpp8u, Rpp16u>(static_cast<Rpp8u*>(srcPtr), static_cast<Rpp16u*>(dstPtr), 
                                                2, 
                                                tensorDimension, static_cast<Rpp32u*>(tensorDimensionValues));

    return RPP_SUCCESS;
}

RppStatus
rppi_tensor_convert_bit_depth_u8s16_host(RppPtr_t srcPtr, RppPtr_t dstPtr, 
                                        Rpp32u tensorDimension, RppPtr_t tensorDimensionValues)
{
    tensor_convert_bit_depth_host<Rpp8u, Rpp16s>(static_cast<Rpp8u*>(srcPtr), static_cast<Rpp16s*>(dstPtr), 
                                                3, 
                                                tensorDimension, static_cast<Rpp32u*>(tensorDimensionValues));

    return RPP_SUCCESS;
}