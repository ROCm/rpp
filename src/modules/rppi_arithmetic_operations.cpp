#include <rppi_arithmetic_operations.h>
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

#include "cpu/host_arithmetic_operations.hpp" 

RppStatus  
rppi_add_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
        add_cl_batch(
            static_cast<cl_mem>(srcPtr1),
            static_cast<cl_mem>(srcPtr2),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#elif defined (HIP_COMPILE)
    {
        add_hip_batch(
            static_cast<Rpp8u*>(srcPtr1),
            static_cast<Rpp8u*>(srcPtr2),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_add_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
        add_cl_batch(
            static_cast<cl_mem>(srcPtr1),
            static_cast<cl_mem>(srcPtr2),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        add_hip_batch(
            static_cast<Rpp8u*>(srcPtr1),
            static_cast<Rpp8u*>(srcPtr2),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_add_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
        add_cl_batch(
            static_cast<cl_mem>(srcPtr1),
            static_cast<cl_mem>(srcPtr2),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        add_hip_batch(
            static_cast<Rpp8u*>(srcPtr1),
            static_cast<Rpp8u*>(srcPtr2),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_add_u8_pln1_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    add_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr1),
        static_cast<Rpp8u*>(srcPtr2),
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
rppi_add_u8_pln3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    add_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr1),
        static_cast<Rpp8u*>(srcPtr2),
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
rppi_add_u8_pkd3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    add_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr1),
        static_cast<Rpp8u*>(srcPtr2),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr),
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PACKED, 3
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_subtract_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
        subtract_cl_batch(
            static_cast<cl_mem>(srcPtr1),
            static_cast<cl_mem>(srcPtr2),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#elif defined (HIP_COMPILE)
    {
        subtract_hip_batch(
            static_cast<Rpp8u*>(srcPtr1),
            static_cast<Rpp8u*>(srcPtr2),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_subtract_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
        subtract_cl_batch(
            static_cast<cl_mem>(srcPtr1),
            static_cast<cl_mem>(srcPtr2),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        subtract_hip_batch(
            static_cast<Rpp8u*>(srcPtr1),
            static_cast<Rpp8u*>(srcPtr2),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_subtract_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
        subtract_cl_batch(
            static_cast<cl_mem>(srcPtr1),
            static_cast<cl_mem>(srcPtr2),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        subtract_hip_batch(
            static_cast<Rpp8u*>(srcPtr1),
            static_cast<Rpp8u*>(srcPtr2),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_subtract_u8_pln1_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    subtract_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr1),
        static_cast<Rpp8u*>(srcPtr2),
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
rppi_subtract_u8_pln3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    subtract_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr1),
        static_cast<Rpp8u*>(srcPtr2),
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
rppi_subtract_u8_pkd3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    subtract_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr1),
        static_cast<Rpp8u*>(srcPtr2),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr),
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PACKED, 3
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_multiply_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
        multiply_cl_batch(
            static_cast<cl_mem>(srcPtr1),
            static_cast<cl_mem>(srcPtr2),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#elif defined (HIP_COMPILE)
    {
        multiply_hip_batch(
            static_cast<Rpp8u*>(srcPtr1),
            static_cast<Rpp8u*>(srcPtr2),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_multiply_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
        multiply_cl_batch(
            static_cast<cl_mem>(srcPtr1),
            static_cast<cl_mem>(srcPtr2),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        multiply_hip_batch(
            static_cast<Rpp8u*>(srcPtr1),
            static_cast<Rpp8u*>(srcPtr2),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_multiply_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
        multiply_cl_batch(
            static_cast<cl_mem>(srcPtr1),
            static_cast<cl_mem>(srcPtr2),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        multiply_hip_batch(
            static_cast<Rpp8u*>(srcPtr1),
            static_cast<Rpp8u*>(srcPtr2),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_multiply_u8_pln1_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    multiply_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr1),
        static_cast<Rpp8u*>(srcPtr2),
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
rppi_multiply_u8_pln3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    multiply_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr1),
        static_cast<Rpp8u*>(srcPtr2),
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
rppi_multiply_u8_pkd3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    multiply_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr1),
        static_cast<Rpp8u*>(srcPtr2),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr),
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PACKED, 3
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_absolute_difference_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
        absolute_difference_cl_batch(
            static_cast<cl_mem>(srcPtr1),
            static_cast<cl_mem>(srcPtr2),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#elif defined (HIP_COMPILE)
    {
        absolute_difference_hip_batch(
            static_cast<Rpp8u*>(srcPtr1),
            static_cast<Rpp8u*>(srcPtr2),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_absolute_difference_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
        absolute_difference_cl_batch(
            static_cast<cl_mem>(srcPtr1),
            static_cast<cl_mem>(srcPtr2),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        absolute_difference_hip_batch(
            static_cast<Rpp8u*>(srcPtr1),
            static_cast<Rpp8u*>(srcPtr2),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_absolute_difference_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
        absolute_difference_cl_batch(
            static_cast<cl_mem>(srcPtr1),
            static_cast<cl_mem>(srcPtr2),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        absolute_difference_hip_batch(
            static_cast<Rpp8u*>(srcPtr1),
            static_cast<Rpp8u*>(srcPtr2),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_absolute_difference_u8_pln1_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    absolute_difference_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr1),
        static_cast<Rpp8u*>(srcPtr2),
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
rppi_absolute_difference_u8_pln3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    absolute_difference_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr1),
        static_cast<Rpp8u*>(srcPtr2),
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
rppi_absolute_difference_u8_pkd3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    absolute_difference_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr1),
        static_cast<Rpp8u*>(srcPtr2),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr),
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PACKED, 3
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_phase_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
        phase_cl_batch(
            static_cast<cl_mem>(srcPtr1),
            static_cast<cl_mem>(srcPtr2),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#elif defined (HIP_COMPILE)
    {
        phase_hip_batch(
            static_cast<Rpp8u*>(srcPtr1),
            static_cast<Rpp8u*>(srcPtr2),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_phase_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
        phase_cl_batch(
            static_cast<cl_mem>(srcPtr1),
            static_cast<cl_mem>(srcPtr2),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        phase_hip_batch(
            static_cast<Rpp8u*>(srcPtr1),
            static_cast<Rpp8u*>(srcPtr2),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_phase_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
        phase_cl_batch(
            static_cast<cl_mem>(srcPtr1),
            static_cast<cl_mem>(srcPtr2),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        phase_hip_batch(
            static_cast<Rpp8u*>(srcPtr1),
            static_cast<Rpp8u*>(srcPtr2),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_phase_u8_pln1_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    phase_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr1),
        static_cast<Rpp8u*>(srcPtr2),
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
rppi_phase_u8_pln3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    phase_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr1),
        static_cast<Rpp8u*>(srcPtr2),
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
rppi_phase_u8_pkd3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    phase_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr1),
        static_cast<Rpp8u*>(srcPtr2),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr),
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PACKED, 3
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_magnitude_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
        magnitude_cl_batch(
            static_cast<cl_mem>(srcPtr1),
            static_cast<cl_mem>(srcPtr2),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#elif defined (HIP_COMPILE)
    {
        magnitude_hip_batch(
            static_cast<Rpp8u*>(srcPtr1),
            static_cast<Rpp8u*>(srcPtr2),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_magnitude_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
        magnitude_cl_batch(
            static_cast<cl_mem>(srcPtr1),
            static_cast<cl_mem>(srcPtr2),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        magnitude_hip_batch(
            static_cast<Rpp8u*>(srcPtr1),
            static_cast<Rpp8u*>(srcPtr2),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_magnitude_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
        magnitude_cl_batch(
            static_cast<cl_mem>(srcPtr1),
            static_cast<cl_mem>(srcPtr2),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        magnitude_hip_batch(
            static_cast<Rpp8u*>(srcPtr1),
            static_cast<Rpp8u*>(srcPtr2),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_magnitude_u8_pln1_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    magnitude_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr1),
        static_cast<Rpp8u*>(srcPtr2),
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
rppi_magnitude_u8_pln3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    magnitude_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr1),
        static_cast<Rpp8u*>(srcPtr2),
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
rppi_magnitude_u8_pkd3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    magnitude_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr1),
        static_cast<Rpp8u*>(srcPtr2),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        static_cast<Rpp8u*>(dstPtr),
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PACKED, 3
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_accumulate_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
        accumulate_cl_batch(
            static_cast<cl_mem>(srcPtr1),
            static_cast<cl_mem>(srcPtr2),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#elif defined (HIP_COMPILE)
    {
        accumulate_hip_batch(
            static_cast<Rpp8u*>(srcPtr1),
            static_cast<Rpp8u*>(srcPtr2),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_accumulate_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
        accumulate_cl_batch(
            static_cast<cl_mem>(srcPtr1),
            static_cast<cl_mem>(srcPtr2),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        accumulate_hip_batch(
            static_cast<Rpp8u*>(srcPtr1),
            static_cast<Rpp8u*>(srcPtr2),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_accumulate_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
        accumulate_cl_batch(
            static_cast<cl_mem>(srcPtr1),
            static_cast<cl_mem>(srcPtr2),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        accumulate_hip_batch(
            static_cast<Rpp8u*>(srcPtr1),
            static_cast<Rpp8u*>(srcPtr2),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_accumulate_u8_pln1_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    accumulate_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr1),
        static_cast<Rpp8u*>(srcPtr2),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 1
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_accumulate_u8_pln3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    accumulate_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr1),
        static_cast<Rpp8u*>(srcPtr2),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 3
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_accumulate_u8_pkd3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    accumulate_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr1),
        static_cast<Rpp8u*>(srcPtr2),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PACKED, 3
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_accumulate_weighted_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,Rpp32f *alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
    copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        accumulate_weighted_cl_batch(
            static_cast<cl_mem>(srcPtr1),
            static_cast<cl_mem>(srcPtr2),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#elif defined (HIP_COMPILE)
    {
        accumulate_weighted_hip_batch(
            static_cast<Rpp8u*>(srcPtr1),
            static_cast<Rpp8u*>(srcPtr2),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_accumulate_weighted_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,Rpp32f *alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
    copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        accumulate_weighted_cl_batch(
            static_cast<cl_mem>(srcPtr1),
            static_cast<cl_mem>(srcPtr2),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        accumulate_weighted_hip_batch(
            static_cast<Rpp8u*>(srcPtr1),
            static_cast<Rpp8u*>(srcPtr2),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_accumulate_weighted_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,Rpp32f *alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
    copy_param_float (alpha, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        accumulate_weighted_cl_batch(
            static_cast<cl_mem>(srcPtr1),
            static_cast<cl_mem>(srcPtr2),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        accumulate_weighted_hip_batch(
            static_cast<Rpp8u*>(srcPtr1),
            static_cast<Rpp8u*>(srcPtr2),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_accumulate_weighted_u8_pln1_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,Rpp32f *alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    accumulate_weighted_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr1),
        static_cast<Rpp8u*>(srcPtr2),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        alpha,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 1
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_accumulate_weighted_u8_pln3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,Rpp32f *alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    accumulate_weighted_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr1),
        static_cast<Rpp8u*>(srcPtr2),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        alpha,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 3
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_accumulate_weighted_u8_pkd3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,Rpp32f *alpha ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    accumulate_weighted_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr1),
        static_cast<Rpp8u*>(srcPtr2),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        alpha,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PACKED, 3
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_accumulate_squared_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
        accumulate_squared_cl_batch(
            static_cast<cl_mem>(srcPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#elif defined (HIP_COMPILE)
    {
        accumulate_squared_hip_batch(
            static_cast<Rpp8u*>(srcPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 1
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_accumulate_squared_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
        accumulate_squared_cl_batch(
            static_cast<cl_mem>(srcPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        accumulate_squared_hip_batch(
            static_cast<Rpp8u*>(srcPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PLANAR, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_accumulate_squared_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
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
        accumulate_squared_cl_batch(
            static_cast<cl_mem>(srcPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#elif defined (HIP_COMPILE)
    {
        accumulate_squared_hip_batch(
            static_cast<Rpp8u*>(srcPtr),
            rpp::deref(rppHandle),
            RPPI_CHN_PACKED, 3
        );
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus  
rppi_accumulate_squared_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    accumulate_squared_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 1
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_accumulate_squared_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    accumulate_squared_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PLANAR, 3
    );

    return RPP_SUCCESS;
}

RppStatus  
rppi_accumulate_squared_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
    Rpp32u paramIndex = 0;
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    accumulate_squared_host_batch<Rpp8u>(
        static_cast<Rpp8u*>(srcPtr),
        srcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
        rpp::deref(rppHandle).GetBatchSize(),
        RPPI_CHN_PACKED, 3
    );

    return RPP_SUCCESS;
}

// ----------------------------------------
// Host tensor functions calls 
// ----------------------------------------


RppStatus
rppi_tensor_add_u8_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t dstPtr, 
                        Rpp32u tensorDimension, RppPtr_t tensorDimensionValues ,rppHandle_t rppHandle)
{
    tensor_add_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), static_cast<Rpp8u*>(dstPtr), 
                           tensorDimension, static_cast<Rpp32u*>(tensorDimensionValues));

    return RPP_SUCCESS;

}

RppStatus
rppi_tensor_subtract_u8_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t dstPtr, 
                        Rpp32u tensorDimension, RppPtr_t tensorDimensionValues ,rppHandle_t rppHandle)
{
    tensor_subtract_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), static_cast<Rpp8u*>(dstPtr), 
                           tensorDimension, static_cast<Rpp32u*>(tensorDimensionValues));

    return RPP_SUCCESS;

}

RppStatus
rppi_tensor_multiply_u8_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t dstPtr, 
                        Rpp32u tensorDimension, RppPtr_t tensorDimensionValues ,rppHandle_t rppHandle)
{
    tensor_multiply_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), static_cast<Rpp8u*>(dstPtr), 
                           tensorDimension, static_cast<Rpp32u*>(tensorDimensionValues));

    return RPP_SUCCESS;

}


RppStatus
rppi_tensor_matrix_multiply_u8_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t dstPtr, 
                                    RppPtr_t tensorDimensionValues1, RppPtr_t tensorDimensionValues2 ,rppHandle_t rppHandle)
{
    tensor_matrix_multiply_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), static_cast<Rpp8u*>(dstPtr), 
                                       static_cast<Rpp32u*>(tensorDimensionValues1), static_cast<Rpp32u*>(tensorDimensionValues2));

    return RPP_SUCCESS;

}
//----------------------------------------
//GPU Tensor functions calls 
//----------------------------------------

RppStatus
rppi_tensor_add_u8_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t dstPtr, Rpp32u tensorDimension, RppPtr_t tensorDimensionValues, rppHandle_t rppHandle)
{

#ifdef OCL_COMPILE
      {
      tensor_add_cl(tensorDimension,
            static_cast<Rpp32u*>(tensorDimensionValues),
            static_cast<cl_mem>(srcPtr1), 
            static_cast<cl_mem>(srcPtr2), 
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle));
      } 
#elif defined (HIP_COMPILE) 
      {
      tensor_add_hip(tensorDimension,
            static_cast<Rpp32u*>(tensorDimensionValues),
            static_cast<Rpp8u*>(srcPtr1), 
            static_cast<Rpp8u*>(srcPtr2), 
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle));
      } 
#endif //BACKEND 
        return RPP_SUCCESS;
}

RppStatus
rppi_tensor_subtract_u8_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t dstPtr, Rpp32u tensorDimension, RppPtr_t tensorDimensionValues, rppHandle_t rppHandle)
{

#ifdef OCL_COMPILE
      {
      tensor_subtract_cl(tensorDimension, 
            static_cast<Rpp32u*>(tensorDimensionValues),
            static_cast<cl_mem>(srcPtr1), 
            static_cast<cl_mem>(srcPtr2), 
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle)); 	
    } 
#elif defined (HIP_COMPILE) 
      {
      tensor_subtract_hip(tensorDimension,
            static_cast<Rpp32u*>(tensorDimensionValues),
            static_cast<Rpp8u*>(srcPtr1), 
            static_cast<Rpp8u*>(srcPtr2), 
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle));
      } 
#endif //BACKEND 
        return RPP_SUCCESS;
}
RppStatus
rppi_tensor_multiply_u8_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t dstPtr, Rpp32u tensorDimension, RppPtr_t tensorDimensionValues, rppHandle_t rppHandle)
{

#ifdef OCL_COMPILE
      {
      tensor_multiply_cl(tensorDimension, 
            static_cast<Rpp32u*>(tensorDimensionValues),
            static_cast<cl_mem>(srcPtr1), 
            static_cast<cl_mem>(srcPtr2), 
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle)); 	 
      } 
#elif defined (HIP_COMPILE) 
      {
      tensor_multiply_hip(tensorDimension,
            static_cast<Rpp32u*>(tensorDimensionValues),
            static_cast<Rpp8u*>(srcPtr1), 
            static_cast<Rpp8u*>(srcPtr2), 
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle));
      } 
#endif //BACKEND 
        return RPP_SUCCESS;
}

RppStatus
rppi_tensor_matrix_multiply_u8_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t dstPtr, RppPtr_t tensorDimensionValues1, RppPtr_t tensorDimensionValues2, rppHandle_t rppHandle)
{
#ifdef OCL_COMPILE
      {
      tensor_matrix_multiply_cl(static_cast<cl_mem>(srcPtr1), 
            static_cast<cl_mem>(srcPtr2), 
            static_cast<Rpp32u*>(tensorDimensionValues1),
            static_cast<Rpp32u*>(tensorDimensionValues2),
            static_cast<cl_mem>(dstPtr),
            rpp::deref(rppHandle));
      } 
#elif defined (HIP_COMPILE) 
      {
      tensor_matrix_multiply_hip(static_cast<Rpp8u*>(srcPtr1), 
            static_cast<Rpp8u*>(srcPtr2), 
            static_cast<Rpp32u*>(tensorDimensionValues1),
            static_cast<Rpp32u*>(tensorDimensionValues2),
            static_cast<Rpp8u*>(dstPtr),
            rpp::deref(rppHandle));
      } 
#endif //BACKEND 
        return RPP_SUCCESS;
}


// RppStatus
// rppi_tensor_table_look_up_u8_gpu(RppPtr_t srcPtr, RppPtr_t dstPtr, Rpp8u *look_up_table, Rpp32u tensorDimension, RppPtr_t tensorDimensionValues, rppHandle_t rppHandle)
// {
// #ifdef OCL_COMPILE
//  	 {
//  	 tensor_table_look_up_cl( 
//             static_cast<cl_mem>(srcPtr), 
// 			static_cast<cl_mem>(dstPtr),
// 			look_up_table, 
// 			tensorDimension,
// 			static_cast<Rpp32u *>(tensorDimensionValues),
// 			rpp::deref(rppHandle)); 	
// 	  }
// #endif //BACKEND 
// 		return RPP_SUCCESS;
// } 

RppStatus
rppi_tensor_transpose_u8_gpu(RppPtr_t srcPtr, RppPtr_t dstPtr, RppPtr_t in_tensor_dims, RppPtr_t perm, rppHandle_t rppHandle)
{
#ifdef OCL_COMPILE
      {
        tensor_transpose_cl(static_cast<cl_mem>(srcPtr), static_cast<cl_mem>(dstPtr), 
         static_cast<Rpp32u*>(in_tensor_dims), static_cast<Rpp32u*>(perm), RPPTensorDataType::U8, rpp::deref(rppHandle));
      } 
#elif defined (HIP_COMPILE) 
      {
        tensor_transpose_hip_u8(static_cast<Rpp8u*>(srcPtr), static_cast<Rpp8u*>(dstPtr), 
         static_cast<Rpp32u*>(in_tensor_dims), static_cast<Rpp32u*>(perm), rpp::deref(rppHandle));
      } 
#endif //BACKEND 
    return RPP_SUCCESS;
}

RppStatus
rppi_tensor_transpose_f16_gpu(RppPtr_t srcPtr, RppPtr_t dstPtr, RppPtr_t in_tensor_dims, RppPtr_t perm, rppHandle_t rppHandle)
{
#ifdef OCL_COMPILE
      {
        tensor_transpose_cl(static_cast<cl_mem>(srcPtr), static_cast<cl_mem>(dstPtr), 
         static_cast<Rpp32u*>(in_tensor_dims), static_cast<Rpp32u*>(perm), RPPTensorDataType::FP16, rpp::deref(rppHandle));
      } 
#elif defined (HIP_COMPILE) 
      {
        tensor_transpose_hip_fp16(static_cast<Rpp16f*>(srcPtr), static_cast<Rpp16f*>(dstPtr), 
         static_cast<Rpp32u*>(in_tensor_dims), static_cast<Rpp32u*>(perm), rpp::deref(rppHandle));
      } 
#endif //BACKEND 
    return RPP_SUCCESS;
}

RppStatus
rppi_tensor_transpose_f32_gpu(RppPtr_t srcPtr, RppPtr_t dstPtr, RppPtr_t in_tensor_dims, RppPtr_t perm, rppHandle_t rppHandle)
{
#ifdef OCL_COMPILE
      {
        tensor_transpose_cl(static_cast<cl_mem>(srcPtr), static_cast<cl_mem>(dstPtr), 
         static_cast<Rpp32u*>(in_tensor_dims), static_cast<Rpp32u*>(perm), RPPTensorDataType::FP32, rpp::deref(rppHandle));
      } 
#elif defined (HIP_COMPILE) 
      {
        tensor_transpose_hip_fp32(static_cast<Rpp32f*>(srcPtr), static_cast<Rpp32f*>(dstPtr), 
         static_cast<Rpp32u*>(in_tensor_dims), static_cast<Rpp32u*>(perm), rpp::deref(rppHandle));
      } 
#endif //BACKEND 
    return RPP_SUCCESS;
}

RppStatus
rppi_tensor_transpose_i8_gpu(RppPtr_t srcPtr, RppPtr_t dstPtr, RppPtr_t in_tensor_dims, RppPtr_t perm, rppHandle_t rppHandle)
{
#ifdef OCL_COMPILE
      {
        tensor_transpose_cl(static_cast<cl_mem>(srcPtr), static_cast<cl_mem>(dstPtr), 
         static_cast<Rpp32u*>(in_tensor_dims), static_cast<Rpp32u*>(perm), RPPTensorDataType::I8, rpp::deref(rppHandle));
      } 
#elif defined (HIP_COMPILE) 
      {
        tensor_transpose_hip_i8(static_cast<Rpp8s*>(srcPtr), static_cast<Rpp8s*>(dstPtr), 
         static_cast<Rpp32u*>(in_tensor_dims), static_cast<Rpp32u*>(perm), rpp::deref(rppHandle));
      } 
#endif //BACKEND 
    return RPP_SUCCESS;
}
