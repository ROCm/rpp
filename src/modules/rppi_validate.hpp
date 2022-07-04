#ifndef RPPI_VALIDATE_OPERATIONS_FUNCTIONS
#define RPPI_VALIDATE_OPERATIONS_FUNCTIONS
#include <iostream>
#include <stdlib.h>
#include <rpp.h>
#include <rppdefs.h>
#include <hip/rpp/handle.hpp>

#ifdef OCL_COMPILE
#include <CL/cl.h>
#elif defined (HIP_COMPILE)
#include <hip/hip_runtime_api.h>
#endif

inline RppLayoutParams get_layout_params(RpptLayout layout, Rpp32u channels)
{
    RppLayoutParams layoutParams;
    if(layout == RpptLayout::NCHW)
    {
        if (channels == 1) // PLN1
        {
            layoutParams.channelParam = 1;
            layoutParams.bufferMultiplier = 1;
        }
        else if (channels == 3) // PLN3
        {
            layoutParams.channelParam = 3;
            layoutParams.bufferMultiplier = 1;
        }
    }
    else if(layout == RpptLayout::NHWC)
    {
        if (channels == 3) // PKD3
        {
            layoutParams.channelParam = 1;
            layoutParams.bufferMultiplier = 3;
        }
    }

    return layoutParams;
}

inline void copy_srcSize(RppiSize srcSize, rpp::Handle& handle)
{
    for(int i = 0; i < handle.GetBatchSize() ; i++)
    {
           handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] = srcSize.height;
           handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] = srcSize.width;
    }
#ifdef OCL_COMPILE
    {
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.srcSize.height, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.csrcSize.height, 0, NULL, NULL);
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.srcSize.width, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.csrcSize.width, 0, NULL, NULL);
    }
#elif defined(HIP_COMPILE)
    {
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.srcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.height, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.srcSize.width, handle.GetInitHandle()->mem.mgpu.csrcSize.width, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    }
#endif
}

inline void copy_srcSize(RppiSize *srcSize, rpp::Handle& handle)
{
    for(int i = 0; i < handle.GetBatchSize() ; i++)
    {
           handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] = srcSize[i].height;
           handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] = srcSize[i].width;
    }
#ifdef OCL_COMPILE
    {
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.srcSize.height, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.csrcSize.height, 0, NULL, NULL);
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.srcSize.width, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.csrcSize.width, 0, NULL, NULL);
    }
#elif defined(HIP_COMPILE)
    {
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.srcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.height, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.srcSize.width, handle.GetInitHandle()->mem.mgpu.csrcSize.width, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    }
#endif
}

inline void copy_dstSize(RppiSize *dstSize, rpp::Handle& handle)
{
    for(int i = 0; i < handle.GetBatchSize() ; i++)
    {
           handle.GetInitHandle()->mem.mgpu.cdstSize.height[i] = dstSize[i].height;
           handle.GetInitHandle()->mem.mgpu.cdstSize.width[i] = dstSize[i].width;
    }
#ifdef OCL_COMPILE
    {
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.dstSize.height, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.cdstSize.height, 0, NULL, NULL);
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.dstSize.width, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.cdstSize.width, 0, NULL, NULL);
    }
#elif defined(HIP_COMPILE)
    {
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.dstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.height, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.dstSize.width, handle.GetInitHandle()->mem.mgpu.cdstSize.width, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    }
#endif
}

inline void copy_host_srcSize(RppiSize srcSize, rpp::Handle& handle)
{
    for(int i = 0; i < handle.GetBatchSize() ; i++)
    {
           handle.GetInitHandle()->mem.mcpu.srcSize[i].height = srcSize.height;
           handle.GetInitHandle()->mem.mcpu.srcSize[i].width = srcSize.width;
    }
}

inline void copy_host_dstSize(RppiSize dstSize, rpp::Handle& handle)
{
    for(int i = 0; i < handle.GetBatchSize() ; i++)
    {
           handle.GetInitHandle()->mem.mcpu.dstSize[i].height = dstSize.height;
           handle.GetInitHandle()->mem.mcpu.dstSize[i].width = dstSize.width;
    }
}

inline void copy_host_maxSrcSize(RppiSize maxSrcSize, rpp::Handle& handle)
{
    for(int i = 0; i < handle.GetBatchSize() ; i++)
    {
           handle.GetInitHandle()->mem.mcpu.maxSrcSize[i].height = maxSrcSize.height;
           handle.GetInitHandle()->mem.mcpu.maxSrcSize[i].width = maxSrcSize.width;
    }
}

inline void copy_host_maxDstSize(RppiSize maxDstSize, rpp::Handle& handle)
{
    for(int i = 0; i < handle.GetBatchSize() ; i++)
    {
           handle.GetInitHandle()->mem.mcpu.maxDstSize[i].height = maxDstSize.height;
           handle.GetInitHandle()->mem.mcpu.maxDstSize[i].width = maxDstSize.width;
    }
}

inline void copy_dstSize(RppiSize dstSize, rpp::Handle& handle)
{
    for(int i = 0; i < handle.GetBatchSize() ; i++)
    {
           handle.GetInitHandle()->mem.mgpu.cdstSize.height[i] = dstSize.height;
           handle.GetInitHandle()->mem.mgpu.cdstSize.width[i] = dstSize.width;
    }
#ifdef OCL_COMPILE
    {
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.dstSize.height, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.cdstSize.height, 0, NULL, NULL);
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.dstSize.width, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.cdstSize.width, 0, NULL, NULL);
    }
#elif defined(HIP_COMPILE)
    {
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.dstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.height, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.dstSize.width, handle.GetInitHandle()->mem.mgpu.cdstSize.width, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);

    }
#endif
}

inline void copy_host_roi(RppiROI roiPoints, rpp::Handle& handle)
{
    for(int i = 0; i < handle.GetBatchSize(); i++)
    {
        handle.GetInitHandle()->mem.mcpu.roiPoints[i].roiHeight = roiPoints.roiHeight;
        handle.GetInitHandle()->mem.mcpu.roiPoints[i].roiWidth = roiPoints.roiWidth;
        handle.GetInitHandle()->mem.mcpu.roiPoints[i].x = roiPoints.x;
        handle.GetInitHandle()->mem.mcpu.roiPoints[i].y = roiPoints.y;
    }
}

inline void copy_host_roi(RppiROI *roiPoints, rpp::Handle& handle)
{
    for(int i = 0; i < handle.GetBatchSize(); i++)
    {
        handle.GetInitHandle()->mem.mcpu.roiPoints[i].roiHeight = roiPoints[i].roiHeight;
        handle.GetInitHandle()->mem.mcpu.roiPoints[i].roiWidth = roiPoints[i].roiWidth;
        handle.GetInitHandle()->mem.mcpu.roiPoints[i].x = roiPoints[i].x;
        handle.GetInitHandle()->mem.mcpu.roiPoints[i].y = roiPoints[i].y;
    }
}

inline void copy_roi(RppiROI roiPoints, rpp::Handle& handle)
{
    for(int i = 0; i < handle.GetBatchSize(); i++)
    {
#if defined(OCL_COMPILE) || defined (HIP_COMPILE)
        {
            if(roiPoints.roiHeight == 0 && roiPoints.roiWidth == 0)
            {
                handle.GetInitHandle()->mem.mgpu.croiPoints.roiHeight[i] = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
                handle.GetInitHandle()->mem.mgpu.croiPoints.roiWidth[i] = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
            }
            else
            {
                handle.GetInitHandle()->mem.mgpu.croiPoints.roiHeight[i] = roiPoints.roiHeight + roiPoints.y;
                handle.GetInitHandle()->mem.mgpu.croiPoints.roiWidth[i] = roiPoints.roiWidth + roiPoints.x;
            }
        }
#else
        {
            handle.GetInitHandle()->mem.mgpu.croiPoints.roiHeight[i] = roiPoints.roiHeight;
            handle.GetInitHandle()->mem.mgpu.croiPoints.roiWidth[i] = roiPoints.roiWidth;
        }
#endif
        handle.GetInitHandle()->mem.mgpu.croiPoints.x[i] = roiPoints.x;
        handle.GetInitHandle()->mem.mgpu.croiPoints.y[i] = roiPoints.y;
    }
#ifdef OCL_COMPILE
    {
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.croiPoints.roiHeight, 0, NULL, NULL);
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.croiPoints.roiWidth, 0, NULL, NULL);
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.roiPoints.x, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.croiPoints.x, 0, NULL, NULL);
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.roiPoints.y, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.croiPoints.y, 0, NULL, NULL);
    }
#elif defined(HIP_COMPILE)
    {
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight, handle.GetInitHandle()->mem.mgpu.croiPoints.roiHeight, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth, handle.GetInitHandle()->mem.mgpu.croiPoints.roiWidth, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.roiPoints.x, handle.GetInitHandle()->mem.mgpu.croiPoints.x, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.roiPoints.y, handle.GetInitHandle()->mem.mgpu.croiPoints.y, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);

    }
#endif
}

inline void copy_roi(RppiROI *roiPoints, rpp::Handle& handle)
{
    for(int i = 0; i < handle.GetBatchSize(); i++)
    {
#if defined(OCL_COMPILE) || defined (HIP_COMPILE)
        {
            if(roiPoints[i].roiHeight == 0 && roiPoints[i].roiWidth == 0)
            {
                handle.GetInitHandle()->mem.mgpu.croiPoints.roiHeight[i] = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
                handle.GetInitHandle()->mem.mgpu.croiPoints.roiWidth[i] = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
            }
            else
            {
                handle.GetInitHandle()->mem.mgpu.croiPoints.roiHeight[i] = roiPoints[i].roiHeight + roiPoints[i].y;
                handle.GetInitHandle()->mem.mgpu.croiPoints.roiWidth[i] = roiPoints[i].roiWidth + roiPoints[i].x;
            }
        }
#else
        {
            handle.GetInitHandle()->mem.mgpu.croiPoints.roiHeight[i] = roiPoints[i].roiHeight;
            handle.GetInitHandle()->mem.mgpu.croiPoints.roiWidth[i] = roiPoints[i].roiWidth;
        }
#endif
        handle.GetInitHandle()->mem.mgpu.croiPoints.x[i] = roiPoints[i].x;
        handle.GetInitHandle()->mem.mgpu.croiPoints.y[i] = roiPoints[i].y;
    }
#ifdef OCL_COMPILE
    {
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.croiPoints.roiHeight, 0, NULL, NULL);
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.croiPoints.roiWidth, 0, NULL, NULL);
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.roiPoints.x, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.croiPoints.x, 0, NULL, NULL);
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.roiPoints.y, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.croiPoints.y, 0, NULL, NULL);
    }
#elif defined(HIP_COMPILE)
    {
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight, handle.GetInitHandle()->mem.mgpu.croiPoints.roiHeight, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth, handle.GetInitHandle()->mem.mgpu.croiPoints.roiWidth, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.roiPoints.x, handle.GetInitHandle()->mem.mgpu.croiPoints.x, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.roiPoints.y, handle.GetInitHandle()->mem.mgpu.croiPoints.y, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    }
#endif
}

inline void copy_param_float(float param, rpp::Handle& handle, Rpp32u paramIndex)
{
    for(int i = 0; i < handle.GetBatchSize() ; i++)
    {
        handle.GetInitHandle()->mem.mcpu.floatArr[paramIndex].floatmem[i] = param;
    }
#ifdef OCL_COMPILE
    {
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.floatArr[paramIndex].floatmem, CL_FALSE, 0, sizeof(Rpp32f) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mcpu.floatArr[paramIndex].floatmem, 0, NULL, NULL);
    }
#elif defined(HIP_COMPILE)
    {
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.floatArr[paramIndex].floatmem, handle.GetInitHandle()->mem.mcpu.floatArr[paramIndex].floatmem, sizeof(Rpp32f) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    }
#endif
}

inline void copy_param_float(float *param, rpp::Handle& handle, Rpp32u paramIndex)
{
    for(int i = 0; i < handle.GetBatchSize() ; i++)
    {
        handle.GetInitHandle()->mem.mcpu.floatArr[paramIndex].floatmem[i] = param[i];
    }
#ifdef OCL_COMPILE
    {
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.floatArr[paramIndex].floatmem, CL_FALSE, 0, sizeof(Rpp32f) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mcpu.floatArr[paramIndex].floatmem, 0, NULL, NULL);
    }
#elif defined(HIP_COMPILE)
    {
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.floatArr[paramIndex].floatmem, handle.GetInitHandle()->mem.mcpu.floatArr[paramIndex].floatmem, sizeof(Rpp32f) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    }
#endif
}

inline void copy_param_float3(float *param, rpp::Handle& handle, Rpp32u paramIndex)
{
#ifdef HIP_COMPILE
    {
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.float3Arr[paramIndex].floatmem, param, sizeof(Rpp32f) * handle.GetBatchSize() * 3, hipMemcpyHostToDevice);
    }
#endif
}

inline void copy_param_uint(uint param, rpp::Handle& handle, Rpp32u paramIndex)
{
    for(int i = 0; i < handle.GetBatchSize() ; i++)
    {
        handle.GetInitHandle()->mem.mcpu.uintArr[paramIndex].uintmem[i] = param;
    }
#ifdef OCL_COMPILE
    {
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.uintArr[paramIndex].uintmem, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mcpu.uintArr[paramIndex].uintmem, 0, NULL, NULL);
    }
#elif defined(HIP_COMPILE)
    {
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.uintArr[paramIndex].uintmem, handle.GetInitHandle()->mem.mcpu.uintArr[paramIndex].uintmem, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    }
#endif
}

inline void copy_param_uint(uint *param, rpp::Handle& handle, Rpp32u paramIndex)
{
    for(int i = 0; i < handle.GetBatchSize() ; i++)
    {
        handle.GetInitHandle()->mem.mcpu.uintArr[paramIndex].uintmem[i] = param[i];
    }
#ifdef OCL_COMPILE
    {
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.uintArr[paramIndex].uintmem, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mcpu.uintArr[paramIndex].uintmem, 0, NULL, NULL);
    }
#elif defined(HIP_COMPILE)
    {
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.uintArr[paramIndex].uintmem, handle.GetInitHandle()->mem.mcpu.uintArr[paramIndex].uintmem, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    }
#endif
}

inline void copy_param_int(int param, rpp::Handle& handle, Rpp32u paramIndex)
{
    for(int i = 0; i < handle.GetBatchSize() ; i++)
    {
        handle.GetInitHandle()->mem.mcpu.intArr[paramIndex].intmem[i] = param;
    }
#ifdef OCL_COMPILE
    {
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.intArr[paramIndex].intmem, CL_FALSE, 0, sizeof(Rpp32s) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mcpu.intArr[paramIndex].intmem, 0, NULL, NULL);
    }
#elif defined(HIP_COMPILE)
    {
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.intArr[paramIndex].intmem, handle.GetInitHandle()->mem.mcpu.intArr[paramIndex].intmem, sizeof(Rpp32s) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    }
#endif
}

inline void copy_param_int(int *param, rpp::Handle& handle, Rpp32u paramIndex)
{
    for(int i = 0; i < handle.GetBatchSize() ; i++)
    {
        handle.GetInitHandle()->mem.mcpu.intArr[paramIndex].intmem[i] = param[i];
    }
#ifdef OCL_COMPILE
    {
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.intArr[paramIndex].intmem, CL_FALSE, 0, sizeof(Rpp32s) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mcpu.intArr[paramIndex].intmem, 0, NULL, NULL);
    }
#elif defined(HIP_COMPILE)
    {
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.intArr[paramIndex].intmem, handle.GetInitHandle()->mem.mcpu.intArr[paramIndex].intmem, sizeof(Rpp32s) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    }
#endif
}

inline void copy_param_uchar(Rpp8u param, rpp::Handle& handle, Rpp32u paramIndex)
{
    for(int i = 0; i < handle.GetBatchSize() ; i++)
    {
        handle.GetInitHandle()->mem.mcpu.ucharArr[paramIndex].ucharmem[i] = param;
    }
#ifdef OCL_COMPILE
    {
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.ucharArr[paramIndex].ucharmem, CL_FALSE, 0, sizeof(Rpp8u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mcpu.ucharArr[paramIndex].ucharmem, 0, NULL, NULL);
    }
#elif defined(HIP_COMPILE)
    {
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.ucharArr[paramIndex].ucharmem, handle.GetInitHandle()->mem.mcpu.ucharArr[paramIndex].ucharmem, sizeof(Rpp8u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    }
#endif
}

inline void copy_param_uchar(Rpp8u *param, rpp::Handle& handle, Rpp32u paramIndex)
{
    for(int i = 0; i < handle.GetBatchSize() ; i++)
    {
        handle.GetInitHandle()->mem.mcpu.ucharArr[paramIndex].ucharmem[i] = param[i];
    }
#ifdef OCL_COMPILE
    {
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.ucharArr[paramIndex].ucharmem, CL_FALSE, 0, sizeof(Rpp8u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mcpu.ucharArr[paramIndex].ucharmem, 0, NULL, NULL);
    }
#elif defined(HIP_COMPILE)
    {
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.ucharArr[paramIndex].ucharmem, handle.GetInitHandle()->mem.mcpu.ucharArr[paramIndex].ucharmem, sizeof(Rpp8u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    }
#endif
}

inline void copy_param_char(char param, rpp::Handle& handle, Rpp32u paramIndex)
{
    for(int i = 0; i < handle.GetBatchSize() ; i++)
    {
        handle.GetInitHandle()->mem.mcpu.charArr[paramIndex].charmem[i] = param;
    }
#ifdef OCL_COMPILE
    {
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.charArr[paramIndex].charmem, CL_FALSE, 0, sizeof(Rpp8s) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mcpu.charArr[paramIndex].charmem, 0, NULL, NULL);
    }
#elif defined(HIP_COMPILE)
    {
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.charArr[paramIndex].charmem, handle.GetInitHandle()->mem.mcpu.charArr[paramIndex].charmem, sizeof(Rpp8s) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    }
#endif
}

inline void copy_param_char(char *param, rpp::Handle& handle, Rpp32u paramIndex)
{
    for(int i = 0; i < handle.GetBatchSize() ; i++)
    {
        handle.GetInitHandle()->mem.mcpu.charArr[paramIndex].charmem[i] = param[i];
    }
#ifdef OCL_COMPILE
    {
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.charArr[paramIndex].charmem, CL_FALSE, 0, sizeof(Rpp8s) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mcpu.charArr[paramIndex].charmem, 0, NULL, NULL);
    }
#elif defined(HIP_COMPILE)
    {
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.charArr[paramIndex].charmem, handle.GetInitHandle()->mem.mcpu.charArr[paramIndex].charmem, sizeof(Rpp8s) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    }
#endif
}

inline void copy_param_RpptRGB(RpptRGB *param, rpp::Handle& handle)
{
    for(int i = 0; i < handle.GetBatchSize() ; i++)
    {
        handle.GetInitHandle()->mem.mcpu.rgbArr.rgbmem[i] = param[i];
    }
#ifdef OCL_COMPILE

#elif defined(HIP_COMPILE)
    {
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.rgbArr.rgbmem, handle.GetInitHandle()->mem.mcpu.rgbArr.rgbmem, sizeof(RpptRGB) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    }
#endif
}

inline void copy_srcMaxSize(rpp::Handle& handle)
{
    for(int i = 0; i < handle.GetBatchSize(); i++)
    {
        handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.height[i] = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.width[i] = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
    }
#ifdef OCL_COMPILE
    {
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.maxSrcSize.height, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.height, 0, NULL, NULL);
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.maxSrcSize.width, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.width, 0, NULL, NULL);
    }
#elif defined(HIP_COMPILE)
    {
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.maxSrcSize.height, handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.height, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.maxSrcSize.width, handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.width, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    }
#endif
}

inline void copy_dstMaxSize(rpp::Handle& handle)
{
    for(int i = 0; i < handle.GetBatchSize(); i++)
    {
        handle.GetInitHandle()->mem.mgpu.cmaxDstSize.height[i] = handle.GetInitHandle()->mem.mgpu.cdstSize.height[i];
        handle.GetInitHandle()->mem.mgpu.cmaxDstSize.width[i] = handle.GetInitHandle()->mem.mgpu.cdstSize.width[i];
    }
#ifdef OCL_COMPILE
    {
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.maxDstSize.height, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.cmaxDstSize.height, 0, NULL, NULL);
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.maxDstSize.width, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.cmaxDstSize.width, 0, NULL, NULL);
    }
#elif defined(HIP_COMPILE)
    {
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.maxDstSize.height, handle.GetInitHandle()->mem.mgpu.cmaxDstSize.height, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.maxDstSize.width, handle.GetInitHandle()->mem.mgpu.cmaxDstSize.width, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    }
#endif
}

inline void copy_srcMaxSize(RppiSize maxSrcSize, rpp::Handle& handle)
{
    for(int i = 0; i < handle.GetBatchSize(); i++)
    {
        handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.height[i] = maxSrcSize.height;
        handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.width[i] = maxSrcSize.width;
    }
#ifdef OCL_COMPILE
    {
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.maxSrcSize.height, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.height, 0, NULL, NULL);
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.maxSrcSize.width, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.width, 0, NULL, NULL);
    }
#elif defined(HIP_COMPILE)
    {
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.maxSrcSize.height, handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.height, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.maxSrcSize.width, handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.width, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    }
#endif
}

inline void copy_dstMaxSize(RppiSize maxDstSize, rpp::Handle& handle)
{
    for(int i = 0; i < handle.GetBatchSize(); i++)
    {
        handle.GetInitHandle()->mem.mgpu.cmaxDstSize.height[i] = maxDstSize.height;
        handle.GetInitHandle()->mem.mgpu.cmaxDstSize.width[i] = maxDstSize.width;
    }
#ifdef OCL_COMPILE
    {
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.maxDstSize.height, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.cmaxDstSize.height, 0, NULL, NULL);
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.maxDstSize.width, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.cmaxDstSize.width, 0, NULL, NULL);
    }
#elif defined(HIP_COMPILE)
    {
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.maxDstSize.height, handle.GetInitHandle()->mem.mgpu.cmaxDstSize.height, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.maxDstSize.width, handle.GetInitHandle()->mem.mgpu.cmaxDstSize.width, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    }
#endif
}

inline void get_srcBatchIndex(rpp::Handle& handle, unsigned int channel, RppiChnFormat chnFormat, bool is_padded = true)
{
    int i;
    handle.GetInitHandle()->mem.mcpu.srcBatchIndex[0] = 0;
    for(i =0; i < handle.GetBatchSize() - 1 ; i++)
    {
        handle.GetInitHandle()->mem.mcpu.srcBatchIndex[i+1] = handle.GetInitHandle()->mem.mcpu.srcBatchIndex[i] + handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.width[i] * channel;
    }
    for(i =0; i < handle.GetBatchSize() ; i++)
    {
        if(chnFormat != RPPI_CHN_PLANAR)
        {
            handle.GetInitHandle()->mem.mcpu.inc[i] = 1;
        }
        else
        {
            if(!is_padded)
                handle.GetInitHandle()->mem.mcpu.inc[i] = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
            else
                handle.GetInitHandle()->mem.mcpu.inc[i] = handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.width[i];
        }
    }
#ifdef OCL_COMPILE
    {

        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.srcBatchIndex, CL_FALSE, 0, sizeof(Rpp64u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mcpu.srcBatchIndex, 0, NULL, NULL);
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.inc, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mcpu.inc, 0, NULL, NULL);
    }
#elif defined (HIP_COMPILE)
    {
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.srcBatchIndex, handle.GetInitHandle()->mem.mcpu.srcBatchIndex, sizeof(Rpp64u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.inc, handle.GetInitHandle()->mem.mcpu.inc, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    }
#endif
}

inline void get_dstBatchIndex(rpp::Handle& handle, unsigned int channel, RppiChnFormat chnFormat, bool is_padded = true)
{
    int i;
    handle.GetInitHandle()->mem.mcpu.dstBatchIndex[0] = 0;
    for(i =0; i < handle.GetBatchSize() - 1 ; i++)
    {
       handle.GetInitHandle()->mem.mcpu.dstBatchIndex[i+1] = handle.GetInitHandle()->mem.mcpu.dstBatchIndex[i] + handle.GetInitHandle()->mem.mgpu.cmaxDstSize.height[i] * handle.GetInitHandle()->mem.mgpu.cmaxDstSize.width[i] * channel;
    }
    for(i =0; i < handle.GetBatchSize() ; i++)
    {
        if(chnFormat != RPPI_CHN_PLANAR)
        {
            handle.GetInitHandle()->mem.mcpu.dstInc[i] = 1;
        }
        else
        {
            if(!is_padded)
                handle.GetInitHandle()->mem.mcpu.dstInc[i] = handle.GetInitHandle()->mem.mgpu.cdstSize.height[i] * handle.GetInitHandle()->mem.mgpu.cdstSize.width[i];
            else
                handle.GetInitHandle()->mem.mcpu.dstInc[i] = handle.GetInitHandle()->mem.mgpu.cmaxDstSize.height[i] * handle.GetInitHandle()->mem.mgpu.cmaxDstSize.width[i];
        }
    }
#ifdef OCL_COMPILE
    {
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.dstBatchIndex, CL_FALSE, 0, sizeof(Rpp64u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mcpu.dstBatchIndex, 0, NULL, NULL);
        clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.dstInc, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mcpu.dstInc, 0, NULL, NULL);
    }
#elif defined (HIP_COMPILE)
    {
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.dstBatchIndex, handle.GetInitHandle()->mem.mcpu.dstBatchIndex, sizeof(Rpp64u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
        hipMemcpy(handle.GetInitHandle()->mem.mgpu.dstInc, handle.GetInitHandle()->mem.mcpu.dstInc, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    }
#endif
}

template <typename T>
inline void copy_luptr(Rpp8u *luptr,Rpp8u * batch_luptr,Rpp32u nbatchSize, int channel)
{
    int count = 0;
    for(int i = 0; i < nbatchSize; i++)
    {
        for(int j = 0 ; j < 256 * channel ; j++)
        {
            batch_luptr[count] = luptr[j];
            count++;
        }
    }
}

template <typename T>
inline void copy_kernel(Rpp32f *kernel,Rpp32f * batch_kernel, Rpp32u nbatchSize, unsigned int size)
{
    int count = 0;
    for(int i = 0; i < nbatchSize; i++)
    {
        for(int j = 0 ; j < size ; j++)
        {
            batch_kernel[count] = kernel[j];
            count++;
        }
    }
}

inline void validate_image_size(RppiSize imgSize)
{
    if(!(imgSize.width >= 0) || !(imgSize.height >= 0))
    {
        exit(0);
    }
}

inline void validate_float_range(Rpp32f min, Rpp32f max, Rpp32f *value)
{
    if(!(*value <= max) || !(*value >= min))
    {
        *value = max;
    }
}

inline void validate_double_range(Rpp64f min, Rpp64f max, Rpp64f *value)
{
    if(!(*value <= max) || !(*value >= min))
    {
        *value = max;
    }
}

inline void validate_int_range(Rpp32s min, Rpp32s max, Rpp32s *value)
{
    if(!(*value <= max) || !(*value >= min))
    {
        *value = max;
    }
}

inline void validate_unsigned_int_range(Rpp32u min, Rpp32u max, Rpp32u *value)
{
    if(!(*value <= max) || !(*value >= min))
    {
        *value = max;
    }
}

inline void validate_int_max(Rpp32s max, Rpp32s *value)
{
    if(!(*value <= max))
    {
       *value = max;
    }
}

inline void validate_unsigned_int_max(Rpp32u max, Rpp32u *value)
{
    if(!(*value <= max))
    {
       *value = max;
    }
}

inline void validate_int_min(Rpp32s min, Rpp32s *value)
{
    if(!(*value >= min))
    {
       *value = min;
    }
}

inline void validate_unsigned_int_min(Rpp32u min, Rpp32u *value)
{
    if(!(*value >= min))
    {
       *value = min;
    }
}

inline void validate_float_max(Rpp32f max, Rpp32f *value)
{
    if(!(*value <= max))
    {
       *value = max;
    }
}

inline void validate_float_min(Rpp32f min, Rpp32f *value)
{
    if(!(*value >= min))
    {
       *value = min;
    }
}

inline void validate_affine_matrix(Rpp32f* affine)
{
    if((affine[0] * affine[4] - affine[1] * affine[3]) == 0)
    {
        affine[0] = 1;
        affine[1] = 0;
        affine[3] = 0;
        affine[4] = 1;
    }
}

inline void brightness_validate(RppiSize srcSize, Rpp32f alpha, Rpp32f beta)
{
    validate_image_size(srcSize);
    validate_float_range(0, 2, &alpha);
    validate_float_range(0, 255, &beta);
}

inline void brightness_validate(RppiSize srcSize, Rpp32f alpha, Rpp32f beta, Rpp32u nbatchSize)
{
    validate_image_size(srcSize);
    validate_float_range(0, 2, &alpha);
    validate_float_range(0, 255, &beta);
}

inline void brightness_validate(RppiSize *srcSize, Rpp32f alpha, Rpp32f beta, Rpp32u nbatchSize)
{
    for(int i = 0; i < nbatchSize; i++)
    {
        validate_image_size(srcSize[i]);
    }
    validate_float_range(0, 2, &alpha);
    validate_float_range(0, 255, &beta);
}

inline void brightness_validate(RppiSize srcSize, Rpp32f *alpha, Rpp32f *beta, Rpp32u nbatchSize)
{
    validate_image_size(srcSize);
    for(int i = 0; i < nbatchSize; i++)
    {
        validate_float_range(0, 2, &alpha[i]);
        validate_float_range(0, 255, &beta[i]);
    }
}

inline void brightness_validate(RppiSize *srcSize, Rpp32f *alpha, Rpp32f *beta, Rpp32u nbatchSize)
{
    for(int i = 0; i < nbatchSize; i++)
    {
        validate_image_size(srcSize[i]);
        validate_float_range(0, 2, &alpha[i]);
        validate_float_range(0, 255, &beta[i]);
    }
}

inline void histogram_equalize_validate(RppiSize srcSize)
{
    validate_image_size(srcSize);
}

inline void histogram_equalize_validate(RppiSize *srcSize, Rpp32u nbatchSize)
{
    for(int i = 0; i < nbatchSize; i++)
    {
        validate_image_size(srcSize[i]);
    }
}

inline void histogram_equalize_validate(RppiSize srcSize, Rpp32u nbatchSize)
{
    validate_image_size(srcSize);
}

#endif
