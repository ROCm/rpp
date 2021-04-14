#include "hip/hip_runtime_api.h"
#include "hip_declarations.hpp"
#include "kernel/rpp_hip_host_decls.hpp"

/******************** absolute_difference ********************/

RppStatus
absolute_difference_hip(Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};

    handle.AddKernel("", "", "absolute_difference.cpp", "absolute_difference", vld, vgd, "")(srcPtr1,
                                                                                             srcPtr2,
                                                                                             dstPtr,
                                                                                             srcSize.height,
                                                                                             srcSize.width,
                                                                                             channel);

    return RPP_SUCCESS;
}

RppStatus
absolute_difference_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    
#if defined (HIPRTC)

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};

    handle.AddKernel("", "", "absolute_difference.cpp", "absolute_difference_batch", vld, vgd, "")(srcPtr1,
                                                                                                   srcPtr2,
                                                                                                   dstPtr,
                                                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                                                   handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                                   handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                                   handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                                   handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                                   channel,
                                                                                                   handle.GetInitHandle()->mem.mgpu.inc,
                                                                                                   plnpkdind);
#elif defined(STATIC)

    hip_exec_absolute_difference_batch(srcPtr1, srcPtr2, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

/******************** accumulate ********************/

RppStatus
accumulate_hip(Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};

    handle.AddKernel("", "", "accumulate.cpp", "accumulate", vld, vgd, "")(srcPtr1,
                                                                           srcPtr2,
                                                                           srcSize.height,
                                                                           srcSize.width,
                                                                           channel);

    return RPP_SUCCESS;
}

RppStatus
accumulate_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2,rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};

    handle.AddKernel("", "", "accumulate.cpp", "accumulate_batch", vld, vgd, "")(srcPtr1,
                                                                                 srcPtr2,
                                                                                 handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                                 handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                                 handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                                 handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                                 handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                 handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                 handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                 handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                 channel,
                                                                                 handle.GetInitHandle()->mem.mgpu.inc,
                                                                                 plnpkdind);
#elif defined(STATIC)

    hip_exec_accumulate_batch(srcPtr1, srcPtr2, handle, chnFormat, channel, plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

/******************** accumulate_weighted ********************/

RppStatus
accumulate_weighted_hip(Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp64f alpha, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};

    handle.AddKernel("", "", "accumulate.cpp", "accumulate_weighted", vld, vgd, "")(srcPtr1,
                                                                                   srcPtr2,
                                                                                   alpha,
                                                                                   srcSize.height,
                                                                                   srcSize.width,
                                                                                   channel);

    return RPP_SUCCESS;
}

RppStatus
accumulate_weighted_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2,rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    #if defined (HIPRTC)

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};

    handle.AddKernel("", "", "accumulate.cpp", "accumulate_weighted_batch", vld, vgd, "")(srcPtr1,
                                                                                          srcPtr2,
                                                                                          handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                                          handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                                          handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                                          handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                                          handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                                          handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                          handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                          handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                          handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                          channel,
                                                                                          handle.GetInitHandle()->mem.mgpu.inc,
                                                                                          plnpkdind);

#elif defined(STATIC)

    hip_exec_accumulate_weighted_batch(srcPtr1, srcPtr2, handle, chnFormat, channel, plnpkdind, max_height, max_width);

#endif
    return RPP_SUCCESS;
}

/******************** add ********************/

RppStatus
add_hip(Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};

    handle.AddKernel("", "", "add.cpp", "add", vld, vgd, "")(srcPtr1,
                                                             srcPtr2,
                                                             dstPtr,
                                                             srcSize.height,
                                                             srcSize.width,
                                                             channel);

    return RPP_SUCCESS;
}

RppStatus
add_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};

    handle.AddKernel("", "", "add.cpp", "add_batch", vld, vgd, "")(srcPtr1,
                                                                   srcPtr2,
                                                                   dstPtr,
                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                   handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                   handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                   handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                   handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                   channel,
                                                                   handle.GetInitHandle()->mem.mgpu.inc,
                                                                   plnpkdind);

#elif defined(STATIC)

    hip_exec_add_batch(srcPtr1, srcPtr2, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

/******************** subtract ********************/

RppStatus
subtract_hip(Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};

    handle.AddKernel("", "", "subtract.cpp", "subtract", vld, vgd, "")(srcPtr1,
                                                                       srcPtr2,
                                                                       dstPtr,
                                                                       srcSize.height,
                                                                       srcSize.width,
                                                                       channel);

    return RPP_SUCCESS;
}

RppStatus
subtract_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    
#if defined (HIPRTC)

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};

    handle.AddKernel("", "", "subtract.cpp", "subtract_batch", vld, vgd, "")(srcPtr1,
                                                                             srcPtr2,
                                                                             dstPtr,
                                                                             handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                             handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                             handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                             handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                             handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                             handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                             handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                             handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                             channel,
                                                                             handle.GetInitHandle()->mem.mgpu.inc,
                                                                             plnpkdind);
#elif defined(STATIC)

    hip_exec_subtract_batch(srcPtr1, srcPtr2, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

#endif
    return RPP_SUCCESS;
}

/******************** magnitude ********************/

RppStatus
magnitude_hip(Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};

    handle.AddKernel("", "", "magnitude.cpp", "magnitude", vld, vgd, "")(srcPtr1,
                                                                         srcPtr2,
                                                                         dstPtr,
                                                                         srcSize.height,
                                                                         srcSize.width,
                                                                         channel);

    return RPP_SUCCESS;
}

RppStatus
magnitude_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};

    handle.AddKernel("", "", "magnitude.cpp", "magnitude_batch", vld, vgd, "")(srcPtr1,
                                                                               srcPtr2,
                                                                               dstPtr,
                                                                               handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                               handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                               handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                               handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                               handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                               handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                               handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                               handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                               channel,
                                                                               handle.GetInitHandle()->mem.mgpu.inc,
                                                                               plnpkdind);
#elif defined(STATIC)

    hip_exec_magnitude_batch(srcPtr1, srcPtr2, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

/******************** multiply ********************/

RppStatus
multiply_hip(Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};

    handle.AddKernel("", "", "multiply.cpp", "multiply", vld, vgd, "")(srcPtr1,
                                                                       srcPtr2,
                                                                       dstPtr,
                                                                       srcSize.height,
                                                                       srcSize.width,
                                                                       channel);

    return RPP_SUCCESS;
}

RppStatus
multiply_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};

    handle.AddKernel("", "", "multiply.cpp", "multiply_batch", vld, vgd, "")(srcPtr1,
                                                                             srcPtr2,
                                                                             dstPtr,
                                                                             handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                             handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                             handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                             handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                             handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                             handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                             handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                             handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                             channel,
                                                                             handle.GetInitHandle()->mem.mgpu.inc,
                                                                             plnpkdind);
#elif defined(STATIC)

    hip_exec_multiply_batch(srcPtr1, srcPtr2, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

/******************** phase ********************/

RppStatus
phase_hip(Rpp8u* srcPtr1, Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};

    handle.AddKernel("", "", "phase.cpp", "phase", vld, vgd, "")(srcPtr1,
                                                                 srcPtr2,
                                                                 dstPtr,
                                                                 srcSize.height,
                                                                 srcSize.width,
                                                                 channel);

    return RPP_SUCCESS;
}

RppStatus
phase_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};

    handle.AddKernel("", "", "phase.cpp", "phase_batch", vld, vgd, "")(srcPtr1,
                                                                       srcPtr2,
                                                                       dstPtr,
                                                                       handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                       handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                       channel,
                                                                       handle.GetInitHandle()->mem.mgpu.inc,
                                                                       plnpkdind);
#elif defined(STATIC)

    hip_exec_phase_batch(srcPtr1, srcPtr2, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

/******************** accumulate_squared ********************/

RppStatus
accumulate_squared_hip(Rpp8u* srcPtr, RppiSize srcSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};

    handle.AddKernel("", "", "accumulate.cpp", "accumulate_squared", vld, vgd, "")(srcPtr,
                                                                                   srcSize.height,
                                                                                   srcSize.width,
                                                                                   channel);

    return RPP_SUCCESS;
}

RppStatus
accumulate_squared_hip_batch(Rpp8u* srcPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};

    handle.AddKernel("", "", "accumulate.cpp", "accumulate_squared_batch", vld, vgd, "")(srcPtr,
                                                                                         handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                                         handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                                         handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                                         handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                                         handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                         handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                         handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                         handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                         channel,
                                                                                         handle.GetInitHandle()->mem.mgpu.inc,
                                                                                         plnpkdind);
#elif defined(STATIC)

    hip_exec_accumulate_squared_batch(srcPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

/******************** tensor_add ********************/

RppStatus
tensor_add_hip(Rpp32u tensorDimension, Rpp32u *tensorDimensionValues, Rpp8u* srcPtr1, Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle &handle)
{
    size_t gDim3[3];
    if (tensorDimension == 1)
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = 1;
        gDim3[2] = 1;
    }
    else if (tensorDimension == 2)
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = tensorDimensionValues[1];
        gDim3[2] = 1;
    }
    else
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = tensorDimensionValues[1];
        int value = 1;
        for (int i = 2; i < tensorDimension; i++)
        {
            value *= tensorDimensionValues[i];
        }
        gDim3[2] = value;
    }
    unsigned int dim1, dim2, dim3;
    dim1 = gDim3[0];
    dim2 = gDim3[1];
    dim3 = gDim3[2];
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};

    handle.AddKernel("", "", "tensor.cpp", "tensor_add", vld, vgd, "")(tensorDimension,
                                                                       srcPtr1,
                                                                       srcPtr2,
                                                                       dstPtr,
                                                                       dim1,
                                                                       dim2,
                                                                       dim3);

    return RPP_SUCCESS;
}

/******************** tensor_subtract ********************/

RppStatus
tensor_subtract_hip(Rpp32u tensorDimension, Rpp32u *tensorDimensionValues, Rpp8u* srcPtr1, Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle &handle)
{
    size_t gDim3[3];
    if (tensorDimension == 1)
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = 1;
        gDim3[2] = 1;
    }
    else if (tensorDimension == 2)
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = tensorDimensionValues[1];
        gDim3[2] = 1;
    }
    else
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = tensorDimensionValues[1];
        int value = 1;
        for (int i = 2; i < tensorDimension; i++)
        {
            value *= tensorDimensionValues[i];
        }
        gDim3[2] = value;
    }
    unsigned int dim1, dim2, dim3;
    dim1 = gDim3[0];
    dim2 = gDim3[1];
    dim3 = gDim3[2];
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};

    handle.AddKernel("", "", "tensor.cpp", "tensor_subtract", vld, vgd, "")(tensorDimension,
                                                                            srcPtr1,
                                                                            srcPtr2,
                                                                            dstPtr,
                                                                            dim1,
                                                                            dim2,
                                                                            dim3);

    return RPP_SUCCESS;
}

/******************** tensor_multiply ********************/

RppStatus
tensor_multiply_hip(Rpp32u tensorDimension, Rpp32u *tensorDimensionValues, Rpp8u* srcPtr1, Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle &handle)
{
    size_t gDim3[3];
    if (tensorDimension == 1)
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = 1;
        gDim3[2] = 1;
    }
    else if (tensorDimension == 2)
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = tensorDimensionValues[1];
        gDim3[2] = 1;
    }
    else
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = tensorDimensionValues[1];
        int value = 1;
        for (int i = 2; i < tensorDimension; i++)
        {
            value *= tensorDimensionValues[i];
        }
        gDim3[2] = value;
    }
    unsigned int dim1, dim2, dim3;
    dim1 = gDim3[0];
    dim2 = gDim3[1];
    dim3 = gDim3[2];
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};

    handle.AddKernel("", "", "tensor.cpp", "tensor_multiply", vld, vgd, "")(tensorDimension,
                                                                            srcPtr1,
                                                                            srcPtr2,
                                                                            dstPtr,
                                                                            dim1,
                                                                            dim2,
                                                                            dim3);

    return RPP_SUCCESS;
}

/******************** tensor_matrix_multiply ********************/

RppStatus
tensor_matrix_multiply_hip(Rpp8u* srcPtr1, Rpp8u* srcPtr2, Rpp32u *tensorDimensionValues1, Rpp32u *tensorDimensionValues2, Rpp8u* dstPtr, rpp::Handle &handle)
{
    size_t gDim3[3];
    gDim3[0] = tensorDimensionValues2[1];
    gDim3[1] = tensorDimensionValues1[0];
    gDim3[2] = 1;
    unsigned int a, b, c, d;
    a = tensorDimensionValues1[0];
    b = tensorDimensionValues1[1];
    c = tensorDimensionValues2[0];
    d = tensorDimensionValues2[1];
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};

    handle.AddKernel("", "", "tensor.cpp", "tensor_matrix_multiply", vld, vgd, "")(srcPtr1,
                                                                                   srcPtr2,
                                                                                   dstPtr,
                                                                                   a,
                                                                                   b,
                                                                                   c,
                                                                                   d);
    return RPP_SUCCESS;
}