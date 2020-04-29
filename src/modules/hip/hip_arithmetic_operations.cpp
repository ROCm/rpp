#include "hip_declarations.hpp"
/********* Absolute Difference *********/

RppStatus
absolute_difference_hip ( Rpp8u* srcPtr1,Rpp8u* srcPtr2,
                 RppiSize srcSize, Rpp8u* dstPtr,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};
    handle.AddKernel("", "", "absolute_difference.cpp", "absolute_difference", vld, vgd, "")(srcPtr1,
                                                                                            srcPtr2,
                                                                                            dstPtr,
                                                                                            srcSize.height,
                                                                                            srcSize.width,
                                                                                            channel);

    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    return RPP_SUCCESS;

}


RppStatus
absolute_difference_hip_batch ( Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{

    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};
    handle.AddKernel("", "", "absolute_difference.cpp", "absolute_difference_batch", vld, vgd, "")(srcPtr1, srcPtr2, dstPtr,
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
                                                                                        plnpkdind
                                                                                        );
    return RPP_SUCCESS;
}


/**************** Accumulate *******************/
RppStatus
accumulate_hip ( Rpp8u* srcPtr1,Rpp8u* srcPtr2,
                 RppiSize srcSize,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};
    handle.AddKernel("", "", "accumulate.cpp", "accumulate", vld, vgd, "")(srcPtr1,
                                                                        srcPtr2,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel);
    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    return RPP_SUCCESS;

}



RppStatus
accumulate_hip_batch ( Rpp8u* srcPtr1,Rpp8u* srcPtr2,rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{

    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};
    handle.AddKernel("", "", "accumulate.cpp", "accumulate_batch", vld, vgd, "")(srcPtr1, srcPtr2,
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
                                                                plnpkdind
                                                                );
    return RPP_SUCCESS;
}

/**************** Accumulate weighted *******************/

RppStatus
accumulate_weighted_hip ( Rpp8u* srcPtr1,Rpp8u* srcPtr2,
                 RppiSize srcSize, Rpp64f alpha,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};
    handle.AddKernel("", "", "accumulate.cpp", "accumulate_weighted", vld, vgd, "")(srcPtr1,
                                                                                   srcPtr2,
                                                                                   alpha,
                                                                                   srcSize.height,
                                                                                   srcSize.width,
                                                                                   channel);
    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    return RPP_SUCCESS;

}


RppStatus
accumulate_weighted_hip_batch ( Rpp8u* srcPtr1,Rpp8u* srcPtr2,rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{

    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};
    handle.AddKernel("", "", "accumulate.cpp", "accumulate_weighted_batch", vld, vgd, "")(srcPtr1, srcPtr2,
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
                                                                plnpkdind
                                                                );
    return RPP_SUCCESS;
}


 /************* Arithmetic Add ************/

RppStatus
add_hip ( Rpp8u* srcPtr1,Rpp8u* srcPtr2,
                 RppiSize srcSize, Rpp8u* dstPtr,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};
    handle.AddKernel("", "", "add.cpp", "add", vld, vgd, "")(srcPtr1,
                                                            srcPtr2,
                                                            dstPtr,
                                                            srcSize.height,
                                                            srcSize.width,
                                                            channel);

    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    return RPP_SUCCESS;

}


RppStatus
add_hip_batch ( Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{

    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};
    handle.AddKernel("", "", "add.cpp", "add_batch", vld, vgd, "")(srcPtr1, srcPtr2, dstPtr,
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
                                                                plnpkdind
                                                                );
    return RPP_SUCCESS;
}

/**************** Arithmetic Subtract *******************/
RppStatus
subtract_hip ( Rpp8u* srcPtr1,Rpp8u* srcPtr2,
                 RppiSize srcSize, Rpp8u* dstPtr,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};
    handle.AddKernel("", "", "subtract.cpp", "subtract", vld, vgd, "")(srcPtr1,
                                                                      srcPtr2,
                                                                      dstPtr,
                                                                      srcSize.height,
                                                                      srcSize.width,
                                                                      channel);
    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    return RPP_SUCCESS;

}



RppStatus
subtract_hip_batch ( Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{

    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};
    handle.AddKernel("", "", "subtract.cpp", "subtract_batch", vld, vgd, "")(srcPtr1, srcPtr2, dstPtr,
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
                                                                plnpkdind
                                                                );
    return RPP_SUCCESS;
}

/**************** Magnitude *******************/
RppStatus
magnitude_hip( Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr,
              RppiChnFormat chnFormat, unsigned int channel,
              rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};
    handle.AddKernel("", "", "magnitude.cpp", "magnitude", vld, vgd, "")(srcPtr1,
                                                                      srcPtr2,
                                                                      dstPtr,
                                                                      srcSize.height,
                                                                      srcSize.width,
                                                                      channel);
    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    return RPP_SUCCESS;
}


RppStatus
magnitude_hip_batch ( Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{

    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};
    handle.AddKernel("", "", "magnitude.cpp", "magnitude_batch", vld, vgd, "")(srcPtr1, srcPtr2, dstPtr,
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
                                                                plnpkdind
                                                                );
    return RPP_SUCCESS;
}
/**************** Multiply *******************/
RppStatus
multiply_hip( Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr,
            RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};
    handle.AddKernel("", "", "multiply.cpp", "multiply", vld, vgd, "")(srcPtr1,
                                                                      srcPtr2,
                                                                      dstPtr,
                                                                      srcSize.height,
                                                                      srcSize.width,
                                                                      channel);

    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    return RPP_SUCCESS;
}

RppStatus
multiply_hip_batch ( Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{

    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};
    handle.AddKernel("", "", "multiply.cpp", "multiply_batch", vld, vgd, "")(srcPtr1, srcPtr2, dstPtr,
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
                                                                plnpkdind
                                                                );
    return RPP_SUCCESS;
}

/**************** Phase *******************/
RppStatus
phase_hip( Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr,
         RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};
    handle.AddKernel("", "", "phase.cpp", "phase", vld, vgd, "")(srcPtr1,
                                                                      srcPtr2,
                                                                      dstPtr,
                                                                      srcSize.height,
                                                                      srcSize.width,
                                                                      channel);
    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    return RPP_SUCCESS;
}

RppStatus
phase_hip_batch ( Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{

    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};
    handle.AddKernel("", "", "phase.cpp", "phase_batch", vld, vgd, "")(srcPtr1, srcPtr2, dstPtr,
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
                                                                plnpkdind
                                                                );
    return RPP_SUCCESS;
}



/**************** Accumulate squared *******************/
RppStatus
accumulate_squared_hip(Rpp8u* srcPtr, RppiSize srcSize, RppiChnFormat chnFormat,
                    unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};
    handle.AddKernel("", "", "accumulate.cpp", "accumulate_squared", vld, vgd, "")(srcPtr,
                                                                      srcSize.height,
                                                                      srcSize.width,
                                                                      channel);
    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    return RPP_SUCCESS;
    
}

RppStatus
accumulate_squared_hip_batch ( Rpp8u* srcPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{

    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

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
                                                                plnpkdind
                                                                );
    return RPP_SUCCESS;
}

/**************** Tensor functions *******************/
RppStatus
tensor_add_hip(Rpp32u tensorDimension, Rpp32u* tensorDimensionValues, Rpp8u srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle)
{ 
    // unsigned short counter=0;

    size_t gDim3[3];
    if(tensorDimension == 1)
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = 1;
        gDim3[2] = 1;
    }
    else if(tensorDimension == 2)
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
        for(int i = 2 ; i < tensorDimension ; i++)
        {    
            value *= tensorDimensionValues[i];
        }
        gDim3[2] = value;
    }
    unsigned int dim1,dim2,dim3;
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
    //---- Args Setter
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &tensorDimension);
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr1);
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr2);
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &dim1);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &dim2);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &dim3);
    // //----
    // CreateProgramFromBinary(theQueue,"tensor.cpp","tensor.cpp.bin","tensor_add",theProgram,theKernel);
    // clRetainKernel(theKernel);    
    // cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    return RPP_SUCCESS;
}

RppStatus
tensor_subtract_hip(Rpp32u tensorDimension, Rpp32u* tensorDimensionValues, Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle)
{ 

    size_t gDim3[3];
    if(tensorDimension == 1)
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = 1;
        gDim3[2] = 1;
    }
    else if(tensorDimension == 2)
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
        for(int i = 2 ; i < tensorDimension ; i++)
        {    
            value *= tensorDimensionValues[i];
        }
        gDim3[2] = value;
    }
    
    unsigned int dim1,dim2,dim3;
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
    //---- Args Setter
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &tensorDimension);
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr1);
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr2);
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &dim1);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &dim2);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &dim3);
    // //----
    // CreateProgramFromBinary(theQueue,"tensor.cpp","tensor.cpp.bin","tensor_subtract",theProgram,theKernel);
    // clRetainKernel(theKernel);
    // cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    return RPP_SUCCESS;
}

RppStatus
tensor_multiply_hip(Rpp32u tensorDimension, Rpp32u* tensorDimensionValues, Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle)
{ 

    size_t gDim3[3];
    if(tensorDimension == 1)
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = 1;
        gDim3[2] = 1;
    }
    else if(tensorDimension == 2)
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
        for(int i = 2 ; i < tensorDimension ; i++)
        {    
            value *= tensorDimensionValues[i];
        }
        gDim3[2] = value;
    }
    
    unsigned int dim1,dim2,dim3;
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
    //---- Args Setter
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &tensorDimension);
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr1);
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr2);
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &dim1);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &dim2);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &dim3);
    // //---- 
    // CreateProgramFromBinary(theQueue,"tensor.cpp","tensor.cpp.bin","tensor_multiply",theProgram,theKernel);
    // clRetainKernel(theKernel);
    // cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    return RPP_SUCCESS;    
}

RppStatus
tensor_matrix_multiply_hip(Rpp8u* srcPtr1, Rpp8u* srcPtr2, Rpp32u* tensorDimensionValues1,
 Rpp32u* tensorDimensionValues2, Rpp8u* dstPtr, rpp::Handle& handle)
{ 

    size_t gDim3[3];
    gDim3[0] = tensorDimensionValues2[1];
    gDim3[1] = tensorDimensionValues1[0];
    gDim3[2] = 1;

    unsigned int a,b,c,d;
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
    //---- Args Setter
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr1);
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr2);
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &a);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &b);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &c);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &d);
    // //----  
    // CreateProgramFromBinary(theQueue,"tensor.cpp","tensor.cpp.bin","tensor_matrix_multiply",theProgram,theKernel);
    // clRetainKernel(theKernel);
    // cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    return RPP_SUCCESS;    
}


/**************** Accumulate squared *******************/
// RppStatus
// mean_stddev_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp32f *mean, Rpp32f *stddev, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
// {
//     unsigned short counter=0;
    
//     cl_kernel theKernel;
//     cl_program theProgram;
//     CreateProgramFromBinary(theQueue,"mean_stddev.cpp","mean_stddev.cpp.bin","sum",theProgram,theKernel);
//     clRetainKernel(theKernel);   
    
//     int i;
    
//     const int LIST_SIZE = srcSize.height * srcSize.width * channel;
//     int numGroups = std::ceil(LIST_SIZE / 256);
    
//     cl_context theContext;
//     clGetCommandQueueInfo(theQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
//     cl_device_id theDevice;
//     clGetCommandQueueInfo(theQueue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);

//     float sum = 0;
//     long *partial_sum;
//     partial_sum = (long *) calloc (numGroups, sizeof(long));
//     Rpp8u* b_mem_obj = clCreateBuffer(theContext, Rpp8u*_WRITE_ONLY, numGroups * sizeof(long), NULL, NULL);
//     clEnqueueWriteBuffer(theQueue, b_mem_obj, CL_TRUE, 0, numGroups * sizeof(long), partial_sum, 0, NULL, NULL);

//     float mean_sum = 0;
//     float *partial_mean_sum;
//     partial_mean_sum = (float *) calloc (numGroups, sizeof(float));
//     Rpp8u* c_mem_obj = clCreateBuffer(theContext, Rpp8u*_WRITE_ONLY, numGroups * sizeof(float), NULL, NULL);
//     clEnqueueWriteBuffer(theQueue, c_mem_obj, CL_TRUE, 0, numGroups * sizeof(float), partial_mean_sum, 0, NULL, NULL);


//     clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &srcPtr);
//     clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &b_mem_obj);

//     size_t gDim3[3];
//     gDim3[0] = LIST_SIZE;
//     gDim3[1] = 1;
//     gDim3[2] = 1;
//     size_t local_item_size[3];
//     local_item_size[0] = 256;
//     local_item_size[1] = 1;
//     local_item_size[2] = 1;
//     cl_kernel_implementer (theQueue, gDim3, local_item_size, theProgram, theKernel);
//     clEnqueueReadBuffer(theQueue, b_mem_obj, CL_TRUE, 0, numGroups * sizeof(long), partial_sum, 0, NULL, NULL);   
    
//     for(i = 0; i < numGroups; i++)
//     {
//         sum += (float)partial_sum[i];
//     }

//     *mean = (sum) / LIST_SIZE ;


//     CreateProgramFromBinary(theQueue,"mean_stddev.cpp","mean_stddev.cpp.bin","mean_stddev",theProgram,theKernel);
//     clRetainKernel(theKernel); 

//     counter = 0;
//     float meanCopy = *mean;
//     clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &srcPtr);
//     clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &c_mem_obj);
//     clSetKernelArg(theKernel, counter++, sizeof(float), &meanCopy);
//     cl_kernel_implementer (theQueue, gDim3, local_item_size, theProgram, theKernel);
//     clEnqueueReadBuffer(theQueue, c_mem_obj, CL_TRUE, 0, numGroups * sizeof(float), partial_mean_sum, 0, NULL, NULL);  
//     for(i = 0; i < numGroups; i++)
//     {
//         mean_sum += partial_mean_sum[i];
//     }
    
//     mean_sum = mean_sum / LIST_SIZE ;
//     *stddev = mean_sum;

//     clReleaseMemObject(b_mem_obj); 
//     free(partial_sum);
//     clReleaseMemObject(c_mem_obj); 
//     free(partial_mean_sum);
    
// }
