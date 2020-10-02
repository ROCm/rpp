#include <cl/rpp_cl_common.hpp>
#include "cl_declarations.hpp"

/********* Absolute Difference *********/

RppStatus
absolute_difference_cl ( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize, cl_mem dstPtr,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
    handle.AddKernel("", "", "absolute_difference.cl", "absolute_difference", vld, vgd, "")(srcPtr1,
                                                                                            srcPtr2,
                                                                                            dstPtr,
                                                                                            srcSize.height,
                                                                                            srcSize.width,
                                                                                            channel);
    return RPP_SUCCESS;

}


RppStatus
absolute_difference_cl_batch ( cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle& handle,
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
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "absolute_difference.cl", "absolute_difference_batch", vld, vgd, "")(srcPtr1, srcPtr2, dstPtr,
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
accumulate_cl ( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
    handle.AddKernel("", "", "accumulate.cl", "accumulate", vld, vgd, "")(srcPtr1,
                                                                        srcPtr2,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel);
    return RPP_SUCCESS;

}



RppStatus
accumulate_cl_batch ( cl_mem srcPtr1,cl_mem srcPtr2,rpp::Handle& handle,
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
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "accumulate.cl", "accumulate_batch", vld, vgd, "")(srcPtr1, srcPtr2,
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
accumulate_weighted_cl ( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize, Rpp32f alpha,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
    handle.AddKernel("", "", "accumulate.cl", "accumulate_weighted", vld, vgd, "")(srcPtr1,
                                                                                   srcPtr2,
                                                                                   alpha,
                                                                                   srcSize.height,
                                                                                   srcSize.width,
                                                                                   channel);
    return RPP_SUCCESS;

}


RppStatus
accumulate_weighted_cl_batch ( cl_mem srcPtr1,cl_mem srcPtr2,rpp::Handle& handle,
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
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "accumulate.cl", "accumulate_weighted_batch", vld, vgd, "")(srcPtr1, srcPtr2,
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
                                                                plnpkdind
                                                                );
    return RPP_SUCCESS;
}


 /************* Arithmetic Add ************/

RppStatus
add_cl ( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize, cl_mem dstPtr,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
    handle.AddKernel("", "", "add.cl", "add", vld, vgd, "")(srcPtr1,
                                                            srcPtr2,
                                                            dstPtr,
                                                            srcSize.height,
                                                            srcSize.width,
                                                            channel);
    return RPP_SUCCESS;

}


RppStatus
add_cl_batch ( cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle& handle,
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
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "add.cl", "add_batch", vld, vgd, "")(srcPtr1, srcPtr2, dstPtr,
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
subtract_cl ( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize, cl_mem dstPtr,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
    handle.AddKernel("", "", "subtract.cl", "subtract", vld, vgd, "")(srcPtr1,
                                                                      srcPtr2,
                                                                      dstPtr,
                                                                      srcSize.height,
                                                                      srcSize.width,
                                                                      channel);
    return RPP_SUCCESS;

}



RppStatus
subtract_cl_batch ( cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle& handle,
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
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "subtract.cl", "subtract_batch", vld, vgd, "")(srcPtr1, srcPtr2, dstPtr,
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
magnitude_cl( cl_mem srcPtr1,cl_mem srcPtr2, RppiSize srcSize, cl_mem dstPtr,
              RppiChnFormat chnFormat, unsigned int channel,
              rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
    handle.AddKernel("", "", "magnitude.cl", "magnitude", vld, vgd, "")(srcPtr1,
                                                                      srcPtr2,
                                                                      dstPtr,
                                                                      srcSize.height,
                                                                      srcSize.width,
                                                                      channel);
    return RPP_SUCCESS;
}


RppStatus
magnitude_cl_batch ( cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle& handle,
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
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "magnitude.cl", "magnitude_batch", vld, vgd, "")(srcPtr1, srcPtr2, dstPtr,
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
multiply_cl( cl_mem srcPtr1,cl_mem srcPtr2, RppiSize srcSize, cl_mem dstPtr,
            RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
    handle.AddKernel("", "", "multiply.cl", "multiply", vld, vgd, "")(srcPtr1,
                                                                      srcPtr2,
                                                                      dstPtr,
                                                                      srcSize.height,
                                                                      srcSize.width,
                                                                      channel);

    return RPP_SUCCESS;
}

RppStatus
multiply_cl_batch ( cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle& handle,
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
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "multiply.cl", "multiply_batch", vld, vgd, "")(srcPtr1, srcPtr2, dstPtr,
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
phase_cl( cl_mem srcPtr1,cl_mem srcPtr2, RppiSize srcSize, cl_mem dstPtr,
         RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
    handle.AddKernel("", "", "phase.cl", "phase", vld, vgd, "")(srcPtr1,
                                                                      srcPtr2,
                                                                      dstPtr,
                                                                      srcSize.height,
                                                                      srcSize.width,
                                                                      channel);
    return RPP_SUCCESS;
}



RppStatus
phase_cl_batch ( cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle& handle,
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
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "phase.cl", "phase_batch", vld, vgd, "")(srcPtr1, srcPtr2, dstPtr,
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
accumulate_squared_cl(cl_mem srcPtr, RppiSize srcSize, RppiChnFormat chnFormat,
                    unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
    handle.AddKernel("", "", "accumulate.cl", "accumulate_squared", vld, vgd, "")(srcPtr,
                                                                      srcSize.height,
                                                                      srcSize.width,
                                                                      channel);
    return RPP_SUCCESS;
    
}

RppStatus
accumulate_squared_cl_batch ( cl_mem srcPtr, rpp::Handle& handle,
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
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "accumulate.cl", "accumulate_squared_batch", vld, vgd, "")(srcPtr,
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
tensor_add_cl(Rpp32u tensorDimension, Rpp32u* tensorDimensionValues, cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle& handle)
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
    handle.AddKernel("", "", "tensor.cl", "tensor_add", vld, vgd, "")(tensorDimension,
                                                                    srcPtr1,
                                                                    srcPtr2,
                                                                    dstPtr,
                                                                    dim1,
                                                                    dim2,
                                                                    dim3);
    return RPP_SUCCESS;
}

RppStatus
tensor_subtract_cl(Rpp32u tensorDimension, Rpp32u* tensorDimensionValues, cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle& handle)
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
    handle.AddKernel("", "", "tensor.cl", "tensor_subtract", vld, vgd, "")(tensorDimension,
                                                                    srcPtr1,
                                                                    srcPtr2,
                                                                    dstPtr,
                                                                    dim1,
                                                                    dim2,
                                                                    dim3);
    return RPP_SUCCESS;
}

RppStatus
tensor_multiply_cl(Rpp32u tensorDimension, Rpp32u* tensorDimensionValues, cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle& handle)
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
    handle.AddKernel("", "", "tensor.cl", "tensor_multiply", vld, vgd, "")(tensorDimension,
                                                                    srcPtr1,
                                                                    srcPtr2,
                                                                    dstPtr,
                                                                    dim1,
                                                                    dim2,
                                                                    dim3);
    return RPP_SUCCESS;    
}

RppStatus
tensor_matrix_multiply_cl(cl_mem srcPtr1, cl_mem srcPtr2, Rpp32u* tensorDimensionValues1, Rpp32u* tensorDimensionValues2, cl_mem dstPtr, rpp::Handle& handle)
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
    handle.AddKernel("", "", "tensor.cl", "tensor_matrix_multiply", vld, vgd, "")(srcPtr1,
                                                                                srcPtr2,
                                                                                dstPtr,
                                                                                a,
                                                                                b,
                                                                                c,
                                                                                d);
    return RPP_SUCCESS;    
}

RppStatus
tensor_transpose_cl( cl_mem srcPtr, cl_mem dstPtr,  Rpp32u* in_tensor_dims, Rpp32u *perm,  RPPTensorFunctionMetaData &tensor_info, rpp::Handle& handle)
{ 
    // unsigned short counter=0;
    size_t gDim3[3];
    gDim3[0] = in_tensor_dims[perm[0]];
    gDim3[1] = in_tensor_dims[perm[1]];
    gDim3[1] = in_tensor_dims[perm[2]] * in_tensor_dims[perm[3]];
      
    unsigned int dim1,dim2,dim3;
    dim1 = gDim3[0];
    dim2 = gDim3[1];
    dim3 = gDim3[2];
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
    handle.AddKernel("", "", "tensor.cl", "tensor_transpose", vld, vgd, "")(
                                                                    srcPtr,
                                                                    dstPtr
                                                                    );
    return RPP_SUCCESS;
}



// RppStatus
// mean_stddev_cl(cl_mem srcPtr, RppiSize srcSize, Rpp32f *mean, Rpp32f *stddev, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
// {
//     unsigned short counter=0;
    
//     cl_kernel theKernel;
//     cl_program theProgram;
//     //CreateProgramFromBinary(theQueue,"mean_stddev.cl","mean_stddev.cl.bin","sum",theProgram,theKernel);
//     //clRetainKernel(theKernel);   
    
//     int i;
    
//     const int LIST_SIZE = srcSize.height * srcSize.width * channel;
//     int numGroups = std::ceil(LIST_SIZE / 256);
    
//     cl_context theContext;
//     clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
//     cl_device_id theDevice;
//     clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);

//     float sum = 0;
//     long *partial_sum;
//     partial_sum = (long *) calloc (numGroups, sizeof(long));
//     cl_mem b_mem_obj = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, numGroups * sizeof(long), NULL, NULL); // For Sum
//     clEnqueueWriteBuffer(handle.GetStream(), b_mem_obj, CL_TRUE, 0, numGroups * sizeof(long), partial_sum, 0, NULL, NULL);

//     float mean_sum = 0;
//     float *partial_mean_sum;
//     partial_mean_sum = (float *) calloc (numGroups, sizeof(float));
//     cl_mem c_mem_obj = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, numGroups * sizeof(float), NULL, NULL); // For StdDev
//     clEnqueueWriteBuffer(handle.GetStream(), c_mem_obj, CL_TRUE, 0, numGroups * sizeof(float), partial_mean_sum, 0, NULL, NULL);


//     //clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
//     //clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &b_mem_obj);

//     size_t gDim3[3];
//     gDim3[0] = LIST_SIZE;
//     gDim3[1] = 1;
//     gDim3[2] = 1;
//     size_t lDim3[3];
//     lDim3[0] = 256;
//     lDim3[1] = 1;
//     lDim3[2] = 1;

//     std::vector<size_t> vld{lDim3[0], lDim3[1], lDim3[2]};
//     std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};

//     handle.AddKernel("", "", "mean_stddev.cl", "sum", vld, vgd, "")(srcPtr, b_mem_obj);

//     //cl_kernel_implementer (theQueue, gDim3, local_item_size, theProgram, theKernel);
//     clEnqueueReadBuffer(handle.GetStream(), b_mem_obj, CL_TRUE, 0, numGroups * sizeof(long), partial_sum, 0, NULL, NULL);   
    
//     for(i = 0; i < numGroups; i++)
//     {
//         sum += (float)partial_sum[i];
//     }

//     *mean = (sum) / LIST_SIZE ;


//     //CreateProgramFromBinary(theQueue,"mean_stddev.cl","mean_stddev.cl.bin","mean_stddev",theProgram,theKernel);
//     //clRetainKernel(theKernel); 

//     counter = 0;
//     float meanCopy = *mean;
//     //clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
//     //clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &c_mem_obj);
//     //clSetKernelArg(theKernel, counter++, sizeof(float), &meanCopy);
//     //cl_kernel_implementer (theQueue, gDim3, local_item_size, theProgram, theKernel);
//     handle.AddKernel("", "", "mean_stddev.cl", "sum", vld, vgd, "")(srcPtr, c_mem_obj, meanCopy);
//     clEnqueueReadBuffer(handle.GetStream(), c_mem_obj, CL_TRUE, 0, numGroups * sizeof(float), partial_mean_sum, 0, NULL, NULL);  
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