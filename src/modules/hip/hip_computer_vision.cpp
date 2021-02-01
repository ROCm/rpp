#include "hip_declarations.hpp"
#include <hip/rpp_hip_common.hpp>

/********************** data_object_copy ************************/
RppStatus
data_object_copy_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat,
                     unsigned int channel, rpp::Handle& handle)
{
    hipMemcpy(dstPtr,srcPtr,sizeof(unsigned char) * srcSize.width * srcSize.height * channel, hipMemcpyDeviceToDevice);
    
    return RPP_SUCCESS;    
}

RppStatus
data_object_copy_hip_batch (Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
                            RppiChnFormat chnFormat, unsigned int channel
                            ){
    unsigned long buffer_size=0;
    for(int i =0; i< handle.GetBatchSize(); i++){
     buffer_size += handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.width[i] * channel;
    }
    hipMemcpy( dstPtr,srcPtr,buffer_size * sizeof(unsigned char),hipMemcpyDeviceToDevice);    
    return RPP_SUCCESS;  
 }
/********************** local binary pattern ************************/
RppStatus
local_binary_pattern_hip ( Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "local_binary_pattern.cpp", "local_binary_pattern_pkd", vld, vgd, "")(srcPtr,
                                                                                            dstPtr,
                                                                                            srcSize.height,
                                                                                            srcSize.width,
                                                                                            channel
                                                                                            );
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "local_binary_pattern.cpp", "local_binary_pattern_pln", vld, vgd, "")(srcPtr,
                                                                                            dstPtr,
                                                                                            srcSize.height,
                                                                                            srcSize.width,
                                                                                            channel
                                                                                            );
    }
    return RPP_SUCCESS;    
}


RppStatus
local_binary_pattern_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
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
    handle.AddKernel("", "", "local_binary_pattern.cpp", "local_binary_pattern_batch", vld, vgd, "")(srcPtr, dstPtr,
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

/********************** Gaussian image pyramid ************************/

RppStatus
gaussian_image_pyramid_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr,
                         Rpp32f stdDev, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    Rpp32f *kernelMain = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
    generate_gaussian_kernel_gpu(stdDev, kernelMain, kernelSize);
    
    Rpp32f* kernel;
    hipMalloc(&kernel,  kernelSize * kernelSize * sizeof(Rpp32f));
    hipMemcpy(kernel,kernelMain,kernelSize * kernelSize * sizeof(Rpp32f),hipMemcpyHostToDevice);

    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "gaussian_image_pyramid.cpp", "gaussian_image_pyramid_pkd", vld, vgd, "")(srcPtr,
                                                                                            dstPtr,
                                                                                            srcSize.height,
                                                                                            srcSize.width,
                                                                                            channel,
                                                                                            kernel,
                                                                                            kernelSize,
                                                                                            kernelSize
                                                                                            );
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "gaussian_image_pyramid.cpp", "gaussian_image_pyramid_pln", vld, vgd, "")(srcPtr,
                                                                                            dstPtr,
                                                                                            srcSize.height,
                                                                                            srcSize.width,
                                                                                            channel,
                                                                                            kernel,
                                                                                            kernelSize,
                                                                                            kernelSize
                                                                                            );
    }
    //---- Args Setter
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &kernel);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);

    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);
    return RPP_SUCCESS;  
}

RppStatus
gaussian_image_pyramid_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
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
    handle.AddKernel("", "", "gaussian_image_pyramid.cpp", "gaussian_image_pyramid_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                                                                                        handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
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
/********************** Control Flow ************************/

RppStatus
control_flow_hip(Rpp8u* srcPtr1, Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr,
 Rpp32u type, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{    
    // unsigned short counter=0;
    // cl_kernel theKernel;
    // cl_program theProgram;
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
    switch(type)
    {
        case 1:
            handle.AddKernel("", "", "bitwise_AND.cpp", "bitwise_AND", vld, vgd, "")(srcPtr1,
                                                                            srcPtr2,
                                                                            dstPtr,
                                                                            srcSize.height,
                                                                            srcSize.width,
                                                                            channel);
            break;
        case 2:
            handle.AddKernel("", "", "inclusive_OR.cpp", "inclusive_OR", vld, vgd, "")(srcPtr1,
                                                                            srcPtr2,
                                                                            dstPtr,
                                                                            srcSize.height,
                                                                            srcSize.width,
                                                                            channel);
            // CreateProgramFromBinary(handle.GetStream(),"inclusive_OR.cpp","inclusive_OR.cpp.bin","inclusive_OR",theProgram,theKernel);
            // clRetainKernel(theKernel);
            break;
        case 3:
            handle.AddKernel("", "", "exclusive_OR.cpp", "exclusive_OR", vld, vgd, "")(srcPtr1,
                                                                            srcPtr2,
                                                                            dstPtr,
                                                                            srcSize.height,
                                                                            srcSize.width,
                                                                            channel);
            // CreateProgramFromBinary(handle.GetStream(),"exclusive_OR.cpp","exclusive_OR.cpp.bin","exclusive_OR",theProgram,theKernel);
            // clRetainKernel(theKernel);
            break;
        case 4:
            handle.AddKernel("", "", "add.cpp", "add", vld, vgd, "")(srcPtr1,
                                                                            srcPtr2,
                                                                            dstPtr,
                                                                            srcSize.height,
                                                                            srcSize.width,
                                                                            channel);
            // CreateProgramFromBinary(handle.GetStream(),"add.cpp","add.cpp.bin","add",theProgram,theKernel);
            // clRetainKernel(theKernel);
            break;
        case 5:
            handle.AddKernel("", "", "subtract.cpp", "subtract", vld, vgd, "")(srcPtr1,
                                                                            srcPtr2,
                                                                            dstPtr,
                                                                            srcSize.height,
                                                                            srcSize.width,
                                                                            channel);
            // CreateProgramFromBinary(handle.GetStream(),"subtract.cpp","subtract.cpp.bin","subtract",theProgram,theKernel);
            // clRetainKernel(theKernel);
            break;
        case 6:
            handle.AddKernel("", "", "multiply.cpp", "multiply", vld, vgd, "")(srcPtr1,
                                                                            srcPtr2,
                                                                            dstPtr,
                                                                            srcSize.height,
                                                                            srcSize.width,
                                                                            channel);
            // CreateProgramFromBinary(handle.GetStream(),"multiply.cpp","multiply.cpp.bin","multiply",theProgram,theKernel);
            // clRetainKernel(theKernel);
            break;
        case 7:
            handle.AddKernel("", "", "min.cpp", "min", vld, vgd, "")(srcPtr1,
                                                                            srcPtr2,
                                                                            dstPtr,
                                                                            srcSize.height,
                                                                            srcSize.width,
                                                                            channel);
            // CreateProgramFromBinary(handle.GetStream(),"min.cpp","min.cpp.bin","min",theProgram,theKernel);
            // clRetainKernel(theKernel);
            break;
        case 8:
            handle.AddKernel("", "", "max.cpp", "max", vld, vgd, "")(srcPtr1,
                                                                            srcPtr2,
                                                                            dstPtr,
                                                                            srcSize.height,
                                                                            srcSize.width,
                                                                            channel);
            // CreateProgramFromBinary(handle.GetStream(),"max.cpp","max.cpp.bin","max",theProgram,theKernel);
            // clRetainKernel(theKernel);
            break;
    }
    // //---- Args Setter
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr1);
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr2);
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
    // //----

    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);

    return RPP_SUCCESS;
}

RppStatus
control_flow_hip_batch ( Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, Rpp32u type, rpp::Handle& handle,
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
    switch(type)
    {
        case 1:
            handle.AddKernel("", "", "bitwise_AND.cpp", "bitwise_AND_batch", vld, vgd, "")(srcPtr1, srcPtr2, dstPtr,
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
            break;
        case 2:
            handle.AddKernel("", "", "inclusive_OR.cpp", "inclusive_OR_batch", vld, vgd, "")(srcPtr1, srcPtr2, dstPtr,
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
            break;
        case 3:
            handle.AddKernel("", "", "exclusive_OR.cpp", "exclusive_OR_batch", vld, vgd, "")(srcPtr1, srcPtr2, dstPtr,
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
            break;
        case 4:
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
            break;
        case 5:
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
            break;
        case 6:
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
            break;
        case 7:
            handle.AddKernel("", "", "min.cpp", "min_batch", vld, vgd, "")(srcPtr1, srcPtr2, dstPtr,
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
            break;
        case 8:
            handle.AddKernel("", "", "max.cpp", "max_batch", vld, vgd, "")(srcPtr1, srcPtr2, dstPtr,
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
            break;
    }
    
    return RPP_SUCCESS;
}

/********************** Convert bit depth ************************/
template <typename T, typename U>
RppStatus
convert_bit_depth_hip(T* srcPtr, RppiSize srcSize, U* dstPtr, Rpp32u type,
             RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    if(type == 1)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "convert_bit_depth.cpp", "convert_bit_depth_u8s8", vld, vgd, "")(srcPtr,
                                                                                            dstPtr,
                                                                                            srcSize.height,
                                                                                            srcSize.width,
                                                                                            channel
                                                                                            );
    }
    else if(type == 2)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "convert_bit_depth.cpp", "convert_bit_depth_u8u16", vld, vgd, "")(srcPtr,
                                                                                            dstPtr,
                                                                                            srcSize.height,
                                                                                            srcSize.width,
                                                                                            channel
                                                                                            );
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "convert_bit_depth.cpp", "convert_bit_depth_u8s16", vld, vgd, "")(srcPtr,
                                                                                            dstPtr,
                                                                                            srcSize.height,
                                                                                            srcSize.width,
                                                                                            channel
                                                                                            );
    }
    
    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);
    
    return RPP_SUCCESS;    
}
template <typename T, typename U>
RppStatus
convert_bit_depth_hip_batch (   T* srcPtr, U* dstPtr, 
                            Rpp32u type,rpp::Handle& handle,
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
    if (type == 1)
    {
        handle.AddKernel("", "", "convert_bit_depth.cpp", "convert_bit_depth_batch_u8s8", vld, vgd, "")(srcPtr, dstPtr,
                                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                        channel,
                                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                                        plnpkdind
                                                                                        );
    }
    else if (type == 2)
    {
        handle.AddKernel("", "", "convert_bit_depth.cpp", "convert_bit_depth_batch_u8u16", vld, vgd, "")(srcPtr, dstPtr,
                                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                        channel,
                                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                                        plnpkdind
                                                                                        );
    }  
    else
    {
        handle.AddKernel("", "", "convert_bit_depth.cpp", "convert_bit_depth_batch_u8s16", vld, vgd, "")(srcPtr, dstPtr,
                                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                        channel,
                                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                                        plnpkdind
                                                                                        );
    }  
    
    return RPP_SUCCESS;
}


/********************** laplacian_image_pyramid ************************/

RppStatus
laplacian_image_pyramid_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr,
                            Rpp32f stdDev, Rpp32u kernelSize, RppiChnFormat chnFormat,
                            unsigned int channel, rpp::Handle& handle)
{
    Rpp32f *kernelMain = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
    generate_gaussian_kernel_gpu(stdDev, kernelMain, kernelSize);
    Rpp32f* kernel;
    hipMalloc(&kernel,  kernelSize * kernelSize * sizeof(Rpp32f));
    hipMemcpy(kernel,kernelMain,kernelSize * kernelSize * sizeof(Rpp32f), hipMemcpyHostToDevice);
    Rpp8u* srcPtr1;
    hipMalloc(&srcPtr1,  srcSize.height * srcSize.width * channel * sizeof(Rpp8u));

    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "gaussian_image_pyramid.cpp", "gaussian_image_pyramid_pkd", vld, vgd, "")(srcPtr,
                                                                                            srcPtr1,
                                                                                            srcSize.height,
                                                                                            srcSize.width,
                                                                                            channel,
                                                                                            kernel,
                                                                                            kernelSize,
                                                                                            kernelSize
                                                                                            );
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "gaussian_image_pyramid.cpp", "gaussian_image_pyramid_pln", vld, vgd, "")(srcPtr,
                                                                                            srcPtr1,
                                                                                            srcSize.height,
                                                                                            srcSize.width,
                                                                                            channel,
                                                                                            kernel,
                                                                                            kernelSize,
                                                                                            kernelSize
                                                                                            );
    }

    // if(chnFormat == RPPI_CHN_PACKED)
    // {
    //     CreateProgramFromBinary(handle.GetStream(),"gaussian_image_pyramid.cpp","gaussian_image_pyramid.cpp.bin","gaussian_image_pyramid_pkd",theProgram,theKernel);
    //     clRetainKernel(theKernel);
    // }
    // else
    // {
    //     CreateProgramFromBinary(handle.GetStream(),"gaussian_image_pyramid.cpp","gaussian_image_pyramid.cpp.bin","gaussian_image_pyramid_pln",theProgram,theKernel);
    //     clRetainKernel(theKernel);
    // }

    // //---- Args Setter
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr1);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &kernel);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);

    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "laplacian_image_pyramid.cpp", "laplacian_image_pyramid_pkd", vld, vgd, "")(srcPtr1,
                                                                                            dstPtr,
                                                                                            srcSize.height,
                                                                                            srcSize.width,
                                                                                            channel,
                                                                                            kernel,
                                                                                            kernelSize,
                                                                                            kernelSize
                                                                                            );
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "laplacian_image_pyramid.cpp", "laplacian_image_pyramid_pln", vld, vgd, "")(srcPtr1,
                                                                                            dstPtr,
                                                                                            srcSize.height,
                                                                                            srcSize.width,
                                                                                            channel,
                                                                                            kernel,
                                                                                            kernelSize,
                                                                                            kernelSize
                                                                                            );
    }

    // if(chnFormat == RPPI_CHN_PACKED)
    // {
    //     CreateProgramFromBinary(handle.GetStream(),"laplacian_image_pyramid.cpp","laplacian_image_pyramid.cpp.bin","laplacian_image_pyramid_pkd",theProgram,theKernel);
    //     clRetainKernel(theKernel);
    // }
    // else
    // {
    //     CreateProgramFromBinary(handle.GetStream(),"laplacian_image_pyramid.cpp","laplacian_image_pyramid.cpp.bin","laplacian_image_pyramid_pln",theProgram,theKernel);
    //     clRetainKernel(theKernel);
    // }
    // ctr=0;
    // //---- Args Setter
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr1);
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &kernel);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);
    
    // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);


    return RPP_SUCCESS;  
}
RppStatus
laplacian_image_pyramid_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{
    unsigned int maxHeight, maxWidth, maxKernelSize;
    maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    maxKernelSize = handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[0];
    for(int i = 0 ; i < handle.GetBatchSize() ; i++)
    {
        if(maxHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if(maxWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        if(maxKernelSize < handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i])
            maxKernelSize = handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i];
    }

    unsigned long batchIndex = 0;
    
    Rpp32f *kernelMain = (Rpp32f *)calloc(maxKernelSize * maxKernelSize, sizeof(Rpp32f));
    
    Rpp8u* srcPtr1;
    hipMalloc(&srcPtr1,  maxHeight * maxWidth * channel * sizeof(Rpp8u));
    Rpp32f* kernel;
    hipMalloc(&kernel,  maxKernelSize * maxKernelSize * sizeof(Rpp32f));

    for(int i = 0 ; i < handle.GetBatchSize(); i++)
    {       
        generate_gaussian_kernel_gpu(handle.GetInitHandle()->mem.mcpu.floatArr[0].floatmem[i], kernelMain, handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i]);
        hipMemcpy(kernel,kernelMain,handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i] * handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i] * sizeof(Rpp32f),hipMemcpyHostToDevice);
        if(chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{handle.GetInitHandle()->mem.mgpu.csrcSize.width[i], handle.GetInitHandle()->mem.mgpu.csrcSize.height[i], channel};
            handle.AddKernel("", "", "laplacian_image_pyramid.cpp", "gaussian_image_pyramid_pkd_batch", vld, vgd, "")(srcPtr,srcPtr1,
                                                                                                                maxHeight,
                                                                                                                maxWidth,
                                                                                                                channel,
                                                                                                                kernel,
                                                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                                                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                                                                                                                batchIndex);
        }
        else
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{handle.GetInitHandle()->mem.mgpu.csrcSize.width[i], handle.GetInitHandle()->mem.mgpu.csrcSize.height[i], channel};
            handle.AddKernel("", "", "laplacian_image_pyramid.cpp", "gaussian_image_pyramid_pln_batch", vld, vgd, "")(srcPtr,srcPtr1,
                                                                                                                maxHeight,
                                                                                                                maxWidth,
                                                                                                                channel,
                                                                                                                kernel,
                                                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                                                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                                                                                                                batchIndex);
        }

        if(chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{handle.GetInitHandle()->mem.mgpu.csrcSize.width[i], handle.GetInitHandle()->mem.mgpu.csrcSize.height[i], channel};
            handle.AddKernel("", "", "laplacian_image_pyramid.cpp", "laplacian_image_pyramid_pkd_batch", vld, vgd, "")(srcPtr1,dstPtr,
                                                                                                                maxHeight,
                                                                                                                maxWidth,
                                                                                                                channel,
                                                                                                                kernel,
                                                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                                                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                                                                                                                batchIndex);
        }
        else
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{handle.GetInitHandle()->mem.mgpu.csrcSize.width[i], handle.GetInitHandle()->mem.mgpu.csrcSize.height[i], channel};
            handle.AddKernel("", "", "laplacian_image_pyramid.cpp", "laplacian_image_pyramid_pln_batch", vld, vgd, "")(srcPtr1,dstPtr,
                                                                                                                maxHeight,
                                                                                                                maxWidth,
                                                                                                                channel,
                                                                                                                kernel,
                                                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                                                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                                                                                                                batchIndex);
        }
        batchIndex += maxHeight * maxWidth * channel;
    }
    hipFree(srcPtr1);
    hipFree(kernel);

    return RPP_SUCCESS;  
}

// /********************** canny_edge_detector ************************/


RppStatus
canny_edge_detector_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp8u minThreshold,
                        Rpp8u maxThreshold, RppiChnFormat chnFormat, unsigned int channel,
                        rpp::Handle& handle)
{
    Rpp8u* gsin;
    hipMalloc(&gsin, sizeof(unsigned char) * srcSize.height * srcSize.width);
    Rpp8u* gsout;
    hipMalloc(&gsout, sizeof(unsigned char) * srcSize.height * srcSize.width);
    
    Rpp8u* tempDest1;
    hipMalloc(&tempDest1, sizeof(unsigned char) * srcSize.height * srcSize.width);
    Rpp8u* tempDest2;
    hipMalloc(&tempDest2, sizeof(unsigned char) * srcSize.height * srcSize.width);

    Rpp8u* sobelX;
    hipMalloc(&sobelX, sizeof(unsigned char) * srcSize.height * srcSize.width);
    Rpp8u* sobelY;
    hipMalloc(&sobelY, sizeof(unsigned char) * srcSize.height * srcSize.width);
    
    int ctr;
           
    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = 1;
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
    if(channel == 3)
    {
        if(chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "canny_edge_detector.cpp", "ced_pkd3_to_pln1", vld, vgd, "")(srcPtr,
                                                                                                gsin,
                                                                                                srcSize.height,
                                                                                                srcSize.width,
                                                                                                channel);
            // CreateProgramFromBinary(handle.GetStream(),"canny_edge_detector.cpp","canny_edge_detector.cpp.bin","ced_pkd3_to_pln1",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        else
        {
            handle.AddKernel("", "", "canny_edge_detector.cpp", "ced_pln3_to_pln1", vld, vgd, "")(srcPtr,
                                                                                                gsin,
                                                                                                srcSize.height,
                                                                                                srcSize.width,
                                                                                                channel);
            // CreateProgramFromBinary(handle.GetStream(),"canny_edge_detector.cpp","canny_edge_detector.cpp.bin","ced_pln3_to_pln1",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        
        // ctr = 0;
        
        // //---- Args Setter
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &gsin);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);

        // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);
    }
    
    
    
    unsigned int sobelType = 2;
    unsigned int sobelTypeX = 0;
    unsigned int sobelTypeY = 1;
    unsigned int newChannel = 1;
    
    // ctr = 0;
    
    //---- Args Setter
    if(channel == 1)
    {
        handle.AddKernel("", "", "sobel.cpp", "sobel_pln", vld, vgd, "")(srcPtr,
                                                                        tempDest1,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        newChannel,
                                                                        sobelType);
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
        // CreateProgramFromBinary(handle.GetStream(),"sobel.cpp","sobel.cpp.bin","sobel_pln",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    else
    {
        handle.AddKernel("", "", "sobel.cpp", "sobel_pln", vld, vgd, "")(gsin,
                                                                        tempDest1,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        newChannel,
                                                                        sobelType);
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &gsin);
        // CreateProgramFromBinary(handle.GetStream(),"sobel.cpp","sobel.cpp.bin","sobel_pln",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &tempDest1);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &newChannel);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &sobelType);
    
    // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);

    // ctr = 0;
    
    //---- Args Setter
    if(channel == 1)
    {
        handle.AddKernel("", "", "sobel.cpp", "sobel_pln", vld, vgd, "")(srcPtr,
                                                                        sobelX,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        newChannel,
                                                                        sobelTypeX);
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
    }
    else
    {
        handle.AddKernel("", "", "sobel.cpp", "sobel_pln", vld, vgd, "")(gsin,
                                                                        sobelX,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        newChannel,
                                                                        sobelTypeX);
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &gsin);
    }
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &sobelX);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &newChannel);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &sobelTypeX);
    
    // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);

    // ctr = 0;
    
    //---- Args Setter
    if(channel == 1)
    {
        handle.AddKernel("", "", "sobel.cpp", "sobel_pln", vld, vgd, "")(srcPtr,
                                                                        sobelY,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        newChannel,
                                                                        sobelTypeY);
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
    }
    else
    {
        handle.AddKernel("", "", "sobel.cpp", "sobel_pln", vld, vgd, "")(gsin,
                                                                        sobelY,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        newChannel,
                                                                        sobelTypeY);
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &gsin);
    }
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &sobelY);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &newChannel);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &sobelTypeY);
    
    // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);
    handle.AddKernel("", "", "canny_edge_detector.cpp", "ced_non_max_suppression", vld, vgd, "")(tempDest1,
                                                                                sobelX,
                                                                                sobelY,
                                                                                tempDest2,
                                                                                srcSize.height,
                                                                                srcSize.width,
                                                                                newChannel,
                                                                                minThreshold,
                                                                                maxThreshold);
    // CreateProgramFromBinary(handle.GetStream(),"canny_edge_detector.cpp","canny_edge_detector.cpp.bin","ced_non_max_suppression",theProgram,theKernel);
    // clRetainKernel(theKernel);
    
    // ctr = 0;
    
    //---- Args Setter
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &tempDest1);
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &sobelX);
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &sobelY);
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &tempDest2);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &newChannel);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned char), &minThreshold);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned char), &maxThreshold);
    
    // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);
    
    // ctr = 0;
    
    //---- Args Setter
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &tempDest2);
    if(channel == 1)
    {
        handle.AddKernel("", "", "canny_edge_detector.cpp", "canny_edge", vld, vgd, "")(tempDest2,
                                                                                dstPtr,
                                                                                srcSize.height,
                                                                                srcSize.width,
                                                                                newChannel,
                                                                                minThreshold,
                                                                                maxThreshold);
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
        // CreateProgramFromBinary(handle.GetStream(),"canny_edge_detector.cpp","canny_edge_detector.cpp.bin","canny_edge",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    else
    {
        handle.AddKernel("", "", "canny_edge_detector.cpp", "canny_edge", vld, vgd, "")(tempDest2,
                                                                                gsout,
                                                                                srcSize.height,
                                                                                srcSize.width,
                                                                                newChannel,
                                                                                minThreshold,
                                                                                maxThreshold);
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &gsout);
        // CreateProgramFromBinary(handle.GetStream(),"canny_edge_detector.cpp","canny_edge_detector.cpp.bin","canny_edge",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &newChannel);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned char), &minThreshold);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned char), &maxThreshold);
    
    // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);
    
    if(channel == 3)
    {
        if(chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "canny_edge_detector.cpp", "ced_pln1_to_pkd3", vld, vgd, "")(gsout,
                                                                                dstPtr,
                                                                                srcSize.height,
                                                                                srcSize.width,
                                                                                channel);
            // CreateProgramFromBinary(handle.GetStream(),"canny_edge_detector.cpp","canny_edge_detector.cpp.bin","ced_pln1_to_pkd3",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        else
        {
            handle.AddKernel("", "", "canny_edge_detector.cpp", "ced_pln1_to_pln3", vld, vgd, "")(gsout,
                                                                                dstPtr,
                                                                                srcSize.height,
                                                                                srcSize.width,
                                                                                channel);
            // CreateProgramFromBinary(handle.GetStream(),"canny_edge_detector.cpp","canny_edge_detector.cpp.bin","ced_pln1_to_pln3",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        
        // ctr = 0;
        
        // //---- Args Setter
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &gsout);
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
        
        // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);
    }
    hipFree( gsin );
    hipFree( gsout );
    
    hipFree( tempDest1 );
    hipFree( tempDest2 );

    hipFree( sobelX );
    hipFree( sobelY );

    return RPP_SUCCESS;    
}

RppStatus
canny_edge_detector_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
                            RppiChnFormat chnFormat, unsigned int channel)
{
    unsigned int maxHeight, maxWidth;
    maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    for(int i = 0 ; i < handle.GetBatchSize() ; i++)
    {
        if(maxHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if(maxWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
    }

    Rpp32u imageDim = maxHeight * maxWidth;

    Rpp8u* gsin;
    hipMalloc(&gsin, sizeof(unsigned char) * imageDim);
    Rpp8u* gsout;
    hipMalloc(&gsout, sizeof(unsigned char) * imageDim);
    
    Rpp8u* tempDest1;
    hipMalloc(&tempDest1, sizeof(unsigned char) * imageDim);
    Rpp8u* tempDest2;
    hipMalloc(&tempDest2, sizeof(unsigned char) * imageDim);

    Rpp8u* sobelX;
    hipMalloc(&sobelX, sizeof(unsigned char) * imageDim);
    Rpp8u* sobelY;
    hipMalloc(&sobelY, sizeof(unsigned char) * imageDim);
    
    unsigned long batchIndex = 0;
    Rpp8u* srcPtr1;
    hipMalloc(&srcPtr1, sizeof(unsigned char) * imageDim * channel);
    Rpp8u* dstPtr1;
    hipMalloc(&dstPtr1, sizeof(unsigned char) * imageDim * channel);

    // int ctr;
    for(int i = 0 ; i < handle.GetBatchSize() ; i++)
    {       
        hipMemcpy(srcPtr1, srcPtr + batchIndex, sizeof(unsigned char) * imageDim * channel, hipMemcpyHostToDevice);
        size_t gDim3[3];
        gDim3[0] = maxWidth;
        gDim3[1] = maxHeight;
        gDim3[2] = 1;
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
        if(channel == 3)
        {
            if(chnFormat == RPPI_CHN_PACKED)
            {
                handle.AddKernel("", "", "canny_edge_detector.cpp", "canny_ced_pkd3_to_pln1", vld, vgd, "")(srcPtr1,
                                                                                                gsin,
                                                                                                maxHeight,
                                                                                                maxWidth,
                                                                                                channel);
            }
            else
            {
                handle.AddKernel("", "", "canny_edge_detector.cpp", "canny_ced_pln3_to_pln1", vld, vgd, "")(srcPtr1,
                                                                                                gsin,
                                                                                                maxHeight,
                                                                                                maxWidth,
                                                                                                channel);
            }
        }
        unsigned int sobelType = 2;
        unsigned int sobelTypeX = 0;
        unsigned int sobelTypeY = 1;
        unsigned int newChannel = 1;
        if(channel == 1)
        {    
            handle.AddKernel("", "", "sobel.cpp", "sobel_pln", vld, vgd, "")(srcPtr1,
                                                                        tempDest1,
                                                                        maxHeight,
                                                                        maxWidth,
                                                                        newChannel,
                                                                        sobelType);
        }
        else
        {
            handle.AddKernel("", "", "sobel.cpp", "sobel_pln", vld, vgd, "")(gsin,
                                                                        tempDest1,
                                                                        maxHeight,
                                                                        maxWidth,
                                                                        newChannel,
                                                                        sobelType);
        }
        if(channel == 1)
        {    
            handle.AddKernel("", "", "sobel.cpp", "sobel_pln", vld, vgd, "")(srcPtr1,
                                                                        sobelX,
                                                                        maxHeight,
                                                                        maxWidth,
                                                                        newChannel,
                                                                        sobelTypeX);
        }
        else
        {
            handle.AddKernel("", "", "sobel.cpp", "sobel_pln", vld, vgd, "")(gsin,
                                                                        sobelX,
                                                                        maxHeight,
                                                                        maxWidth,
                                                                        newChannel,
                                                                        sobelTypeX);
        }
        if(channel == 1)
        {    
            handle.AddKernel("", "", "sobel.cpp", "sobel_pln", vld, vgd, "")(srcPtr1,
                                                                        sobelY,
                                                                        maxHeight,
                                                                        maxWidth,
                                                                        newChannel,
                                                                        sobelTypeY);
        }
        else
        {
            handle.AddKernel("", "", "sobel.cpp", "sobel_pln", vld, vgd, "")(gsin,
                                                                        sobelY,
                                                                        maxHeight,
                                                                        maxWidth,
                                                                        newChannel,
                                                                        sobelTypeY);
        }
        
        handle.AddKernel("", "", "canny_edge_detector.cpp", "ced_non_max_suppression", vld, vgd, "")(tempDest1,
                                                                                sobelX,
                                                                                sobelY,
                                                                                tempDest2,
                                                                                maxHeight,
                                                                                maxWidth,
                                                                                newChannel,
                                                                                handle.GetInitHandle()->mem.mcpu.ucharArr[0].ucharmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem[i]);
        if(channel == 1)
        {
            handle.AddKernel("", "", "canny_edge_detector.cpp", "canny_edge", vld, vgd, "")(tempDest2,
                                                                                dstPtr1,
                                                                                maxHeight,
                                                                                maxWidth,
                                                                                newChannel,
                                                                                handle.GetInitHandle()->mem.mcpu.ucharArr[0].ucharmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem[i]);
        }
        else
        {
            handle.AddKernel("", "", "canny_edge_detector.cpp", "canny_edge", vld, vgd, "")(tempDest2,
                                                                                gsout,
                                                                                maxHeight,
                                                                                maxWidth,
                                                                                newChannel,
                                                                                handle.GetInitHandle()->mem.mcpu.ucharArr[0].ucharmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem[i]);
        }
        
        if(channel == 3)
        {
            if(chnFormat == RPPI_CHN_PACKED)
            {
                handle.AddKernel("", "", "canny_edge_detector.cpp", "canny_ced_pln1_to_pkd3", vld, vgd, "")(gsout,
                                                                                dstPtr1,
                                                                                maxHeight,
                                                                                maxWidth,
                                                                                channel);
            }
            else
            {
                handle.AddKernel("", "", "canny_edge_detector.cpp", "canny_ced_pln1_to_pln3", vld, vgd, "")(gsout,
                                                                                dstPtr1,
                                                                                maxHeight,
                                                                                maxWidth,
                                                                                channel);
            }
        }
        hipMemcpy(dstPtr + batchIndex, dstPtr1, sizeof(unsigned char) * imageDim * channel, hipMemcpyDeviceToHost);
        batchIndex += imageDim * channel;
    }
    return RPP_SUCCESS;    
}

// /********************** harris corner detector ************************/
RppStatus
harris_corner_detector_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr,
 Rpp32u gaussianKernelSize, Rpp32f stdDev, Rpp32u kernelSize, Rpp32f kValue,
 Rpp32f threshold, Rpp32u nonmaxKernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    /* SETTING UP */
    Rpp8u* gsin;
    hipMalloc(&gsin, sizeof(unsigned char) * srcSize.height * srcSize.width);
    Rpp8u* gsout;
    hipMalloc(&gsout, sizeof(unsigned char) * srcSize.height * srcSize.width);

    Rpp8u* tempDest1;
    hipMalloc(&tempDest1, sizeof(unsigned char) * srcSize.height * srcSize.width);
    Rpp8u* tempDest2;
    hipMalloc(&tempDest2, sizeof(unsigned char) * srcSize.height * srcSize.width);

    Rpp32f* dstFloat;
    hipMalloc(&dstFloat, sizeof(float) * srcSize.height * srcSize.width);
    Rpp32f* nonMaxDstFloat;
    hipMalloc(&nonMaxDstFloat, sizeof(float) * srcSize.height * srcSize.width);

    Rpp8u* sobelX;
    hipMalloc(&sobelX, sizeof(unsigned char) * srcSize.height * srcSize.width);
    Rpp8u* sobelY;
    hipMalloc(&sobelY, sizeof(unsigned char) * srcSize.height * srcSize.width);

    // int ctr;
           
    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = 1;
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
    /* RGB to GREY SCALE */

    if(channel == 3)
    {
        if(chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "fast_corner_detector.cpp", "ced_pkd3_to_pln1", vld, vgd, "")(srcPtr,
                                                                                                gsin,
                                                                                                srcSize.height,
                                                                                                srcSize.width,
                                                                                                channel);
            // CreateProgramFromBinary(handle.GetStream(),"canny_edge_detector.cpp","canny_edge_detector.cpp.bin","ced_pkd3_to_pln1",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        else
        {
            handle.AddKernel("", "", "fast_corner_detector.cpp", "ced_pln3_to_pln1", vld, vgd, "")(srcPtr,
                                                                                                gsin,
                                                                                                srcSize.height,
                                                                                                srcSize.width,
                                                                                                channel);
            // CreateProgramFromBinary(handle.GetStream(),"canny_edge_detector.cpp","canny_edge_detector.cpp.bin","ced_pln3_to_pln1",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        
        // ctr = 0;
        // //---- Args Setter
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &gsin);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);

        // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);
    }

    unsigned int newChannel = 1;

    
    /* GAUSSIAN FILTER */

    Rpp32f *kernelMain = (Rpp32f *)calloc(gaussianKernelSize * gaussianKernelSize, sizeof(Rpp32f));
    generate_gaussian_kernel_gpu(stdDev, kernelMain, gaussianKernelSize);
    Rpp32f* kernel;
    hipMalloc(&kernel,  gaussianKernelSize * gaussianKernelSize * sizeof(Rpp32f));
    hipMemcpy(kernel, kernelMain,gaussianKernelSize * gaussianKernelSize * sizeof(Rpp32f),hipMemcpyHostToDevice);
    // CreateProgramFromBinary(handle.GetStream(),"gaussian.cpp","gaussian.cpp.bin","gaussian_pln",theProgram,theKernel);
    // clRetainKernel(theKernel);

    // ctr = 0;
    
    //---- Args Setter
    if(channel == 1)
    {
        handle.AddKernel("", "", "gaussian_filter.cpp", "gaussian_pln", vld, vgd, "")(srcPtr,
                                                                            tempDest1,
                                                                            srcSize.height,
                                                                            srcSize.width,
                                                                            newChannel,
                                                                            kernel,
                                                                            gaussianKernelSize,
                                                                            gaussianKernelSize);
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
    }
    else
    {
        handle.AddKernel("", "", "gaussian_filter.cpp", "gaussian_pln", vld, vgd, "")(gsin,
                                                                            tempDest1,
                                                                            srcSize.height,
                                                                            srcSize.width,
                                                                            newChannel,
                                                                            kernel,
                                                                            gaussianKernelSize,
                                                                            gaussianKernelSize);
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &gsin);
    }
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &tempDest1);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &newChannel);
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &kernel);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &gaussianKernelSize);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &gaussianKernelSize);
    // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);

    
    /* SOBEL X and Y */
    // CreateProgramFromBinary(handle.GetStream(),"sobel.cpp","sobel.cpp.bin","sobel_pln",theProgram,theKernel);
    // clRetainKernel(theKernel);

    unsigned int sobelType = 2;
    unsigned int sobelTypeX = 0;
    unsigned int sobelTypeY = 1;
    handle.AddKernel("", "", "sobel.cpp", "sobel_pln", vld, vgd, "")(tempDest1,
                                                                    sobelX,
                                                                    srcSize.height,
                                                                    srcSize.width,
                                                                    newChannel,
                                                                    sobelTypeX);
    // ctr = 0;
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &tempDest1);
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &sobelX);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &newChannel);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &sobelTypeX);    
    // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);
    handle.AddKernel("", "", "sobel.cpp", "sobel_pln", vld, vgd, "")(tempDest1,
                                                                    sobelY,
                                                                    srcSize.height,
                                                                    srcSize.width,
                                                                    newChannel,
                                                                    sobelTypeY);
    // ctr = 0;
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &tempDest1);
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &sobelY);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &newChannel);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &sobelTypeY);
    
    // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);

    
    /* HARRIS CORNER STRENGTH MATRIX */
    handle.AddKernel("", "", "harris_corner_detector.cpp", "harris_corner_detector_strength", vld, vgd, "")(sobelX,
                                                                                                           sobelY,
                                                                                                           dstFloat,
                                                                                                           srcSize.height,
                                                                                                           srcSize.width,
                                                                                                           newChannel,
                                                                                                           kernelSize,
                                                                                                           kValue,
                                                                                                           threshold);
    // CreateProgramFromBinary(handle.GetStream(),"harris_corner_detector.cpp","harris_corner_detector.cpp.bin","harris_corner_detector_strength",theProgram,theKernel);
    // clRetainKernel(theKernel);

    // ctr = 0;
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &sobelX);
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &sobelY);
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstFloat);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &newChannel);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);
    // clSetKernelArg(theKernel, ctr++, sizeof(float), &kValue);
    // clSetKernelArg(theKernel, ctr++, sizeof(float), &threshold);

    // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);


    /* NON-MAX SUPRESSION */

    handle.AddKernel("", "", "harris_corner_detector.cpp", "harris_corner_detector_nonmax_supression", vld, vgd, "")(dstFloat,
                                                                                                           nonMaxDstFloat,
                                                                                                           srcSize.height,
                                                                                                           srcSize.width,
                                                                                                           newChannel,
                                                                                                           nonmaxKernelSize);
    // CreateProgramFromBinary(handle.GetStream(),"harris_corner_detector.cpp","harris_corner_detector.cpp.bin","harris_corner_detector_nonmax_supression",theProgram,theKernel);
    // clRetainKernel(theKernel);
    // ctr = 0;
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstFloat);
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &nonMaxDstFloat);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &newChannel);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &nonmaxKernelSize);
    // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);
    hipMemcpy( dstPtr,srcPtr, sizeof(unsigned char) * srcSize.width * srcSize.height * channel,hipMemcpyDeviceToDevice);
    if(chnFormat == RPPI_CHN_PACKED)
    {
        handle.AddKernel("", "", "harris_corner_detector.cpp", "harris_corner_detector_pkd", vld, vgd, "")(dstPtr,
                                                                                                          nonMaxDstFloat,
                                                                                                          srcSize.height,
                                                                                                          srcSize.width,
                                                                                                          channel);
        // CreateProgramFromBinary(handle.GetStream(),"harris_corner_detector.cpp","harris_corner_detector.cpp.bin","harris_corner_detector_pkd",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    else
    {
        handle.AddKernel("", "", "harris_corner_detector.cpp", "harris_corner_detector_pkd", vld, vgd, "")(dstPtr,
                                                                                                          nonMaxDstFloat,
                                                                                                          srcSize.height,
                                                                                                          srcSize.width,
                                                                                                          channel);
        // CreateProgramFromBinary(handle.GetStream(),"harris_corner_detector.cpp","harris_corner_detector.cpp.bin","harris_corner_detector_pln",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }

    // ctr = 0;
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &nonMaxDstFloat);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);

    // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);

    
    return RPP_SUCCESS;
}

RppStatus
harris_corner_detector_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr,rpp::Handle& handle,
                                RppiChnFormat chnFormat, unsigned int channel)
{
    /* SETTING UP */

    unsigned int maxHeight, maxWidth, maxKernelSize;
    unsigned long ioBufferSize = 0, singleImageSize = 0;
    maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    maxKernelSize = handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[0];
    for (int i = 0; i < handle.GetBatchSize(); i++)
    {
        if (maxHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if (maxWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        if (maxKernelSize < handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i])
            maxKernelSize = handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i];
    }

    ioBufferSize = maxHeight * maxWidth * channel * handle.GetBatchSize();
    singleImageSize = maxHeight * maxWidth * channel;

    Rpp32f *kernelMain = (Rpp32f *)calloc(maxKernelSize * maxKernelSize, sizeof(Rpp32f));

    Rpp32f* kernel;
    hipMalloc(&kernel,  maxKernelSize * maxKernelSize * sizeof(Rpp32f));

    Rpp8u* gsin;
    hipMalloc(&gsin, sizeof(unsigned char) * maxHeight * maxWidth);
    Rpp8u* gsout;
    hipMalloc(&gsout, sizeof(unsigned char) * maxHeight * maxWidth);
    
    Rpp8u* tempDest1;
    hipMalloc(&tempDest1, sizeof(unsigned char) * maxHeight * maxWidth);
    Rpp8u* tempDest2;
    hipMalloc(&tempDest2, sizeof(unsigned char) * maxHeight * maxWidth);

    Rpp8u* sobelX;
    hipMalloc(&sobelX, sizeof(unsigned char) * maxHeight * maxWidth);
    Rpp8u* sobelY;
    hipMalloc(&sobelY, sizeof(unsigned char) * maxHeight * maxWidth);

    Rpp32f* dstFloat;
    hipMalloc(&dstFloat, sizeof(float) * maxHeight * maxWidth);
    Rpp32f* nonMaxDstFloat;
    hipMalloc(&nonMaxDstFloat, sizeof(float) * maxHeight * maxWidth);

    hipMemcpy(dstPtr, srcPtr, sizeof(unsigned char) * ioBufferSize, hipMemcpyDeviceToDevice);

    unsigned long batchIndex = 0;
    Rpp8u *srcPtr1, *dstPtr1;
    hipMalloc(&srcPtr1, sizeof(Rpp8u) * singleImageSize);
    hipMalloc(&dstPtr1, sizeof(Rpp8u) * singleImageSize);

    size_t gDim3[3];

    for(int i = 0 ; i < handle.GetBatchSize() ; i++)       
    {        
        hipMemcpy(srcPtr1, srcPtr + batchIndex, sizeof(unsigned char) * singleImageSize, hipMemcpyDeviceToDevice);
        gDim3[0] = maxWidth;
        gDim3[1] = maxHeight;
        gDim3[2] = 1;
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
        if(channel == 3)
        {
            if(chnFormat == RPPI_CHN_PACKED)
            {
                handle.AddKernel("", "", "fast_corner_detector.cpp", "ced_pkd3_to_pln1", vld, vgd, "")(srcPtr1,
                                                                                                gsin,
                                                                                                maxHeight,
                                                                                                maxWidth,
                                                                                                channel);
            }
            else
            {
                handle.AddKernel("", "", "fast_corner_detector.cpp", "ced_pln3_to_pln1", vld, vgd, "")(srcPtr1,
                                                                                                gsin,
                                                                                                maxHeight,
                                                                                                maxWidth,
                                                                                                channel);
            }
        }

        unsigned int newChannel = 1;

        /* GAUSSIAN FILTER */

        generate_gaussian_kernel_gpu(handle.GetInitHandle()->mem.mcpu.floatArr[1].floatmem[i], kernelMain, handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i]);    
        hipMemcpy(kernel, kernelMain, handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i] * handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i] * sizeof(Rpp32f),hipMemcpyHostToDevice);

        if(channel == 1)
        {    
            handle.AddKernel("", "", "gaussian_filter.cpp", "gaussian_pln", vld, vgd, "")(srcPtr1,
                                                                            tempDest1,
                                                                            maxHeight,
                                                                            maxWidth,
                                                                            newChannel,
                                                                            kernel,
                                                                            handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i],
                                                                            handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i]);
        }
        else
        {
            handle.AddKernel("", "", "gaussian_filter.cpp", "gaussian_pln", vld, vgd, "")(gsin,
                                                                            tempDest1,
                                                                            maxHeight,
                                                                            maxWidth,
                                                                            newChannel,
                                                                            kernel,
                                                                            handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i],
                                                                            handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i]);
        }
        
        /* SOBEL X and Y */
        
        unsigned int sobelType = 2;
        unsigned int sobelTypeX = 0;
        unsigned int sobelTypeY = 1;
        handle.AddKernel("", "", "sobel.cpp", "sobel_pln", vld, vgd, "")(tempDest1,
                                                                sobelX,
                                                                maxHeight,
                                                                maxWidth,
                                                                newChannel,
                                                                sobelTypeX);
        handle.AddKernel("", "", "sobel.cpp", "sobel_pln", vld, vgd, "")(tempDest1,
                                                                    sobelY,
                                                                    maxHeight,
                                                                    maxWidth,
                                                                    newChannel,
                                                                    sobelTypeY);
        
        /* HARRIS CORNER STRENGTH MATRIX */

        handle.AddKernel("", "", "harris_corner_detector.cpp", "harris_corner_detector_strength", vld, vgd, "")(sobelX,
                                                                                                           sobelY,
                                                                                                           dstFloat,
                                                                                                           maxHeight,
                                                                                                           maxWidth,
                                                                                                           newChannel,
                                                                                                           handle.GetInitHandle()->mem.mcpu.uintArr[2].uintmem[i],
                                                                                                           handle.GetInitHandle()->mem.mcpu.floatArr[3].floatmem[i],
                                                                                                           handle.GetInitHandle()->mem.mcpu.floatArr[4].floatmem[i]);

        /* NON-MAX SUPRESSION */

        handle.AddKernel("", "", "harris_corner_detector.cpp", "harris_corner_detector_nonmax_supression", vld, vgd, "")(dstFloat,
                                                                                                           nonMaxDstFloat,
                                                                                                           maxHeight,
                                                                                                           maxWidth,
                                                                                                           newChannel,
                                                                                                           handle.GetInitHandle()->mem.mcpu.uintArr[5].uintmem[i]);
        
        hipMemcpy(dstPtr1, srcPtr1, sizeof(unsigned char) * singleImageSize, hipMemcpyDeviceToDevice);
        
        if(chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "harris_corner_detector.cpp", "harris_corner_detector_pkd", vld, vgd, "")(dstPtr1,
                                                                                                          nonMaxDstFloat,
                                                                                                          maxHeight,
                                                                                                          maxWidth,
                                                                                                          channel);
        }
        else
        {
            handle.AddKernel("", "", "harris_corner_detector.cpp", "harris_corner_detector_pln", vld, vgd, "")(dstPtr1,
                                                                                                          nonMaxDstFloat,
                                                                                                          maxHeight,
                                                                                                          maxWidth,
                                                                                                          channel);
        }

        hipMemcpy(dstPtr + batchIndex, dstPtr1, sizeof(unsigned char) * singleImageSize, hipMemcpyDeviceToDevice);
        batchIndex += maxHeight * maxWidth * channel;
    }
    return RPP_SUCCESS;
}

/********************** match template ************************/

RppStatus
match_template_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp16u* dstPtr,
                 Rpp8u* templateImage, RppiSize templateImageSize,
                 RppiChnFormat chnFormat,unsigned int channel, rpp::Handle& handle)
{
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, 1};
        handle.AddKernel("", "", "match_template.cpp", "match_template_pkd", vld, vgd, "")(srcPtr,
                                                                                         dstPtr,
                                                                                         srcSize.height,
                                                                                         srcSize.width,
                                                                                         channel,
                                                                                         templateImage,
                                                                                         templateImageSize.height,
                                                                                         templateImageSize.width);
        // CreateProgramFromBinary(handle.GetStream(),"match_template.cpp","match_template.cpp.bin","match_template_pkd",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, 1};
        handle.AddKernel("", "", "match_template.cpp", "match_template_pln", vld, vgd, "")(srcPtr,
                                                                                         dstPtr,
                                                                                         srcSize.height,
                                                                                         srcSize.width,
                                                                                         channel,
                                                                                         templateImage,
                                                                                         templateImageSize.height,
                                                                                         templateImageSize.width);
        // CreateProgramFromBinary(handle.GetStream(),"match_template.cpp","match_template.cpp.bin","match_template_pln",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    
    //---- Args Setter
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &templateImage);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &templateImageSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &templateImageSize.width);
        
    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = 1;
    // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);
    
    return RPP_SUCCESS;    
}
RppStatus
match_template_hip_batch(Rpp8u* srcPtr, RppiSize *srcSize, Rpp16u* dstPtr, Rpp8u* templateImage, 
                        RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{

    unsigned int maxSrcHeight, maxSrcWidth, maxTmpHeight, maxTmpWidth;
    unsigned long ioSrcBufferSize = 0, ioTmpBufferSize = 0;
    maxSrcHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxSrcWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    maxTmpHeight = handle.GetInitHandle()->mem.mgpu.cdstSize.height[0];
    maxTmpWidth = handle.GetInitHandle()->mem.mgpu.cdstSize.width[0];

    for(int i = 0 ; i < handle.GetBatchSize() ; i++)
    {
        if(maxSrcHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxSrcHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if(maxSrcWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxSrcWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        if(maxTmpHeight < handle.GetInitHandle()->mem.mgpu.cdstSize.height[i])
            maxTmpHeight = handle.GetInitHandle()->mem.mgpu.cdstSize.height[i];
        if(maxTmpWidth < handle.GetInitHandle()->mem.mgpu.cdstSize.width[i])
            maxTmpWidth = handle.GetInitHandle()->mem.mgpu.cdstSize.width[i];
        ioSrcBufferSize += handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel;
        ioTmpBufferSize += handle.GetInitHandle()->mem.mgpu.cdstSize.width[i] * handle.GetInitHandle()->mem.mgpu.cdstSize.height[i] * channel;
    }

    Rpp8u* srcPtr1;
    hipMalloc(&srcPtr1, sizeof(unsigned char) * maxSrcHeight * maxSrcWidth * channel);
    Rpp8u* templateImage1;
    hipMalloc(&templateImage1, sizeof(unsigned char) * maxTmpHeight * maxTmpWidth * channel);
    Rpp16u* dstPtr1;
    hipMalloc(&dstPtr1, sizeof(unsigned short) * maxSrcHeight * maxSrcWidth * channel);

    int ctr;

    size_t gDim3[3];

    size_t batchSrcIndex = 0, batchTmpIndex = 0, batchDstIndex = 0;

    for(int i = 0 ; i < handle.GetBatchSize() ; i++)
    {
        gDim3[0] = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        gDim3[1] = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        gDim3[2] = 1;

        hipMemcpy(srcPtr1, srcPtr+batchSrcIndex,sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, hipMemcpyDeviceToDevice);
        hipMemcpy(templateImage1, templateImage+batchTmpIndex,sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.cdstSize.width[i] * handle.GetInitHandle()->mem.mgpu.cdstSize.height[i] * channel, hipMemcpyDeviceToDevice);

        if(chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
            handle.AddKernel("", "", "match_template.cpp", "match_template_pkd", vld, vgd, "")(srcPtr1,
                                                                                              dstPtr1,
                                                                                              handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                              handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                              channel,
                                                                                              templateImage1,
                                                                                              handle.GetInitHandle()->mem.mgpu.cdstSize.height[i],
                                                                                              handle.GetInitHandle()->mem.mgpu.cdstSize.width[i]);
            // CreateProgramFromBinary(handle.GetStream(),"match_template.cpp","match_template.cpp.bin","match_template_pkd",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        else
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
            handle.AddKernel("", "", "match_template.cpp", "match_template_pln", vld, vgd, "")(srcPtr1,
                                                                                              dstPtr1,
                                                                                              handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                              handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                              channel,
                                                                                              templateImage1,
                                                                                              handle.GetInitHandle()->mem.mgpu.cdstSize.height[i],
                                                                                              handle.GetInitHandle()->mem.mgpu.cdstSize.width[i]);
            // CreateProgramFromBinary(handle.GetStream(),"match_template.cpp","match_template.cpp.bin","match_template_pln",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
    
        //---- Args Setter
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr1);
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr1);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.height[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.width[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &templateImage1);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &templateImageSize[i].height);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &templateImageSize[i].width);

        // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);

        hipMemcpy( dstPtr+batchDstIndex,dstPtr1,sizeof(unsigned short) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, hipMemcpyDeviceToDevice);
        batchSrcIndex += handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel * sizeof(unsigned char);
        batchTmpIndex += handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel * sizeof(unsigned char);
        batchDstIndex += handle.GetInitHandle()->mem.mgpu.cdstSize.height[i] * handle.GetInitHandle()->mem.mgpu.cdstSize.width[i] * channel * sizeof(unsigned short);
    
    }
    
    return RPP_SUCCESS;    
}

/********************** fast corner ************************/
RppStatus
fast_corner_detector_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr,
                        Rpp32u numOfPixels, Rpp8u threshold, Rpp32u nonmaxKernelSize,
                        RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    Rpp8u* gsin;
    hipMalloc(&gsin, sizeof(unsigned char) * srcSize.height * srcSize.width);
    Rpp8u* gsout;
    hipMalloc(&gsout, sizeof(unsigned char) * srcSize.height * srcSize.width);
    
    Rpp8u* tempDest1;
    hipMalloc(&tempDest1, sizeof(unsigned char) * srcSize.height * srcSize.width);
    
    int ctr;
           
    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = 1;
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
    /* RGB to GS */
    if(channel == 3)
    {
        if(chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "fast_corner_detector.cpp", "ced_pkd3_to_pln1", vld, vgd, "")(srcPtr,
                                                                                         gsin,
                                                                                         srcSize.height,
                                                                                         srcSize.width,
                                                                                         channel);
            // CreateProgramFromBinary(handle.GetStream(),"canny_edge_detector.cpp","canny_edge_detector.cpp.bin","ced_pkd3_to_pln1",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        else
        {
            handle.AddKernel("", "", "fast_corner_detector.cpp", "ced_pln3_to_pln1", vld, vgd, "")(srcPtr,
                                                                                         gsin,
                                                                                         srcSize.height,
                                                                                         srcSize.width,
                                                                                         channel);
            // CreateProgramFromBinary(handle.GetStream(),"canny_edge_detector.cpp","canny_edge_detector.cpp.bin","ced_pln3_to_pln1",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        
        // ctr = 0;
        
        // //---- Args Setter
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &gsin);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);

        // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);
    }
    
    /* FAST CORNER IMPLEMENTATION */
    
    
    unsigned int newChannel = 1;
    
    ctr = 0;
    
    //---- Args Setter
    if(channel == 1)
    {    
        handle.AddKernel("", "", "fast_corner_detector.cpp", "fast_corner_detector", vld, vgd, "")(srcPtr,
                                                                                                  tempDest1,
                                                                                                  srcSize.height,
                                                                                                  srcSize.width,
                                                                                                  newChannel,
                                                                                                  threshold,
                                                                                                  numOfPixels);
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
        // CreateProgramFromBinary(handle.GetStream(),"fast_corner_detector.cpp","fast_corner_detector.cpp.bin","fast_corner_detector",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    else
    {
        handle.AddKernel("", "", "fast_corner_detector.cpp", "fast_corner_detector", vld, vgd, "")(gsin,
                                                                                                  tempDest1,
                                                                                                  srcSize.height,
                                                                                                  srcSize.width,
                                                                                                  newChannel,
                                                                                                  threshold,
                                                                                                  numOfPixels);
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &gsin);
        // CreateProgramFromBinary(handle.GetStream(),"fast_corner_detector.cpp","fast_corner_detector.cpp.bin","fast_corner_detector",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &tempDest1);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &newChannel);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned char), &threshold);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &numOfPixels);
    // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);

    /* NON MAX SUPRESSION */
    hipMemcpy(dstPtr,srcPtr,sizeof(unsigned char) * srcSize.width * srcSize.height * channel, hipMemcpyDeviceToDevice);
    if(chnFormat == RPPI_CHN_PACKED)
    {    
        handle.AddKernel("", "", "fast_corner_detector.cpp", "fast_corner_detector_nms_pkd", vld, vgd, "")(tempDest1,
                                                                                                            dstPtr,
                                                                                                            srcSize.height,
                                                                                                            srcSize.width,
                                                                                                            newChannel,
                                                                                                            nonmaxKernelSize);
        // CreateProgramFromBinary(handle.GetStream(),"fast_corner_detector.cpp","fast_corner_detector.cpp.bin","fast_corner_detector_nms_pkd",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    else
    {
        handle.AddKernel("", "", "fast_corner_detector.cpp", "fast_corner_detector_nms_pln", vld, vgd, "")(tempDest1,
                                                                                                            dstPtr,
                                                                                                            srcSize.height,
                                                                                                            srcSize.width,
                                                                                                            newChannel,
                                                                                                            nonmaxKernelSize);
        // CreateProgramFromBinary(handle.GetStream(),"fast_corner_detector.cpp","fast_corner_detector.cpp.bin","fast_corner_detector_nms_pln",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    
    // ctr = 0;    
    //---- Args Setter
    // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &tempDest1);
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &newChannel);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &nonmaxKernelSize);
    // /* CODE */
    // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);

    return RPP_SUCCESS;    
}

RppStatus
fast_corner_detector_hip_batch( Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)
    // cl_mem srcPtr, RppiSize *srcSize, cl_mem dstPtr, Rpp32u *numOfPixels, Rpp8u *threshold, Rpp32u *nonmaxKernelSize, Rpp32f handle.GetBatchSize(), RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    unsigned int maxHeight, maxWidth;
    unsigned long ioBufferSize = 0;
    maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    for(int i = 0 ; i < handle.GetBatchSize() ; i++)
    {
        if(maxHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if(maxWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        ioBufferSize += handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel;
    }

    Rpp8u* srcPtr1;
    hipMalloc(&srcPtr1, sizeof(unsigned char) * maxHeight * maxWidth * channel);
    Rpp8u* dstPtr1;
    hipMalloc(&dstPtr1, sizeof(unsigned char) * maxHeight * maxWidth * channel);
     
    Rpp8u* gsin;
    hipMalloc(&gsin, sizeof(unsigned char) * maxHeight * maxWidth);
    Rpp8u* gsout;
    hipMalloc(&gsout, sizeof(unsigned char) * maxHeight * maxWidth);
     
    Rpp8u* tempDest1;
    hipMalloc(&tempDest1, sizeof(unsigned char) * maxHeight * maxWidth);
    
    int ctr;
           
    size_t gDim3[3];

    size_t batchIndex = 0;

    for(int i = 0 ; i < handle.GetBatchSize() ; i++)
    {        
        gDim3[0] = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        gDim3[1] = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        gDim3[2] = 1;
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
        
        hipMemcpy( srcPtr1, srcPtr+batchIndex,sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, hipMemcpyDeviceToDevice);

        /* RGB to GS */
        if(channel == 3)
        {
            if(chnFormat == RPPI_CHN_PACKED)
            {
                handle.AddKernel("", "", "fast_corner_detector.cpp", "ced_pkd3_to_pln1", vld, vgd, "")(srcPtr,
                                                                                         gsin,
                                                                                         handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                         handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                         channel);
                // CreateProgramFromBinary(handle.GetStream(),"fast_corner_detector.cpp","fast_corner_detector.cpp.bin","ced_pkd3_to_pln1",theProgram,theKernel);
                // clRetainKernel(theKernel);
            }
            else
            {
                handle.AddKernel("", "", "fast_corner_detector.cpp", "ced_pln3_to_pln1", vld, vgd, "")(srcPtr,
                                                                                         gsin,
                                                                                         handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                         handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                         channel);
                // CreateProgramFromBinary(handle.GetStream(),"fast_corner_detector.cpp","fast_corner_detector.cpp.bin","ced_pln3_to_pln1",theProgram,theKernel);
                // clRetainKernel(theKernel);
            }
            
            // ctr = 0;            
            // //---- Args Setter
            // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr1);
            // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &gsin);
            // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.height[i]);
            // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.width[i]);
            // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
            // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);
        }
        
        /* FAST CORNER IMPLEMENTATION */
        // CreateProgramFromBinary(handle.GetStream(),"fast_corner_detector.cpp","fast_corner_detector.cpp.bin","fast_corner_detector",theProgram,theKernel);
        // clRetainKernel(theKernel);
        
        unsigned int newChannel = 1;
        
        ctr = 0;
        
        //---- Args Setter
        if(channel == 1)
        {    
            handle.AddKernel("", "", "fast_corner_detector.cpp", "fast_corner_detector", vld, vgd, "")(srcPtr,
                                                                                                  tempDest1,
                                                                                                  handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                                  handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                                  newChannel,
                                                                                                  handle.GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem[i],
                                                                                                  handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i]);
            // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr1);
        }
        else
        {
            handle.AddKernel("", "", "fast_corner_detector.cpp", "fast_corner_detector", vld, vgd, "")(gsin,
                                                                                                  tempDest1,
                                                                                                  handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                                  handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                                  newChannel,
                                                                                                  handle.GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem[i],
                                                                                                  handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i]);
            // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &gsin);
        }
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &tempDest1);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.height[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.width[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &newChannel);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned char), &threshold[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &numOfPixels[i]);

        // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);

        /* NON MAX SUPRESSION */
        hipMemcpy(dstPtr1, srcPtr1, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, hipMemcpyDeviceToDevice);
        if(chnFormat == RPPI_CHN_PACKED)
        {    
            handle.AddKernel("", "", "fast_corner_detector.cpp", "fast_corner_detector_nms_pkd", vld, vgd, "")(tempDest1,
                                                                                                            dstPtr,
                                                                                                            handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                                            handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                                            newChannel,
                                                                                                            handle.GetInitHandle()->mem.mcpu.uintArr[2].uintmem[i]);
            // CreateProgramFromBinary(handle.GetStream(),"fast_corner_detector.cpp","fast_corner_detector.cpp.bin","fast_corner_detector_nms_pkd",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        else
        {
            handle.AddKernel("", "", "fast_corner_detector.cpp", "fast_corner_detector_nms_pln", vld, vgd, "")(tempDest1,
                                                                                                            dstPtr,
                                                                                                            handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                                            handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                                            newChannel,
                                                                                                            handle.GetInitHandle()->mem.mcpu.uintArr[2].uintmem[i]);
            // CreateProgramFromBinary(handle.GetStream(),"fast_corner_detector.cpp","fast_corner_detector.cpp.bin","fast_corner_detector_nms_pln",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        
        // ctr = 0;        
        // //---- Args Setter        
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &tempDest1);
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr1);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.height[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.width[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &newChannel);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &nonmaxhandle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i]);
        // /* CODE */
        // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);
        
        hipMemcpy( dstPtr+batchIndex, dstPtr1,sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, hipMemcpyDeviceToDevice);

        batchIndex += handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel * sizeof(unsigned char);
    }
    return RPP_SUCCESS;    
}


/********************** Reconstruction of laplacian image pyramid ************************/

RppStatus
reconstruction_laplacian_image_pyramid_hip(Rpp8u* srcPtr1, RppiSize srcSize1,
 Rpp8u* srcPtr2, RppiSize srcSize2, Rpp8u* dstPtr, Rpp32f stdDev, Rpp32u kernelSize,
  RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    Rpp8u* gsin;
    hipMalloc(&gsin, sizeof(unsigned char) * srcSize1.height * srcSize1.width * channel);
    Rpp8u* gsout;
    hipMalloc(&gsout, sizeof(unsigned char) * srcSize1.height * srcSize1.width * channel);
    size_t gDim3[3];
    gDim3[0] = srcSize1.width;
    gDim3[1] = srcSize1.height;
    gDim3[2] = channel;
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
    /* Resize the Source 2 */
    if (chnFormat == RPPI_CHN_PLANAR)
    {   
        handle.AddKernel("", "", "resize.cpp", "resize_pln", vld, vgd, "")(srcPtr2,
                                                                          gsin,
                                                                          srcSize2.height,
                                                                          srcSize2.width,
                                                                          srcSize1.height,
                                                                          srcSize1.width,
                                                                          channel);

        // CreateProgramFromBinary(handle.GetStream(),"resize.cpp","resize.cpp.bin","resize_pln",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {   
        handle.AddKernel("", "", "resize.cpp", "resize_pln", vld, vgd, "")(srcPtr2,
                                                                          gsin,
                                                                          srcSize2.height,
                                                                          srcSize2.width,
                                                                          srcSize1.height,
                                                                          srcSize1.width,
                                                                          channel);
        // CreateProgramFromBinary(handle.GetStream(),"resize.cpp","resize.cpp.bin","resize_pkd",theProgram,theKernel);
        // clRetainKernel(theKernel); 
    }

    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr2);
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &gsin);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize2.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize2.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize1.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize1.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);

    /* Gaussian Blur */
    Rpp32f *kernelMain = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
    generate_gaussian_kernel_gpu(stdDev, kernelMain, kernelSize);
    Rpp32f* kernel;
    hipMalloc(&kernel,  kernelSize * kernelSize * sizeof(Rpp32f));
    hipMemcpy(kernel, kernelMain,kernelSize * kernelSize * sizeof(Rpp32f), hipMemcpyHostToDevice);
    // ctr=0;
    if(chnFormat == RPPI_CHN_PACKED)
    {
        handle.AddKernel("", "", "gaussian_filter.cpp", "gaussian_pkd", vld, vgd, "")(gsin,
                                                                              gsout,
                                                                              srcSize1.height,
                                                                              srcSize1.width,
                                                                              channel,
                                                                              kernel,
                                                                              kernelSize,
                                                                              kernelSize);
        // CreateProgramFromBinary(handle.GetStream(),"gaussian.cpp","gaussian.cpp.bin","gaussian_pkd",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    else
    {
        handle.AddKernel("", "", "gaussian_filter.cpp", "gaussian_pln", vld, vgd, "")(gsin,
                                                                              gsout,
                                                                              srcSize1.height,
                                                                              srcSize1.width,
                                                                              channel,
                                                                              kernel,
                                                                              kernelSize,
                                                                              kernelSize);
        // CreateProgramFromBinary(handle.GetStream(),"gaussian.cpp","gaussian.cpp.bin","gaussian_pln",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    
    //---- Args Setter
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &gsin);
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &gsout);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize1.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize1.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &kernel);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);
    // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);

    /* Reconstruction of Laplacian Image pyramid */
    if(chnFormat == RPPI_CHN_PACKED)
    {
        handle.AddKernel("", "", "reconstruction_laplacian_image_pyramid.cpp", "reconstruction_laplacian_image_pyramid_pkd", vld, vgd, "")(srcPtr1,
                                                                                                                                          gsout,
                                                                                                                                          dstPtr,
                                                                                                                                          srcSize1.height,
                                                                                                                                          srcSize1.width,
                                                                                                                                          srcSize2.height,
                                                                                                                                          srcSize2.width,
                                                                                                                                          channel);
        // CreateProgramFromBinary(handle.GetStream(),"reconstruction_laplacian_image_pyramid.cpp","reconstruction_laplacian_image_pyramid.cpp.bin","reconstruction_laplacian_image_pyramid_pkd",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    else
    {
        handle.AddKernel("", "", "reconstruction_laplacian_image_pyramid.cpp", "reconstruction_laplacian_image_pyramid_pln", vld, vgd, "")(srcPtr1,
                                                                                                                                          gsout,
                                                                                                                                          dstPtr,
                                                                                                                                          srcSize1.height,
                                                                                                                                          srcSize1.width,
                                                                                                                                          srcSize2.height,
                                                                                                                                          srcSize2.width,
                                                                                                                                          channel);
        // CreateProgramFromBinary(handle.GetStream(),"reconstruction_laplacian_image_pyramid.cpp","reconstruction_laplacian_image_pyramid.cpp.bin","reconstruction_laplacian_image_pyramid_pln",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    
    //---- Args Setter
    // ctr=0;
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr1);
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &gsout);
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize1.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize1.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize2.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize2.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);
    
    return RPP_SUCCESS;    
}
RppStatus
reconstruction_laplacian_image_pyramid_hip_batch( Rpp8u* srcPtr1, Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)
    // cl_mem srcPtr1, RppiSize *srcSize1, cl_mem srcPtr2, RppiSize *srcSize2, cl_mem dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32u handle.GetBatchSize(), RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    unsigned int maxHeight1, maxWidth1, maxHeight2, maxWidth2, maxKernelSize;
    unsigned long ioBufferSize1 = 0, ioBufferSize2 = 0;
    maxHeight1 = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxWidth1 = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    maxHeight2 = handle.GetInitHandle()->mem.mgpu.cdstSize.height[0];
    maxWidth2 = handle.GetInitHandle()->mem.mgpu.cdstSize.width[0];
    maxKernelSize = handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[0];
    for(int i = 0 ; i < handle.GetBatchSize() ; i++)
    {
        if(maxHeight1 < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxHeight1 = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if(maxWidth1 < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxWidth1 = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        ioBufferSize1 += handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel;
        if(maxHeight2 < handle.GetInitHandle()->mem.mgpu.cdstSize.height[i])
            maxHeight2 = handle.GetInitHandle()->mem.mgpu.cdstSize.height[i];
        if(maxWidth2 < handle.GetInitHandle()->mem.mgpu.cdstSize.width[i])
            maxWidth2 = handle.GetInitHandle()->mem.mgpu.cdstSize.width[i];
        ioBufferSize2 += handle.GetInitHandle()->mem.mgpu.cdstSize.width[i] * handle.GetInitHandle()->mem.mgpu.cdstSize.height[i] * channel;
        if(maxKernelSize < handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i])
            maxKernelSize = handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i];
    }

    Rpp32f *kernelMain = (Rpp32f *)calloc(maxKernelSize * maxKernelSize, sizeof(Rpp32f));

    Rpp8u* srcPtr1Temp;
    hipMalloc(&srcPtr1Temp, sizeof(unsigned char) * maxHeight1 * maxWidth1 * channel);
    Rpp8u* srcPtr2Temp;
    hipMalloc(&srcPtr2Temp, sizeof(unsigned char) * maxHeight2 * maxWidth2 * channel);
    Rpp8u* dstPtrTemp;
    hipMalloc(&dstPtrTemp, sizeof(unsigned char) * maxHeight1 * maxWidth1 * channel);

    Rpp8u* gsin;
    hipMalloc(&gsin, sizeof(unsigned char) * maxHeight1 * maxWidth1 * channel);
    Rpp8u* gsout;
    hipMalloc(&gsout, sizeof(unsigned char) * maxHeight1 * maxWidth1 * channel);

    Rpp32f* kernel;
    hipMalloc(&kernel,  maxKernelSize * maxKernelSize * sizeof(Rpp32f));

    int ctr;
           
    size_t gDim3[3];

    size_t batchIndex1 = 0, batchIndex2 = 0;

    for(int i = 0 ; i < handle.GetBatchSize() ; i++)
    { 

        gDim3[0] = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        gDim3[1] = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        gDim3[2] = channel;
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};

        hipMemcpy(srcPtr1Temp, srcPtr1+batchIndex1,sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel,hipMemcpyDeviceToDevice);
        hipMemcpy( srcPtr2Temp, srcPtr2+batchIndex2,sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.cdstSize.width[i] * handle.GetInitHandle()->mem.mgpu.cdstSize.height[i] * channel, hipMemcpyDeviceToDevice);

        /* Resize the Source 2 */
        if (chnFormat == RPPI_CHN_PLANAR)
        {   
            handle.AddKernel("", "", "resize.cpp", "resize_pln", vld, vgd, "")(srcPtr2Temp,
                                                                          gsin,
                                                                          handle.GetInitHandle()->mem.mgpu.cdstSize.height[i],
                                                                          handle.GetInitHandle()->mem.mgpu.cdstSize.width[i],
                                                                          handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                          handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                          channel);

            // CreateProgramFromBinary(handle.GetStream(),"reconstruction_laplacian_image_pyramid.cpp","reconstruction_laplacian_image_pyramid.cpp.bin","resize_pln",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {   
            handle.AddKernel("", "", "resize.cpp", "resize_pln", vld, vgd, "")(srcPtr2Temp,
                                                                          gsin,
                                                                          handle.GetInitHandle()->mem.mgpu.cdstSize.height[i],
                                                                          handle.GetInitHandle()->mem.mgpu.cdstSize.width[i],
                                                                          handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                          handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                          channel);
            // CreateProgramFromBinary(handle.GetStream(),"reconstruction_laplacian_image_pyramid.cpp","reconstruction_laplacian_image_pyramid.cpp.bin","resize_pkd",theProgram,theKernel);
            // clRetainKernel(theKernel); 
        }

        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr2Temp);
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &gsin);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.cdstSize.height[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.cdstSize.width[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.height[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.width[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
        // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);

        generate_gaussian_kernel_gpu(handle.GetInitHandle()->mem.mcpu.floatArr[0].floatmem[i], kernelMain, handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i]);    
        hipMemcpy(kernel, kernelMain,handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i] * handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i] * sizeof(Rpp32f),hipMemcpyHostToDevice);
        if(chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "gaussian_filter.cpp", "gaussian_pkd", vld, vgd, "")(gsin,
                                                                              gsout,
                                                                              handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                              handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                              channel,
                                                                              kernel,
                                                                              handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                                                                              handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i]);
            // CreateProgramFromBinary(handle.GetStream(),"reconstruction_laplacian_image_pyramid.cpp","reconstruction_laplacian_image_pyramid.cpp.bin","gaussian_pkd",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        else
        {
            handle.AddKernel("", "", "gaussian_filter.cpp", "gaussian_pln", vld, vgd, "")(gsin,
                                                                              gsout,
                                                                              handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                              handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                              channel,
                                                                              kernel,
                                                                              handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                                                                              handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i]);
            // CreateProgramFromBinary(handle.GetStream(),"reconstruction_laplacian_image_pyramid.cpp","reconstruction_laplacian_image_pyramid.cpp.bin","gaussian_pln",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        
        //---- Args Setter
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &gsin);
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &gsout);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.height[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.width[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &kernel);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i]);
        // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);

        /* Reconstruction of Laplacian Image pyramid */
        if(chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "reconstruction_laplacian_image_pyramid.cpp", "reconstruction_laplacian_image_pyramid_pkd", vld, vgd, "")(srcPtr1Temp,
                                                                                                                                          gsout,
                                                                                                                                          dstPtrTemp,
                                                                                                                                          handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                                                                          handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                                                                          handle.GetInitHandle()->mem.mgpu.cdstSize.height[i],
                                                                                                                                          handle.GetInitHandle()->mem.mgpu.cdstSize.width[i],
                                                                                                                                          channel);
            // CreateProgramFromBinary(handle.GetStream(),"reconstruction_laplacian_image_pyramid.cpp","reconstruction_laplacian_image_pyramid.cpp.bin","reconstruction_laplacian_image_pyramid_pkd",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        else
        {
            handle.AddKernel("", "", "reconstruction_laplacian_image_pyramid.cpp", "reconstruction_laplacian_image_pyramid_pln", vld, vgd, "")(srcPtr1Temp,
                                                                                                                                          gsout,
                                                                                                                                          dstPtrTemp,
                                                                                                                                          handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                                                                          handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                                                                          handle.GetInitHandle()->mem.mgpu.cdstSize.height[i],
                                                                                                                                          handle.GetInitHandle()->mem.mgpu.cdstSize.width[i],
                                                                                                                                          channel);
            // CreateProgramFromBinary(handle.GetStream(),"reconstruction_laplacian_image_pyramid.cpp","reconstruction_laplacian_image_pyramid.cpp.bin","reconstruction_laplacian_image_pyramid_pln",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        
        //---- Args Setter
        // ctr=0;
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr1Temp);
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &gsout);
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtrTemp);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.height[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.width[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.cdstSize.height[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.cdstSize.width[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
        // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);

        hipMemcpy( dstPtr+batchIndex1, dstPtrTemp,sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, hipMemcpyDeviceToDevice);

        batchIndex1 += handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel * sizeof(unsigned char);
        batchIndex2 += handle.GetInitHandle()->mem.mgpu.cdstSize.height[i] * handle.GetInitHandle()->mem.mgpu.cdstSize.width[i] * channel * sizeof(unsigned char);  
    }    

    return RPP_SUCCESS;    
}
/****************  Tensor convert bit depth *******************/
template <typename T, typename U>
RppStatus
tensor_convert_bit_depth_hip( Rpp32u tensorDimension, Rpp32u* tensorDimensionValues, T* srcPtr,
                             U* dstPtr, Rpp32u type, rpp::Handle& handle)
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
    if(type == 1)
    {
        handle.AddKernel("", "", "tensor.cpp", "tensor_convert_bit_depth_u8s8", vld, vgd, "")(tensorDimension,
                                                                                            srcPtr,
                                                                                            dstPtr,
                                                                                            dim1,
                                                                                            dim2,
                                                                                            dim3);
        // CreateProgramFromBinary(theQueue,"convert_bit_depth.cpp","convert_bit_depth.cpp.bin","tensor_convert_bit_depth_u8s8",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    else if(type == 2)
    {
        handle.AddKernel("", "", "tensor.cpp", "tensor_convert_bit_depth_u8u16", vld, vgd, "")(tensorDimension,
                                                                                            srcPtr,
                                                                                            dstPtr,
                                                                                            dim1,
                                                                                            dim2,
                                                                                            dim3);
        // CreateProgramFromBinary(theQueue,"convert_bit_depth.cpp","convert_bit_depth.cpp.bin","tensor_convert_bit_depth_u8u16",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    else
    {
        handle.AddKernel("", "", "tensor.cpp", "tensor_convert_bit_depth_u8s16", vld, vgd, "")(tensorDimension,
                                                                                            srcPtr,
                                                                                            dstPtr,
                                                                                            dim1,
                                                                                            dim2,
                                                                                            dim3);
        // CreateProgramFromBinary(theQueue,"convert_bit_depth.cpp","convert_bit_depth.cpp.bin","tensor_convert_bit_depth_u8s16",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    
    //---- Args Setter
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &tensorDimension);
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &dim1);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &dim2);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &dim3);
    // cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    return RPP_SUCCESS;
}

RppStatus
remap_hip(Rpp8u *srcPtr, RppiSize srcSize, Rpp8u *dstPtr, Rpp32u* rowRemapTable, Rpp32u* colRemapTable,
RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    return RPP_SUCCESS;
}

RppStatus
remap_hip_batch(Rpp8u *srcPtr, Rpp8u* dstPtr, Rpp32u* rowRemapTable, Rpp32u* colRemapTable,
         rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    return RPP_SUCCESS;
}