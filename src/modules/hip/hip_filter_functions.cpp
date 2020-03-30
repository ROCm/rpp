#include "hip/hip_runtime_api.h"
#include "hip_declarations.hpp"
#include <hip/rpp_hip_common.hpp>

/********************** Sobel ************************/
RppStatus
sobel_filter_hip ( Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u sobelType,
            RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    int ctr=0;
    cl_kernel theKernel;
    cl_program theProgram;
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "sobel.cpp", "sobel_pkd", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel,
                                                                        sobelType
                                                                        );
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "sobel.cpp", "sobel_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel,
                                                                        sobelType
                                                                        );
    }
    
    //---- Args Setter
    // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &srcPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &dstPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
        
    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    // cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    
    return RPP_SUCCESS;    
}



RppStatus
sobel_filter_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
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
    handle.AddKernel("", "", "sobel.cpp", "sobel_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
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


/********************** box_filter  ************************/

RppStatus
box_filter_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    float box_3x3[] = {
    0.111, 0.111, 0.111,
    0.111, 0.111, 0.111,
    0.111, 0.111, 0.111,
    };

    int ctr=0;
    // cl_context theContext;
    // clGetCommandQueueInfo(  handle.GetStream(),
    //                         CL_QUEUE_CONTEXT,
    //                         sizeof(cl_context), &theContext, NULL);
    float* filtPtr;
    hipMalloc(&filtPtr,sizeof(float)*3*3);
    hipMemcpy(filtPtr,box_3x3,sizeof(float)*3*3,hipMemcpyHostToDevice);
    kernelSize=3;
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "convolution.cpp", "naive_convolution_planar", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        filtPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel,
                                                                        kernelSize
                                                                        );

    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "convolution.cpp", "naive_convolution_packed", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        filtPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel,
                                                                        kernelSize
                                                                        );
    }
    // 
    //  //---- Args Setter
    // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &srcPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &dstPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &filtPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);

    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    // cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    hipFree(filtPtr);
    return RPP_SUCCESS;  

}

RppStatus
box_filter_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
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
    handle.AddKernel("", "", "box_filter.cpp", "box_filter_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
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

/********************** median_filter ************************/
RppStatus
median_filter_hip ( Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    int ctr=0;
    cl_kernel theKernel;
    cl_program theProgram;
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "median_filter.cpp", "median_filter_pkd", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel,
                                                                        kernelSize
                                                                        );
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "median_filter.cpp", "median_filter_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel,
                                                                        kernelSize
                                                                        );
    }
    
    // //---- Args Setter
    // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &srcPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &dstPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);
        
    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    // cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    
    return RPP_SUCCESS;    
}

RppStatus
median_filter_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
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
    handle.AddKernel("", "", "median_filter.cpp", "median_filter_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
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

/********************** non_max_suppression ************************/

RppStatus
non_max_suppression_hip( Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    int ctr=0;
    cl_kernel theKernel;
    cl_program theProgram;
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "non_max_suppression.cpp", "non_max_suppression_pkd", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel,
                                                                        kernelSize
                                                                        );
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "non_max_suppression.cpp", "non_max_suppression_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel,
                                                                        kernelSize
                                                                        );
    }

     //---- Args Setter
    // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &srcPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &dstPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);

    //  size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    // cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    return RPP_SUCCESS;  

}

RppStatus
non_max_suppression_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
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
    handle.AddKernel("", "", "non_max_suppression.cpp", "non_max_suppression_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
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

/********************** bilateral_filter ************************/

RppStatus
bilateral_filter_hip(Rpp8u* srcPtr, RppiSize srcSize,
                Rpp8u* dstPtr, unsigned int filterSize,
                double sigmaI, double sigmaS,
                RppiChnFormat chnFormat, unsigned int channel,
                rpp::Handle& handle)
{
    // unsigned short counter=0;
    // cl_int err;
    // cl_context theContext;
    // clGetCommandQueueInfo(  handle.GetStream(),
    //                         CL_QUEUE_CONTEXT,
    //                         sizeof(cl_context), &theContext, NULL);

    // cl_kernel theKernel;
    // cl_program theProgram;


    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "bilateral_filter.cpp", "bilateral_filter_planar", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel,
                                                                        sigmaI,
                                                                        sigmaS
                                                                        );
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "bilateral_filter.cpp", "bilateral_filter_packed", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel,
                                                                        sigmaI,
                                                                        sigmaS
                                                                        );
    }
    else
    {
        std::cerr << "Internal error: Unknown Channel format";
    }
//     err  = clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &srcPtr);
//     err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &dstPtr);
//     err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
//     err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
//     err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
//     err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &filterSize);
//     err |= clSetKernelArg(theKernel, counter++, sizeof(double), &sigmaI);
//     err |= clSetKernelArg(theKernel, counter++, sizeof(double), &sigmaS);

// //----
//     size_t gDim3[3];
//     gDim3[0] = srcSize.width;
//     gDim3[1] = srcSize.height;
//     gDim3[2] = channel;
//     cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    return RPP_SUCCESS;
}

RppStatus
bilateral_filter_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
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
    handle.AddKernel("", "", "bilateral_filter.cpp", "bilateral_filter_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                                                handle.GetInitHandle()->mem.mgpu.doubleArr[1].doublemem,
                                                                handle.GetInitHandle()->mem.mgpu.doubleArr[2].doublemem,
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

/********************** gaussian_filter ************************/
RppStatus
gaussian_filter_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    Rpp32f *kernelMain = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
    generate_gaussian_kernel_gpu(stdDev, kernelMain, kernelSize);
    
    // cl_context theContext;
    // clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    // cl_device_id theDevice;
    // clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    // Rpp8u* kernel = clCreateBuffer(theContext, Rpp8u*_WRITE_ONLY, kernelSize * kernelSize * sizeof(Rpp32f), NULL, NULL);
    // clEnqueueWriteBuffer(handle.GetStream(), kernel, CL_TRUE, 0, kernelSize * kernelSize * sizeof(Rpp32f), kernelMain, 0, NULL, NULL);
    Rpp32f *kernel;
    hipMemcpy(kernel,kernelMain,sizeof(Rpp32f)*kernelSize*kernelSize,hipMemcpyHostToDevice);
    int ctr=0;
    cl_kernel theKernel;
    cl_program theProgram;
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "gaussian_filter.cpp", "gaussian_pkd", vld, vgd, "")(srcPtr,
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
        handle.AddKernel("", "", "gaussian_filter.cpp", "gaussian_pln", vld, vgd, "")(srcPtr,
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
    // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &srcPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &dstPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &kernel);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);

    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    // cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    free(kernelMain);
    hipFree(kernel);
    return RPP_SUCCESS;  
}

RppStatus
gaussian_filter_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
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
    handle.AddKernel("", "", "gaussian_filter.cpp", "gaussian_filter_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
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

/********************** custom_convolution ************************/
RppStatus
custom_convolution_cl( Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32f* kernel,
 RppiSize kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    int buffer_size_kernel_size = sizeof(float) * kernelSize.height * kernelSize.width;
    Rpp32f* d_kernel;
    hipMalloc(&d_kernel, sizeof(float) * kernelSize.height * kernelSize.width);
    hipMemcpy(d_kernel,kernel,buffer_size_kernel_size,hipMemcpyHostToDevice);
    // int ctr=0;
    // cl_kernel theKernel;
    // cl_program theProgram;
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "custom_convolution.cpp", "custom_convolution_pkd", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel,
                                                                        d_kernel,
                                                                        kernelSize.height,
                                                                        kernelSize.width
                                                                        );
        // CreateProgramFromBinary(theQueue,"custom_convolution.cpp","custom_convolution.cpp.bin","custom_convolution_pkd",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "custom_convolution.cpp", "custom_convolution_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel,
                                                                        d_kernel,
                                                                        kernelSize.height,
                                                                        kernelSize.width
                                                                        );
        // CreateProgramFromBinary(theQueue,"custom_convolution.cpp","custom_convolution.cpp.bin","custom_convolution_pln",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }

    //---- Args Setter
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &clkernel);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize.width);

    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    // cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    return RPP_SUCCESS;  
}

RppStatus
custom_convolution_cl_batch (Rpp8u* srcPtr, Rpp8u* dstPtr, Rpp32f *kernel, RppiSize KernelSize,
                        rpp::Handle& handle,RppiChnFormat chnFormat, unsigned int channel)
    // cl_mem srcPtr, RppiSize *srcSize, RppiSize *maxSize,
    //                         cl_mem dstPtr, Rpp32f *kernel,
    //                         RppiSize KernelSize, RppiROI *roiPoints, Rpp32u nbatchSize,
    //                         RppiChnFormat chnFormat, unsigned int channel,
    //                         rpp::Handle& handle)
{
    Rpp32u nbatchSize = handle.GetBatchSize();
    int buffer_size_kernel_size = nbatchSize * sizeof(float) * KernelSize.height * KernelSize.width;
    int plnpkdind;
    Rpp32f* d_kernel;
    hipMalloc(&d_kernel, nbatchSize * sizeof(float) * KernelSize.height * KernelSize.width);

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    hipMemcpy(d_kernel,kernel,buffer_size_kernel_size,hipMemcpyHostToDevice);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "custom_convolution.cpp", "custom_convolution_batch", vld, vgd, "")(srcPtr,
                                                                dstPtr,
                                                                d_kernel,
                                                                KernelSize.height,
                                                                KernelSize.width,
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

    // unsigned int max_width = 0, max_height = 0;
    // cl_kernel theKernel;
    // cl_program theProgram;

    // //Getting Size arrays
    // unsigned long *batch_index;
    // unsigned int *widths, *heights, *inc, *max_widths;
    // widths  = (unsigned int *)malloc(sizeof(unsigned int)* nbatchSize);
    // heights = (unsigned int *)malloc(sizeof(unsigned int)* nbatchSize);
    // inc     = (unsigned int *)malloc(sizeof(unsigned int)* nbatchSize);
    // batch_index = (unsigned long *)malloc(sizeof(unsigned long)* nbatchSize);
    // max_widths = (unsigned int *)malloc(sizeof(unsigned int)* nbatchSize);

    // get_size_params(srcSize,maxSize, nbatchSize,widths,heights,max_widths, batch_index,channel);


    // // Getting ROI-Coordinates
    // unsigned int *xroi_begin, *xroi_end, *yroi_begin, *yroi_end;
    // xroi_begin = (unsigned int *)malloc(sizeof(int) * nbatchSize);
    // xroi_end = (unsigned int *)malloc(sizeof(int) * nbatchSize);
    // yroi_begin = (unsigned int *)malloc(sizeof(int) * nbatchSize);
    // yroi_end = (unsigned int *)malloc(sizeof(int) * nbatchSize);
    // get_roi_dims(roiPoints, srcSize, nbatchSize, xroi_begin, xroi_end, yroi_begin, yroi_end);

    // int buffer_size = nbatchSize * sizeof(unsigned int);
    // int buffer_size_kernel_size = nbatchSize * sizeof(float) * KernelSize.height * KernelSize.width;
    // int buffer_size_float = nbatchSize * sizeof(float);
    // int buffer_size_long = nbatchSize * sizeof(unsigned long);
    // cl_context ctx;
    // int plnpkdind;

    // if(chnFormat == RPPI_CHN_PLANAR){
    //     for(int i =0; i<nbatchSize; i++)
    //         inc[i] = maxSize[i].height * maxSize[i].width;
    //     plnpkdind = 1;
    // }
    // else{
    //     for(int i =0; i<nbatchSize; i++)
    //         inc[i] = 1;
    //     plnpkdind = 3;
    // }

    // clGetCommandQueueInfo(  theQueue,
    //                         CL_QUEUE_CONTEXT,
    //                         sizeof(cl_context), &ctx, NULL);

    // cl_mem d_kernel, d_xroi_begin, d_xroi_end,d_yroi_begin, d_yroi_end, d_height, 
    //         d_width, d_batch_index, d_inc, d_max_width;
    // cl_int err;
    
    // d_kernel = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buffer_size_kernel_size, NULL, &err);
    // d_xroi_begin = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buffer_size, NULL, &err);
    // d_xroi_end = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buffer_size, NULL, &err);
    // d_yroi_begin = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buffer_size, NULL, &err);
    // d_yroi_end = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buffer_size, NULL, &err);
    // d_width = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buffer_size, NULL, &err);
    // d_max_width = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buffer_size, NULL, &err);
    // d_height = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buffer_size, NULL, &err);
    // d_batch_index = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buffer_size_long, NULL, &err);
    // d_inc = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buffer_size, NULL, &err);

    

    // err = clEnqueueWriteBuffer(theQueue, d_kernel, CL_FALSE, 0, buffer_size_kernel_size, kernel, 0,
    //                             NULL, NULL);
    // err = clEnqueueWriteBuffer(theQueue, d_height, CL_FALSE, 0, buffer_size, heights, 0,
    //                             NULL, NULL);
    // err = clEnqueueWriteBuffer(theQueue, d_width, CL_FALSE, 0, buffer_size, widths, 0,
    //                             NULL, NULL);
    // err = clEnqueueWriteBuffer(theQueue, d_max_width, CL_FALSE, 0, buffer_size, max_widths, 0,
    //                             NULL, NULL);
    // err = clEnqueueWriteBuffer(theQueue, d_batch_index, CL_FALSE, 0, buffer_size_long, batch_index , 0,
    //                             NULL, NULL);
    // err = clEnqueueWriteBuffer(theQueue, d_xroi_begin, CL_FALSE, 0, buffer_size, xroi_begin, 0,
    //                             NULL, NULL);
    // err = clEnqueueWriteBuffer(theQueue, d_xroi_end, CL_FALSE, 0, buffer_size, xroi_end, 0,
    //                             NULL, NULL);
    // err = clEnqueueWriteBuffer(theQueue, d_yroi_begin, CL_FALSE, 0, buffer_size, yroi_begin, 0,
    //                             NULL, NULL);
    // err = clEnqueueWriteBuffer(theQueue, d_yroi_end, CL_FALSE, 0, buffer_size, yroi_end, 0,
    //                             NULL, NULL);
    // err = clEnqueueWriteBuffer(theQueue, d_inc, CL_FALSE, 0, buffer_size, inc, 0,
    //                             NULL, NULL);

    // CreateProgramFromBinary(theQueue,"custom_convolution.cpp","custom_convolution.cpp.bin","custom_convolution_batch",theProgram,theKernel);
    // clRetainKernel(theKernel);
    // // Arguments Setting//
    // int ctr = 0;
    // err = clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
    // err = clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
    // err = clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &d_kernel);
    // err = clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &KernelSize.height);
    // err = clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &KernelSize.width);
    // err = clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &d_xroi_begin);
    // err = clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &d_xroi_end);
    // err = clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &d_yroi_begin);
    // err = clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &d_yroi_end);
    // err = clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &d_height);
    // err = clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &d_width);
    // err = clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &d_max_width);
    // err = clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &d_batch_index);
    // err = clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    // err = clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &d_inc);
    // err = clSetKernelArg(theKernel, ctr++, sizeof(int), &plnpkdind);

    // max_size(srcSize, nbatchSize, &max_height, &max_width);
    // size_t gDim3[3];
    // gDim3[0] = max_width;
    // gDim3[1] = max_height;
    // gDim3[2] = nbatchSize;
    // cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);

}

