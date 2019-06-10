#include <cl/rpp_cl_common.hpp>
#include "cl_declarations.hpp"


RppStatus
convert_rgb2hsv_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue){
    cl_kernel theKernel;
    cl_program theProgram;

    if (chnFormat == RPPI_CHN_PLANAR)
    cl_kernel_initializer(theQueue,
                          "hsv_kernels.cl",
                          "rgb2hsv_pln",
                          theProgram, theKernel);
    else
    cl_kernel_initializer(theQueue,
                          "hsv_kernels.cl",
                          "rgb2hsv_pkd",
                          theProgram, theKernel);


    //---- Args Setter
    unsigned int n = srcSize.height * srcSize.width * channel ;
    clSetKernelArg(theKernel, 0, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, 1, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, 2, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, 3, sizeof(unsigned int), &srcSize.width);


    size_t gDim3[3];
    gDim3[0]= srcSize.height * srcSize.width;
    gDim3[1] = 1;
    gDim3[2] = 1;
    cl_kernel_implementer (theQueue, gDim3, NULL, theProgram, theKernel);


    return RPP_SUCCESS;

}

RppStatus
convert_hsv2rgb_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue){
    cl_kernel theKernel;
    cl_program theProgram;

    if (chnFormat == RPPI_CHN_PLANAR)
    cl_kernel_initializer(theQueue,
                          "hsv_kernels.cl",
                          "hsv2rgb_pln",
                          theProgram, theKernel);
    else
    cl_kernel_initializer(theQueue,
                          "hsv_kernels.cl",
                          "hsv2rgb_pkd",
                          theProgram, theKernel);


    //---- Args Setter
    unsigned int n = srcSize.height * srcSize.width * channel ;
    clSetKernelArg(theKernel, 0, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, 1, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, 2, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, 3, sizeof(unsigned int), &srcSize.width);


    size_t gDim3[3];
    gDim3[0]= srcSize.height * srcSize.width;
    gDim3[1] = 1;
    gDim3[2] = 1;
    cl_kernel_implementer (theQueue, gDim3, NULL, theProgram, theKernel);


    return RPP_SUCCESS;

}

//////////////////////HUE CODE////////////////////////////////
RppStatus
hue_saturation_rgb_cl (cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, Rpp32f hue, Rpp32f saturation,
                RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue){
    cl_kernel theKernel;
    cl_program theProgram;
    /*
    Intermediate Buffer Should be Allocated for use.
    */
    cl_mem temp; //for intermediate purpose
    unsigned int bytes = srcSize.height * srcSize.width* channel * sizeof(double);
    cl_context theContext;
    clGetCommandQueueInfo(  theQueue,
                            CL_QUEUE_CONTEXT,
                            sizeof(cl_context), &theContext, NULL);
    temp = clCreateBuffer(theContext,  CL_MEM_READ_WRITE , bytes, NULL, NULL);

    if (chnFormat == RPPI_CHN_PLANAR)
       cl_kernel_initializer(theQueue,
                          "hsv_kernels.cl",
                          "huergb_pln",
                          theProgram, theKernel);
    else if (chnFormat == RPPI_CHN_PACKED)
        cl_kernel_initializer(theQueue,
                          "hsv_kernels.cl",
                          "huergb_pkd",
                          theProgram, theKernel);

    //---- Args Setter
    clSetKernelArg(theKernel, 0, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, 1, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, 2, sizeof(cl_mem), &temp);
    clSetKernelArg(theKernel, 3, sizeof(double), &hue);
    clSetKernelArg(theKernel, 4, sizeof(double), &saturation);
    clSetKernelArg(theKernel, 5, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, 6, sizeof(unsigned int), &srcSize.width);

    size_t gDim3[3];
    gDim3[0]= srcSize.height * srcSize.width;
    gDim3[1] = 1;
    gDim3[2] = 1;
    cl_kernel_implementer (theQueue, gDim3, NULL, theProgram, theKernel);

    clReleaseMemObject(temp);
}

RppStatus
hue_saturation_hsv_cl (cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, Rpp32f hue, Rpp32f saturation,
                RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue){
    cl_kernel theKernel;
    cl_program theProgram;

    if (chnFormat == RPPI_CHN_PLANAR)
    cl_kernel_initializer(theQueue,
                          "hsv_kernels.cl",
                          "hsvhsv_pln",
                          theProgram, theKernel);
    else
    cl_kernel_initializer(theQueue,
                          "hsv_kernels.cl",
                          "huehsv_pkd",
                          theProgram, theKernel);

    //---- Args Setter
    clSetKernelArg(theKernel, 0, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, 1, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, 2, sizeof(double), &hue);
    clSetKernelArg(theKernel, 3, sizeof(double), &saturation);
    clSetKernelArg(theKernel, 4, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, 5, sizeof(unsigned int), &srcSize.width);


    size_t gDim3[3];
    gDim3[0]= srcSize.height * srcSize.width;
    gDim3[1] = 1;
    gDim3[2] = 1;
    cl_kernel_implementer (theQueue, gDim3, NULL, theProgram, theKernel);

}
