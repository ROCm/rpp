#include <cl/rpp_cl_common.hpp>
#include <cpu/rpp_cpu_common.hpp>
#include "cl_declarations.hpp"

int generate_gaussian_kernel_asymmetric(Rpp32f stdDev, Rpp32f* kernel, Rpp32u kernelSizeX, Rpp32u kernelSizeY)
{
    Rpp32f s, sum = 0.0, multiplier;
    if (kernelSizeX % 2 == 0)
    {
        return 0;
    }
    if (kernelSizeY % 2 == 0)
    {
        return 0;
    }
    int boundX = ((kernelSizeX - 1) / 2);
    int boundY = ((kernelSizeY - 1) / 2);
    unsigned int c = 0;
    s = 1 / (2 * stdDev * stdDev);
    multiplier = (1 / M_PI) * (s);
    for (int i = -boundY; i <= boundY; i++)
    {
        for (int j = -boundX; j <= boundX; j++)
        {
            kernel[c] = multiplier * exp((-1) * (s) * (i*i + j*j));
            sum += kernel[c];
            c += 1;
        }
    };
    for (int i = 0; i < (kernelSizeX * kernelSizeY); i++)
    {
        kernel[i] /= sum;
    }
     return 1;
}

RppStatus
rgb_to_hsv_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue){
    unsigned short counter=0;
    cl_kernel theKernel;
    cl_program theProgram;

    if (chnFormat == RPPI_CHN_PLANAR)
    {   
        CreateProgramFromBinary(theQueue,"hsv_kernels.cl","hsv_kernels.cl.bin","rgb2hsv_pln",theProgram,theKernel);
        clRetainKernel(theKernel); 
        // cl_kernel_initializer(theQueue,
        //                 "hsv_kernels.cl",
        //                 "rgb2hsv_pln",
        //                 theProgram, theKernel);
    }
    else
    {
        CreateProgramFromBinary(theQueue,"hsv_kernels.cl","hsv_kernels.cl.bin","rgb2hsv_pkd",theProgram,theKernel);
        clRetainKernel(theKernel); 
        // cl_kernel_initializer(theQueue,
        //                 "hsv_kernels.cl",
        //                 "rgb2hsv_pkd",
        //                 theProgram, theKernel);
    }

    //---- Args Setter
    unsigned int n = srcSize.height * srcSize.width * channel ;
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);


    size_t gDim3[3];
    gDim3[0]= srcSize.height * srcSize.width;
    gDim3[1] = 1;
    gDim3[2] = 1;
    cl_kernel_implementer (theQueue, gDim3, NULL, theProgram, theKernel);


    return RPP_SUCCESS;

}

RppStatus
hsv_to_rgb_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue){
    unsigned short counter=0;
    cl_kernel theKernel;
    cl_program theProgram;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        CreateProgramFromBinary(theQueue,"hsv_kernels.cl","hsv_kernels.cl.bin","hsv2rgb_pln",theProgram,theKernel);
        clRetainKernel(theKernel); 
        // cl_kernel_initializer(theQueue,
        //                       "hsv_kernels.cl",
        //                       "hsv2rgb_pln",
        //                       theProgram, theKernel);
    }
    else
    {
        CreateProgramFromBinary(theQueue,"hsv_kernels.cl","hsv_kernels.cl.bin","hsv2rgb_pkd",theProgram,theKernel);
        clRetainKernel(theKernel); 
        // cl_kernel_initializer(theQueue,
        //                       "hsv_kernels.cl",
        //                       "hsv2rgb_pkd",
        //                       theProgram, theKernel);
    }

    //---- Args Setter
    unsigned int n = srcSize.height * srcSize.width * channel ;
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);


    size_t gDim3[3];
    gDim3[0]= srcSize.height * srcSize.width;
    gDim3[1] = 1;
    gDim3[2] = 1;
    cl_kernel_implementer (theQueue, gDim3, NULL, theProgram, theKernel);


    return RPP_SUCCESS;

}

//////////////////////HUE CODE////////////////////////////////
RppStatus
hueRGB_cl (cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, Rpp32f hue, Rpp32f saturation,
                RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue){
    unsigned short counter=0;
    cl_kernel theKernel;
    cl_program theProgram;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        CreateProgramFromBinary(theQueue,"hsv_kernels.cl","hsv_kernels.cl.bin","huergb_pln",theProgram,theKernel);
        clRetainKernel(theKernel); 
        //    cl_kernel_initializer(theQueue,
        //                       "hsv_kernels.cl",
        //                       "huergb_pln",
        //                       theProgram, theKernel);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        CreateProgramFromBinary(theQueue,"hsv_kernels.cl","hsv_kernels.cl.bin","huergb_pkd",theProgram,theKernel);
        clRetainKernel(theKernel);     
        // cl_kernel_initializer(theQueue,
        //                   "hsv_kernels.cl",
        //                   "huergb_pkd",
        //                   theProgram, theKernel);
    }

    //---- Args Setter
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, counter++, sizeof(double), &hue);
    clSetKernelArg(theKernel, counter++, sizeof(double), &saturation);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);

    size_t gDim3[3];
    gDim3[0]= srcSize.height * srcSize.width;
    gDim3[1] = 1;
    gDim3[2] = 1;
    cl_kernel_implementer (theQueue, gDim3, NULL, theProgram, theKernel);
}

RppStatus
hueHSV_cl (cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, Rpp32f hue, Rpp32f saturation,
                RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue){
    unsigned short counter=0;
    cl_kernel theKernel;
    cl_program theProgram;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        CreateProgramFromBinary(theQueue,"hsv_kernels.cl","hsv_kernels.cl.bin","hsvhsv_pln",theProgram,theKernel);
        clRetainKernel(theKernel); 
        // cl_kernel_initializer(theQueue,
        //                       "hsv_kernels.cl",
        //                       "hsvhsv_pln",
        //                       theProgram, theKernel);
    }
    else
    {
        CreateProgramFromBinary(theQueue,"hsv_kernels.cl","hsv_kernels.cl.bin","huehsv_pkd",theProgram,theKernel);
        clRetainKernel(theKernel); 
        // cl_kernel_initializer(theQueue,
        //                       "hsv_kernels.cl",
        //                       "huehsv_pkd",
        //                       theProgram, theKernel);
    }

    //---- Args Setter
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, counter++, sizeof(double), &hue);
    clSetKernelArg(theKernel, counter++, sizeof(double), &saturation);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);


    size_t gDim3[3];
    gDim3[0]= srcSize.height * srcSize.width;
    gDim3[1] = 1;
    gDim3[2] = 1;
    cl_kernel_implementer (theQueue, gDim3, NULL, theProgram, theKernel);

}

/****************  Gamma correction *******************/

RppStatus
gamma_correction_cl ( cl_mem srcPtr1,RppiSize srcSize,
                 cl_mem dstPtr, float gamma,
                 RppiChnFormat chnFormat, unsigned int channel,
                 cl_command_queue theQueue)
{
    unsigned short counter=0;
    cl_kernel theKernel;
    cl_program theProgram;

    CreateProgramFromBinary(theQueue,"gamma_correction.cl","gamma_correction.cl.bin","gamma_correction",theProgram,theKernel);
    clRetainKernel(theKernel); 

    // cl_kernel_initializer(theQueue,
    //                       "gamma_correction.cl",
    //                       "gamma_correction",
    //                       theProgram, theKernel);

    //---- Args Setter
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr1);
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, counter++, sizeof(float), &gamma);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
    //----

    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);

    return RPP_SUCCESS;

}

/****************  Temprature modification *******************/

RppStatus
temprature_cl( cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, float adjustmentValue, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
{
    cl_kernel theKernel;
    cl_program theProgram;

    if (chnFormat == RPPI_CHN_PLANAR)
    cl_kernel_initializer(theQueue,
                          "temprature.cl",
                          "temprature_planar",
                          theProgram, theKernel);
    else
    cl_kernel_initializer(theQueue,
                          "temprature.cl",
                          "temprature_packed",
                          theProgram, theKernel);
    //---- Args Setter
    clSetKernelArg(theKernel, 0, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, 1, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, 2, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, 3, sizeof(unsigned int), &srcSize.width);
    clSetKernelArg(theKernel, 4, sizeof(unsigned int), &channel);
    clSetKernelArg(theKernel, 5, sizeof(float), &adjustmentValue);

    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    
    return RPP_SUCCESS;
}

/****************  vignette effect *******************/

// value should always be greater than 0
//0-> full vignette effect
//100-> no vignette effect

RppStatus
vignette_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, float stdDev, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
{
    int ctr=0;
    cl_kernel theKernel;
    cl_program theProgram;
    cl_kernel_initializer(theQueue,
                          "vignette.cl",
                          "vignette",
                          theProgram, theKernel); 
    
    stdDev=(stdDev/100)*( sqrt(srcSize.width*srcSize.height*channel) * 2);

    Rpp32f *mask = (Rpp32f *)calloc(srcSize.height * srcSize.width, sizeof(Rpp32f));
    Rpp32f *maskTemp;
    maskTemp = mask;

    RppiSize kernelRowsSize, kernelColumnsSize;
    kernelRowsSize.height = srcSize.height;
    kernelRowsSize.width = 1;
    kernelColumnsSize.height = srcSize.width;
    kernelColumnsSize.width = 1;

    Rpp32f *kernelRows = (Rpp32f *)calloc(kernelRowsSize.height * kernelRowsSize.width, sizeof(Rpp32f));
    Rpp32f *kernelColumns = (Rpp32f *)calloc(kernelColumnsSize.height * kernelColumnsSize.width, sizeof(Rpp32f));

    if (kernelRowsSize.height % 2 == 0)
    {
        generate_gaussian_kernel_asymmetric(stdDev, kernelRows, kernelRowsSize.height - 1, kernelRowsSize.width);
        kernelRows[kernelRowsSize.height - 1] = kernelRows[kernelRowsSize.height - 2];
    }
    else
    {
        generate_gaussian_kernel_asymmetric(stdDev, kernelRows, kernelRowsSize.height, kernelRowsSize.width);
    }
    
    if (kernelColumnsSize.height % 2 == 0)
    {
        generate_gaussian_kernel_asymmetric(stdDev, kernelColumns, kernelColumnsSize.height - 1, kernelColumnsSize.width);
        kernelColumns[kernelColumnsSize.height - 1] = kernelColumns[kernelColumnsSize.height - 2];
    }
    else
    {
        generate_gaussian_kernel_asymmetric(stdDev, kernelColumns, kernelColumnsSize.height, kernelColumnsSize.width);
    }

    Rpp32f *kernelRowsTemp, *kernelColumnsTemp;
    kernelRowsTemp = kernelRows;
    kernelColumnsTemp = kernelColumns;
    
    for (int i = 0; i < srcSize.height; i++)
    {
        kernelColumnsTemp = kernelColumns;
        for (int j = 0; j < srcSize.width; j++)
        {
            *maskTemp = *kernelRowsTemp * *kernelColumnsTemp;
            maskTemp++;
            kernelColumnsTemp++;
        }
        kernelRowsTemp++;
    }

    Rpp32f max = 0;
    maskTemp = mask;
    for (int i = 0; i < (srcSize.width * srcSize.height); i++)
    {
        if (*maskTemp > max)
        {
            max = *maskTemp;
        }
        maskTemp++;
    }

    maskTemp = mask;
    for (int i = 0; i < (srcSize.width * srcSize.height); i++)
    {
        *maskTemp = *maskTemp / max;
        maskTemp++;
    }

    Rpp32f *maskFinal = (Rpp32f *)calloc(channel * srcSize.height * srcSize.width, sizeof(Rpp32f));
    Rpp32f *maskFinalTemp;
    maskFinalTemp = maskFinal;
    maskTemp = mask;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            maskTemp = mask;
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    *maskFinalTemp = *maskTemp;
                    maskFinalTemp++;
                    maskTemp++;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {
                    *maskFinalTemp = *maskTemp; // *100;
                    maskFinalTemp++;
                }
                maskTemp++;
            }
        }
    }

    cl_mem d_b;  
    cl_context context;
    clGetCommandQueueInfo(  theQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, NULL); 
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, srcSize.height * srcSize.width * channel * sizeof(float) , NULL, NULL);
    clEnqueueWriteBuffer(theQueue, d_b, CL_TRUE, 0, srcSize.height * srcSize.width * channel * sizeof(float), maskFinal, 0, NULL, NULL);

    //---- Args Setter
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &d_b);
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);

    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    return RPP_SUCCESS;
}