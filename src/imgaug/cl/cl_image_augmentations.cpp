#include <cl/rpp_cl_common.hpp>
#include "cl_declarations.hpp"

/****************** Brightness ******************/

RppStatus
brightness_cl (    cl_mem srcPtr, RppiSize srcSize,
                            cl_mem dstPtr,
                            Rpp32f alpha, Rpp32s beta,
                            RppiChnFormat chnFormat, unsigned int channel,
                            cl_command_queue theQueue)
{
    unsigned short counter=0;
    cl_kernel theKernel;
    cl_program theProgram;

    CreateProgramFromBinary(theQueue,"brightness_contrast.cl","brightness_contrast.cl.bin","brightness_contrast",theProgram,theKernel);
    clRetainKernel(theKernel); 

    // cl_kernel_initializer(theQueue,
    //                       "brightness_contrast.cl",
    //                       "brightness_contrast",
    //                       theProgram, theKernel);

    //---- Args Setter
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, counter++, sizeof(float), &alpha);
    clSetKernelArg(theKernel, counter++, sizeof(int), &beta);
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

/***************** Contrast *********************/

RppStatus
contrast_cl (    cl_mem srcPtr, RppiSize srcSize,
                            cl_mem dstPtr,
                            Rpp32u newMin, Rpp32u newMax,
                            RppiChnFormat chnFormat, unsigned int channel,
                            cl_command_queue theQueue)
{
    unsigned short counter=0;
    Rpp32u min = 0; /* Kernel has to be called */
    Rpp32u max = 255; /* Kernel has to be called */
    cl_kernel theKernel;
    cl_program theProgram;

    CreateProgramFromBinary(theQueue,"contrast_stretch.cl","contrast_stretch.cl.bin","contrast_stretch",theProgram,theKernel);
    clRetainKernel(theKernel); 

    // cl_kernel_initializer(theQueue,
    //                       "contrast_stretch.cl",
    //                       "contrast_stretch",
    //                       theProgram, theKernel);


    //----- Args Setter
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, counter++, sizeof(int), &min);
    clSetKernelArg(theKernel, counter++, sizeof(int), &max);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &newMin);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &newMax);
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &(srcSize.height));
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &(srcSize.width));
    clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
    //-----

    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;

    cl_kernel_implementer(theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);

    return RPP_SUCCESS;

}

/********************** Blur ************************/
float gauss_3x3[] = {
0.0625, 0.125, 0.0625,
0.125 , 0.25 , 0.125,
0.0625, 0.125, 0.0625,
};


cl_int
blur_cl(cl_mem srcPtr, RppiSize srcSize,
                cl_mem dstPtr, unsigned int filterSize,
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue)
{
    unsigned short counter=0;
    cl_int err;

    float* filterBuffer;
    if (filterSize == 3) filterBuffer= gauss_3x3;
    else  std::cerr << "Unimplemeted kernel Size";

    cl_context theContext;
    clGetCommandQueueInfo(  theQueue,
                            CL_QUEUE_CONTEXT,
                            sizeof(cl_context), &theContext, NULL);
    cl_mem filtPtr = clCreateBuffer(theContext, CL_MEM_READ_ONLY,
                                    sizeof(float)*filterSize*filterSize, NULL, NULL);
    err = clEnqueueWriteBuffer(theQueue, filtPtr, CL_TRUE, 0,
                                   sizeof(float)*filterSize*filterSize,
                                   filterBuffer, 0, NULL, NULL);


    cl_kernel theKernel;
    cl_program theProgram;


    if (chnFormat == RPPI_CHN_PLANAR)
    {
        CreateProgramFromBinary(theQueue,"convolution.cl","convolution.cl.bin","naive_convolution_planar",theProgram,theKernel);
        clRetainKernel(theKernel); 

        // cl_kernel_initializer(  theQueue, "convolution.cl",
        //                         "naive_convolution_planar", theProgram, theKernel);

    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {

        CreateProgramFromBinary(theQueue,"convolution.cl","convolution.cl.bin","naive_convolution_packed",theProgram,theKernel);
        clRetainKernel(theKernel); 
        // cl_kernel_initializer(  theQueue, "convolution.cl",
        //                         "naive_convolution_packed", theProgram, theKernel);
    }
    else
    {std::cerr << "Internal error: Unknown Channel format";}




    err  = clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    err |= clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    err |= clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &filtPtr);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &filterSize);

//----
    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    return RPP_SUCCESS;

}

/********************** Blend ************************/

RppStatus
blend_cl( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize, cl_mem dstPtr, float alpha,
                 RppiChnFormat chnFormat, unsigned int channel,
                 cl_command_queue theQueue)
{
    cl_kernel theKernel;
    cl_program theProgram;
    cl_kernel_initializer(theQueue,
                          "blend.cl",
                          "blend",
                          theProgram, theKernel);

    //---- Args Setter
    clSetKernelArg(theKernel, 0, sizeof(cl_mem), &srcPtr1);
    clSetKernelArg(theKernel, 1, sizeof(cl_mem), &srcPtr2);
    clSetKernelArg(theKernel, 2, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, 3, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, 4, sizeof(unsigned int), &srcSize.width);
    clSetKernelArg(theKernel, 5, sizeof(float), &alpha);
    clSetKernelArg(theKernel, 6, sizeof(unsigned int), &channel);
    //----

    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);

    return RPP_SUCCESS;
}

cl_int
pixelate_cl(cl_mem srcPtr, RppiSize srcSize,cl_mem dstPtr, 
            unsigned int filterSize, unsigned int x1, unsigned int y1,
            unsigned int x2, unsigned int y2,RppiChnFormat chnFormat,
            unsigned int channel,cl_command_queue theQueue)
{
    cl_int err;
    unsigned short counter=0;

    float* filterBuffer;
    if (filterSize == 3) filterBuffer= gauss_3x3;
    else  std::cerr << "Unimplemeted kernel Size";

    cl_context theContext;
    clGetCommandQueueInfo(  theQueue,
                            CL_QUEUE_CONTEXT,
                            sizeof(cl_context), &theContext, NULL);
    cl_mem filtPtr = clCreateBuffer(theContext, CL_MEM_READ_ONLY,
                                    sizeof(float)*filterSize*filterSize, NULL, NULL);
    err = clEnqueueWriteBuffer(theQueue, filtPtr, CL_TRUE, 0,
                                   sizeof(float)*filterSize*filterSize,
                                   filterBuffer, 0, NULL, NULL);


    cl_kernel theKernel;
    cl_program theProgram;


    if (chnFormat == RPPI_CHN_PLANAR)
    {
        CreateProgramFromBinary(theQueue, "pixelate.cl","pixelate.bin",
                                "pixelate_planar", theProgram, theKernel);
        clRetainKernel(theKernel); 

    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {

        CreateProgramFromBinary(theQueue, "pixelate.cl","pixelate.bin",
                                "pixelate_packed", theProgram, theKernel);
        clRetainKernel(theKernel); 
    }
    else
    {std::cerr << "Internal error: Unknown Channel format";}

    err  = clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    err |= clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    err |= clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &filtPtr);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &x1);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &y1);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &x2);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &y2);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &filterSize);

    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);

    return RPP_SUCCESS;
}
//----
/********************** ADDING NOISE ************************/

RppStatus  
noise_add_gaussian_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, 
                RppiNoise noiseType,RppiGaussParameter *noiseParameter,
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue)
{
    std::default_random_engine generator;
    std::normal_distribution<>  distribution{noiseParameter->mean, noiseParameter->sigma}; 
    const unsigned int size = (srcSize.height * srcSize.width * channel);
    unsigned char output[size];
    for(int i = 0; i < (srcSize.height * srcSize.width * channel) ; i++)
    {
        Rpp32f pixel = ((Rpp32f)distribution(generator));
        pixel=(pixel < (Rpp32f) 0) ? ((Rpp32f) 0) : ((pixel < (Rpp32f) 255) ? pixel : ((Rpp32f) 255));
        output[i] = (Rpp8u)pixel;
    } 
    
    cl_mem d_b;  
    cl_context context;
    clGetCommandQueueInfo(  theQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, NULL); 
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, srcSize.height * srcSize.width * channel , NULL, NULL);
    clEnqueueWriteBuffer(theQueue, d_b, CL_TRUE, 0, srcSize.height * srcSize.width * channel, output, 0, NULL, NULL);
    Rpp32f mean, sigma;
    mean = noiseParameter->mean;
    sigma = noiseParameter->sigma;

    cl_kernel theKernel;
    cl_program theProgram;
    cl_kernel_initializer(theQueue,
                          "noise.cl",
                          "gaussian",
                          theProgram, theKernel);
     
    //---- Args Setter
    clSetKernelArg(theKernel, 0, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, 1, sizeof(cl_mem), &d_b);
    clSetKernelArg(theKernel, 2, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, 3, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, 4, sizeof(unsigned int), &srcSize.width);
    clSetKernelArg(theKernel, 5, sizeof(float), &mean);
    clSetKernelArg(theKernel, 6, sizeof(float), &sigma);
    clSetKernelArg(theKernel, 7, sizeof(unsigned int), &channel);
    //----

    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);

    clReleaseMemObject(d_b);
    return RPP_SUCCESS;

}

cl_int
jitter_cl( cl_mem srcPtr,RppiSize srcSize, cl_mem dstPtr,
           unsigned int minJitter,unsigned int maxJitter,
           RppiChnFormat chnFormat, unsigned int channel,
           cl_command_queue theQueue)
{
    unsigned short counter=0;
    cl_int err;
    cl_kernel theKernel;
    cl_program theProgram;
    
    CreateProgramFromBinary(theQueue, "jitter.cl","jitter.bin",
                                "jitter", theProgram, theKernel);
    clRetainKernel(theKernel); 

    //---- Args Setter
    err  = clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    err |= clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &minJitter);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &maxJitter);
    return RPP_SUCCESS;
}

RppStatus
noise_add_snp_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, 
                RppiNoise noiseType,Rpp32f *noiseParameter,
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue)
{
    Rpp32u noiseProbability= (Rpp32u)(*noiseParameter * srcSize.width * srcSize.height * channel );
    const unsigned int size = (srcSize.height * srcSize.width * channel);
    unsigned char output[size];
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for(int i = 0 ; i < noiseProbability ; i++)
        {
            Rpp32u row = rand() % srcSize.height;
            Rpp32u column = rand() % srcSize.width;
            Rpp8u newValue = rand()%2 ? 255 : 1;
            for (int c = 0; c < channel; c++)
            {
                output[(row * srcSize.width) + (column) + (c * srcSize.width * srcSize.height) ] = newValue;
            }
        }        
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for(int i = 0 ; i < noiseProbability ; i++)
        {
            Rpp32u row = rand() % srcSize.height;
            Rpp32u column = rand() % srcSize.width;
            Rpp8u newValue = rand()%2 ? 1 : 255;
            for (int c = 0; c < channel; c++)
            {
                output[(channel * row * srcSize.width) + (column * channel) + c] = newValue;
            }
        }
    }   
    cl_mem d_b;  
    cl_context context;
    clGetCommandQueueInfo(  theQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, NULL); 
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, srcSize.height * srcSize.width * channel , NULL, NULL);
    clEnqueueWriteBuffer(theQueue, d_b, CL_TRUE, 0, srcSize.height * srcSize.width * channel, output, 0, NULL, NULL); 
    
    cl_kernel theKernel;
    cl_program theProgram;
    cl_kernel_initializer(theQueue,
                          "noise.cl",
                          "snp",
                          theProgram, theKernel);
     
    //---- Args Setter
    clSetKernelArg(theKernel, 0, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, 1, sizeof(cl_mem), &d_b);
    clSetKernelArg(theKernel, 2, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, 3, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, 4, sizeof(unsigned int), &srcSize.width);
    clSetKernelArg(theKernel, 5, sizeof(unsigned int), &channel);
    //----

    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    clReleaseMemObject(d_b);

    return RPP_SUCCESS;
}

cl_int
snow_cl( cl_mem srcPtr,RppiSize srcSize, cl_mem dstPtr,
           float snowCoefficient,
           RppiChnFormat chnFormat, unsigned int channel,
           cl_command_queue theQueue)
{
    unsigned short counter=0;
    cl_int err;
    cl_kernel theKernel;
    cl_program theProgram;
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        CreateProgramFromBinary(theQueue, "snow.cl","snow.bin",
                                "snow_pln", theProgram, theKernel);
        clRetainKernel(theKernel); 

    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {

        CreateProgramFromBinary(theQueue, "snow.cl","snow.bin",
                                "snow_pkd", theProgram, theKernel);
        clRetainKernel(theKernel); 
    }
    else
    {std::cerr << "Internal error: Unknown Channel format";}

    //---- Args Setter
    err  = clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    err |= clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
    err |= clSetKernelArg(theKernel, counter++, sizeof(float), &snowCoefficient);
    //----

    size_t gDim3[3];
    gDim3[0] = srcSize.height * srcSize.width;
    gDim3[1] = 1;
    gDim3[2] = 1;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);

    return RPP_SUCCESS;
}
/********************** Exposure mocification ************************/

RppStatus
exposure_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32f exposureValue, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
{
    int ctr=0;
    cl_kernel theKernel;
    cl_program theProgram;
    cl_kernel_initializer(theQueue,
                          "exposure.cl",
                          "exposure",
                          theProgram, theKernel);

    //---- Args Setter
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    clSetKernelArg(theKernel, ctr++, sizeof(float), &exposureValue);
        
    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    
    return RPP_SUCCESS;    
}