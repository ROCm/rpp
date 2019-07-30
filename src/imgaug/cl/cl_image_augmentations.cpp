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
    CreateProgramFromBinary(theQueue,"blend.cl","blend.cl.bin","blend",theProgram,theKernel);
    clRetainKernel(theKernel); 
    // cl_kernel_initializer(theQueue,
    //                       "blend.cl",
    //                       "blend",
    //                       theProgram, theKernel);

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
gaussianNoise_cl(cl_mem srcPtr,
                RppiSize srcSize,
                cl_mem dstPtr, 
                Rpp32f mean,Rpp32f sigma,
                RppiChnFormat chnFormat, unsigned int channel, 
                cl_command_queue theQueue)
{
//    std::default_random_engine generator;
//    std::normal_distribution<>  distribution{mean, sigma}; 
//    const unsigned int size = (srcSize.height * srcSize.width * channel);
//    unsigned char output[size];
//    for(int i = 0; i < (srcSize.height * srcSize.width * channel) ; i++)
//    {
//        Rpp32f pixel = ((Rpp32f)distribution(generator));
//        pixel=(pixel < (Rpp32f) 0) ? ((Rpp32f) 0) : ((pixel < (Rpp32f) 255) ? pixel : ((Rpp32f) 255));
//        output[i] = (Rpp8u)pixel;
//    } 
//    
//    cl_mem d_b;  
//    cl_context context;
//    clGetCommandQueueInfo(  theQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, NULL); 
//    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, srcSize.height * srcSize.width * channel , NULL, NULL);
//    clEnqueueWriteBuffer(theQueue, d_b, CL_TRUE, 0, srcSize.height * srcSize.width * channel, output, 0, NULL, NULL);
//
//    cl_kernel theKernel;
//    cl_program theProgram;
//    CreateProgramFromBinary(theQueue,"noise.cl","noise.cl.bin","gaussian",theProgram,theKernel);
//    clRetainKernel(theKernel);
//    // cl_kernel_initializer(theQueue,
//    //                       "noise.cl",
//    //                       "gaussian",
//    //                       theProgram, theKernel);
//     
//    //---- Args Setter
//    clSetKernelArg(theKernel, 0, sizeof(cl_mem), &srcPtr);
//    clSetKernelArg(theKernel, 1, sizeof(cl_mem), &d_b);
//    clSetKernelArg(theKernel, 2, sizeof(cl_mem), &dstPtr);
//    clSetKernelArg(theKernel, 3, sizeof(unsigned int), &srcSize.height);
//    clSetKernelArg(theKernel, 4, sizeof(unsigned int), &srcSize.width);
//    clSetKernelArg(theKernel, 5, sizeof(float), &mean);
//    clSetKernelArg(theKernel, 6, sizeof(float), &sigma);
//    clSetKernelArg(theKernel, 7, sizeof(unsigned int), &channel);
//    //----
//
//    size_t gDim3[3];
//    gDim3[0] = srcSize.width;
//    gDim3[1] = srcSize.height;
//    gDim3[2] = channel;
//    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
//
//    clReleaseMemObject(d_b);
//    return RPP_SUCCESS;

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
    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);

    return err;
}

RppStatus
snpNoise_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, 
                Rpp32f noiseProbability, 
                RppiChnFormat chnFormat, unsigned int channel,
                cl_command_queue theQueue)
{
    srand(time(0));
    int ctr = 0; 
    clEnqueueCopyBuffer(theQueue, srcPtr, dstPtr, 0, 0, sizeof(unsigned char) * srcSize.width * srcSize.height * channel, 0, NULL, NULL);
    if(noiseProbability != 0)
    {    
        Rpp32u noisePixel = (Rpp32u)(noiseProbability * srcSize.width * srcSize.height );
        const Rpp32u pixelDistance = (srcSize.width * srcSize.height) / noisePixel;
        cl_kernel theKernel;
        cl_program theProgram;
        if(chnFormat == RPPI_CHN_PACKED)
        {
            CreateProgramFromBinary(theQueue, "noise.cl", "noise.cl.bin", "snp_pkd", theProgram, theKernel);
            clRetainKernel(theKernel);
        }
        else if(chnFormat == RPPI_CHN_PLANAR)
        {
            CreateProgramFromBinary(theQueue, "noise.cl", "noise.cl.bin", "snp_pln", theProgram, theKernel);
            clRetainKernel(theKernel);
        }
        clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
        clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
        clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
        clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
        clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
        clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &pixelDistance);
        
        size_t gDim3[3];
        gDim3[0] = srcSize.width;
        gDim3[1] = srcSize.height;
        gDim3[2] = 1;
        cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    }

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
                                "snow_pkd", theProgram, theKernel);
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
    CreateProgramFromBinary(theQueue,"exposure.cl","exposure.cl.bin","exposure",theProgram,theKernel);
    clRetainKernel(theKernel);
    // cl_kernel_initializer(theQueue,
    //                       "exposure.cl",
    //                       "exposure",
    //                       theProgram, theKernel);

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


/********************** Rain ************************/

RppStatus
rain_cl(cl_mem srcPtr, RppiSize srcSize,cl_mem dstPtr, Rpp32f rainPercentage, Rpp32u rainWidth, Rpp32u rainHeight, Rpp32f transparency, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
{   
    if(rainPercentage == 0)
    {
        clEnqueueCopyBuffer(theQueue, srcPtr, dstPtr, 0, 0, sizeof(unsigned char) * srcSize.width * srcSize.height * channel, 0, NULL, NULL);
    }
    else
    {
        int ctr=0; 
        Rpp32u rainDrops= (Rpp32u)((rainPercentage * srcSize.width * srcSize.height )/100);
        Rpp32u pixelDistance= (Rpp32u)((srcSize.width * srcSize.height) / rainDrops);
        cl_kernel theKernel;
        cl_program theProgram;
        if(chnFormat == RPPI_CHN_PACKED)
        {
            CreateProgramFromBinary(theQueue,"rain.cl","rain.cl.bin","rain_pkd",theProgram,theKernel);
            clRetainKernel(theKernel);
        }
        else if(chnFormat == RPPI_CHN_PLANAR)
        {
            CreateProgramFromBinary(theQueue,"rain.cl","rain.cl.bin","rain_pln",theProgram,theKernel);
            clRetainKernel(theKernel);
        }

        //---- Args Setter
        clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
        clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
        clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
        clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
        clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &pixelDistance);
        clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &rainWidth);
        clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &rainHeight);
        clSetKernelArg(theKernel, ctr++, sizeof(float), &transparency);
        //----

        size_t gDim3[3];
        gDim3[0] = srcSize.width;
        gDim3[1] = srcSize.height;
        gDim3[2] = 1;
        cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);

        cl_kernel theKernel1;
        cl_program theProgram1;
        CreateProgramFromBinary(theQueue,"rain.cl","rain.cl.bin","rain",theProgram1,theKernel1);
        clRetainKernel(theKernel);
        ctr=0;
        clSetKernelArg(theKernel1, ctr++, sizeof(cl_mem), &srcPtr);
        clSetKernelArg(theKernel1, ctr++, sizeof(cl_mem), &dstPtr);
        clSetKernelArg(theKernel1, ctr++, sizeof(unsigned int), &srcSize.height);
        clSetKernelArg(theKernel1, ctr++, sizeof(unsigned int), &srcSize.width);
        clSetKernelArg(theKernel1, ctr++, sizeof(unsigned int), &channel);
        gDim3[2] = channel;
        cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram1, theKernel1);
    }

    return RPP_SUCCESS;   
}


/********************** Fog ************************/

RppStatus
fog_cl( cl_mem srcPtr, RppiSize srcSize, Rpp32f fogValue, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
{
    int ctr=0;
    cl_kernel theKernel;
    cl_program theProgram;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        
        CreateProgramFromBinary(theQueue,"fog.cl","fog.cl.bin","fog_planar",theProgram,theKernel);
        clRetainKernel(theKernel);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        CreateProgramFromBinary(theQueue,"fog.cl","fog.cl.bin","fog_pkd",theProgram,theKernel);
        clRetainKernel(theKernel);
    }
    else
    {std::cerr << "Internal error: Unknown Channel format";}
    //---- Args Setter
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    clSetKernelArg(theKernel, ctr++, sizeof(float), &fogValue);
    //----

    size_t gDim3[2];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = 1;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);

    return RPP_SUCCESS;   
}   
////////////////////////////////////////////////////

RppStatus
occlusion_cl(   cl_mem srcPtr1,RppiSize srcSize1, 
                cl_mem srcPtr2,RppiSize srcSize2, cl_mem dstPtr,
                const unsigned int x11,
                const unsigned int y11,
                const unsigned int x12,
                const unsigned int y12,
                const unsigned int x21,
                const unsigned int y21,
                const unsigned int x22,
                const unsigned int y22, 
                RppiChnFormat chnFormat,unsigned int channel,
                cl_command_queue theQueue)
{
    cl_kernel theKernel;
    cl_program theProgram;
    
    
    clRetainKernel(theKernel);
     if (chnFormat == RPPI_CHN_PLANAR)
    {
        
        CreateProgramFromBinary(theQueue,"occlusion.cl","occlusion.cl.bin","occlusion_pln",theProgram,theKernel);
        clRetainKernel(theKernel);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        CreateProgramFromBinary(theQueue,"occlusion.cl","occlusion.cl.bin","occlusion_pkd",theProgram,theKernel);
        clRetainKernel(theKernel);
    }
    else
    {std::cerr << "Internal error: Unknown Channel format";}

    //---- Args Setter
    int ctr =0;
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr1);
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr2);
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize1.height);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize1.width);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize2.height);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize2.width);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &x11);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &y11);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &x12);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &y12);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &x21);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &y21);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &x22);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &y22);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    //----

    size_t gDim3[3];
    gDim3[0] = srcSize2.width;
    gDim3[1] = srcSize2.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);

    return RPP_SUCCESS;
}
/********************** Random Shadow ************************/

RppStatus
random_shadow_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, Rpp32u numberOfShadows, Rpp32u maxSizeX, Rpp32u maxSizeY, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
{
    
    Rpp32u row1, row2, column2, column1;
    int i, j, ctr = 0;
    
    cl_kernel theKernel;
    cl_program theProgram;   
    CreateProgramFromBinary(theQueue, "random_shadow.cl", "random_shadow.cl.bin", "random_shadow", theProgram, theKernel);
    clRetainKernel(theKernel);  
    // cl_kernel_initializer(theQueue, "random_shadow.cl", "random_shadow", theProgram, theKernel);
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);

    for(i = 0 ; i < numberOfShadows ; i++)
    {
        ctr = 0;
        do
        {
            row1 = rand() % srcSize.height;
            column1 = rand() % srcSize.width;
        } while (column1 <= x1 || column1 >= x2 || row1 <= y1 || row1 >= y2);
        do
        {
            row2 = rand() % srcSize.height;
            column2 = rand() % srcSize.width;
        } while ((row2 < row1 || column2 < column1) || (column2 <= x1 || column2 >= x2 || row2 <= y1 || row2 >= y2) || (row2 - row1 >= maxSizeY || column2 - column1 >= maxSizeX));

        if(RPPI_CHN_PACKED == chnFormat)
        {
                CreateProgramFromBinary(theQueue, "random_shadow.cl", "random_shadow.cl.bin", "random_shadow_packed", theProgram, theKernel);
                clRetainKernel(theKernel);      
            // cl_kernel_initializer(theQueue,
            //                     "random_shadow.cl",
            //                     "random_shadow_packed",
            //                     theProgram, theKernel);
        }
        else
        {
                CreateProgramFromBinary(theQueue, "random_shadow.cl", "random_shadow.cl.bin", "random_shadow_planar", theProgram, theKernel);
                clRetainKernel(theKernel);
            // cl_kernel_initializer(theQueue,
            //                     "random_shadow.cl",
            //                     "random_shadow_planar",
            //                     theProgram, theKernel);
        }
        //---- Args Setter
        clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
        clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
        clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
        clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
        clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
        clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &column1);
        clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &row1);
        clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &column2);
        clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &row2);
        size_t gDim3[3];
        gDim3[0] = column2 - column1;
        gDim3[1] = row2 - row1;
        gDim3[2] = channel;
        cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    }
}

/********************** Dilate ************************/
RppStatus
dilate_cl ( cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
{
    int ctr=0;
    cl_kernel theKernel;
    cl_program theProgram;
    if(chnFormat == RPPI_CHN_PACKED)
    {
        CreateProgramFromBinary(theQueue,"dilate.cl","dilate.cl.bin","dilate_pkd",theProgram,theKernel);
        clRetainKernel(theKernel);
    }
    else
    {
        CreateProgramFromBinary(theQueue,"dilate.cl","dilate.cl.bin","dilate_pln",theProgram,theKernel);
        clRetainKernel(theKernel);
    }
    
    //---- Args Setter
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);
        
    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    
    return RPP_SUCCESS;    
}

/********************** Erode ************************/
RppStatus
erode_cl ( cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
{
    int ctr=0;
    cl_kernel theKernel;
    cl_program theProgram;
    if(chnFormat == RPPI_CHN_PACKED)
    {
        CreateProgramFromBinary(theQueue,"erode.cl","erode.cl.bin","erode_pkd",theProgram,theKernel);
        clRetainKernel(theKernel);
    }
    else
    {
        CreateProgramFromBinary(theQueue,"erode.cl","erode.cl.bin","erode_pln",theProgram,theKernel);
        clRetainKernel(theKernel);
    }
    
    //---- Args Setter
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &kernelSize);
        
    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    
    return RPP_SUCCESS;    
}

/********************** local binary pattern ************************/
RppStatus
local_binary_pattern_cl ( cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
{
    int ctr=0;
    cl_kernel theKernel;
    cl_program theProgram;
    if(chnFormat == RPPI_CHN_PACKED)
    {
        CreateProgramFromBinary(theQueue,"local_binary_pattern.cl","local_binary_pattern.cl.bin","local_binary_pattern_pkd",theProgram,theKernel);
        clRetainKernel(theKernel);
    }
    else
    {
        CreateProgramFromBinary(theQueue,"local_binary_pattern.cl","local_binary_pattern.cl.bin","local_binary_pattern_pln",theProgram,theKernel);
        clRetainKernel(theKernel);
    }
    
    //---- Args Setter
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
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

/********************** sobel ************************/
RppStatus
sobel_cl ( cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u sobelType, RppiChnFormat chnFormat, unsigned int channel, cl_command_queue theQueue)
{
    int ctr=0;
    cl_kernel theKernel;
    cl_program theProgram;
    if(chnFormat == RPPI_CHN_PACKED)
    {
        CreateProgramFromBinary(theQueue,"sobel.cl","sobel.cl.bin","sobel_pkd",theProgram,theKernel);
        clRetainKernel(theKernel);
    }
    else
    {
        CreateProgramFromBinary(theQueue,"sobel.cl","sobel.cl.bin","sobel_pln",theProgram,theKernel);
        clRetainKernel(theKernel);
    }
    
    //---- Args Setter
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &srcPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &sobelType);
        
    size_t gDim3[3];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    gDim3[2] = channel;
    cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    
    return RPP_SUCCESS;    
}