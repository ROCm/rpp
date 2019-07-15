//==============================================================================
// Only compiled with hipcc compiler
//==============================================================================

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <rpp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

int main( int argc, char* argv[] )
{
    typedef unsigned char TYPE_t;
    TYPE_t* h_a;
    TYPE_t* h_b;
    TYPE_t* h_c;
    int height;
    int width;
    int channel = 3;

    h_a = stbi_load( "/home/mcw/Desktop/AMDRPP/sample_test/images/Image1.jpg",
                        &width, &height, &channel, 0);
    h_b = stbi_load( "/home/mcw/Desktop/AMDRPP/sample_test/images/Image2.jpg",
                        &width, &height, &channel, 0);
    size_t n = height * width * channel;
    size_t bytes = n*sizeof(TYPE_t);
    RppiSize dstSize;
    dstSize.height = height;
    dstSize.width = width;
    size_t dest_bytes = dstSize.width * dstSize.height * channel* sizeof(TYPE_t);
    h_c = (TYPE_t*)malloc(dest_bytes);

    std::cout << "width:" << width << std::endl;
    std::cout << "height:" << height << std::endl;
    std::cout << "channel:" << channel << std::endl;

    RppiSize srcSize;
    srcSize.height=height;
    srcSize.width=width;
    
    Rpp32f alpha=0.5;

    Rpp32f noiseParameter=0.2;
    RppiNoise noiseType=SNP;
    
    RppiGaussParameter np;
    Rpp32f mean=20;
    Rpp32f sigma=30;
    np.mean=mean;
    np.sigma=sigma;
    RppiNoise nt=GAUSSIAN;
    
    //Rpp32f mean;
    Rpp32f sd;

    rppi_accumulate_weighted_u8_pkd3_host(h_a, h_b, srcSize, alpha);
    stbi_write_png("/home/mcw/Desktop/AMDRPP/sample_test/images/ACCUMULATE_WEIGHT_HOST.png",
                          dstSize.width,dstSize.height, channel, h_a, dstSize.width *channel);

    // rppi_noiseAdd_u8_pkd3_host(h_a, srcSize, h_c, noiseType, &noiseParameter);
    // stbi_write_png("/home/mcw/Desktop/AMDRPP/sample_test/images/2.SaltAndPepper.png",
    //                       dstSize.width,dstSize.height, channel, h_c, dstSize.width *channel);
    
    // rppi_noiseAdd_u8_pkd3_host(h_a, srcSize, h_c, nt, &np);
    // stbi_write_png("/home/mcw/Desktop/AMDRPP/sample_test/images/3.Gaussian.png",
    //                       dstSize.width,dstSize.height, channel, h_c, dstSize.width *channel);
    
    // rppi_mean_stddev_u8_pkd3_host(h_a, srcSize, &mean, &sd);
    // std::cout << "mean:" << mean << std::endl;
    // std::cout << "sd:" << sd << std::endl;

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
