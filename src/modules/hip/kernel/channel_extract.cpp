#include <hip/hip_runtime.h>
extern "C" __global__ void channel_extract_pln(   unsigned char* input,
                     unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned int extractChannelNumber
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    if (id_x >= width || id_y >= height) return;

    int OPpixIdx = id_x + id_y * width;
    int IPpixIdx = OPpixIdx + extractChannelNumber * width * height;
    output[OPpixIdx] = input[IPpixIdx];
}
extern "C" __global__ void channel_extract_pkd(   unsigned char* input,
                     unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned int extractChannelNumber
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    if (id_x >= width || id_y >= height) return;

    int OPpixIdx = id_y * width + id_x ;
    int IPpixIdx = id_y * width * channel + id_x * channel + extractChannelNumber;
    output[OPpixIdx] = input[IPpixIdx];
}

extern "C" __global__ void channel_extract_batch(   unsigned char* input,
                                     unsigned char* output,
                                     unsigned int* channelNumber,
                                     unsigned int *height,
                                     unsigned int *width,
                                     unsigned int *max_width,
                                     unsigned long *batch_index,
                                    const unsigned int channel,
                                     unsigned int *inc, // use width * height for pln and 1 for pkd
                                    const int plnpkdindex // use 1 pln 3 for pkd
                                    )
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;    int tempchannelNumber = channelNumber[id_z];
    int indextmp=0;
    unsigned long pixIdx = 0, outPixIdx = 0;
    if((id_y >= 0 ) && (id_y < height[id_z]) && (id_x >= 0) && (id_x < width[id_z]))
    {   
        pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex + tempchannelNumber;
        outPixIdx = (batch_index[id_z] / 3) + (id_x  + id_y * max_width[id_z]);
        output[outPixIdx] = input[pixIdx];
    }
}