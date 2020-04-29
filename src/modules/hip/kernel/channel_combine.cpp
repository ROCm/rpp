#include <hip/hip_runtime.h>
extern "C" __global__ void channel_combine_pln(   unsigned char* input1,
                     unsigned char* input2,
                     unsigned char* input3,
                     unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    if (id_x >= width || id_y >= height) return;

    int IPpixIdx = id_x + id_y * width;
    int OPpixIdx1 = IPpixIdx;
    int OPpixIdx2 = IPpixIdx + width * height;
    int OPpixIdx3 = IPpixIdx + 2 * width * height;

    output[OPpixIdx1] = input1[IPpixIdx];
    output[OPpixIdx2] = input2[IPpixIdx];
    output[OPpixIdx3] = input3[IPpixIdx];
}
extern "C" __global__ void channel_combine_pkd(   unsigned char* input1,
                     unsigned char* input2,
                     unsigned char* input3,
                     unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    if (id_x >= width || id_y >= height) return;

    int IPpixIdx = id_x + id_y * width;
    int OPpixIdx = IPpixIdx * channel;
    output[OPpixIdx] = input1[IPpixIdx];
    output[OPpixIdx + 1] = input2[IPpixIdx];
    output[OPpixIdx + 2] = input3[IPpixIdx];
}

extern "C" __global__ void channel_combine_batch(   unsigned char* input1,
                                     unsigned char* input2,
                                     unsigned char* input3,
                                     unsigned char* output,
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
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;    int indextmp=0;
    unsigned long pixIdx = 0, InPixIdx = 0;
    if((id_y >= 0 ) && (id_y < height[id_z]) && (id_x >= 0) && (id_x < width[id_z]))
    {   
        pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex;
        InPixIdx = (batch_index[id_z] / 3) + (id_x  + id_y * max_width[id_z]);
        output[pixIdx] = input1[InPixIdx];
        output[pixIdx + inc[id_z]] = input2[InPixIdx];
        output[pixIdx + inc[id_z] * 2] = input3[InPixIdx];
    }
}