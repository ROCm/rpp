#include <hip/hip_runtime.h>
#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value)   ))

extern "C" __global__ void scale_pln (   unsigned char* srcPtr,
                             unsigned char* dstPtr,
                            const unsigned int source_height,
                            const unsigned int source_width,
                            const unsigned int dest_height,
                            const unsigned int dest_width,
                            const unsigned int channel,
                            const unsigned int exp_dest_height,
                            const unsigned int exp_dest_width
)
{
    int A, B, C, D, x, y, index, pixVal ;
    float x_ratio = ((float)(source_width -1 ))/exp_dest_width ;
    float y_ratio = ((float)(source_height -1 ))/exp_dest_height;
    float x_diff, y_diff, ya, yb ;

    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= dest_width || id_y >= dest_height || id_z >= channel || id_x >= exp_dest_width || id_y >= exp_dest_height) return;

    x = (int)(x_ratio * id_x) ;
    y = (int)(y_ratio * id_y) ;

    x_diff = (x_ratio * id_x) - x ;
    y_diff = (y_ratio * id_y) - y ;

    unsigned int pixId;
    pixId = id_x + id_y * dest_width + id_z * dest_width * dest_height;
    A = srcPtr[x + y * source_width + id_z * source_height * source_width];
    B = srcPtr[x + 1  + y * source_width + id_z * source_height * source_width];
    C = srcPtr[x + (y + 1) * source_width + id_z * source_height * source_width];
    D = srcPtr[(x+1) + (y+1) * source_width + id_z * source_height * source_width];

    pixVal = (int)(  A*(1-x_diff)*(1-y_diff) +  B*(x_diff)*(1-y_diff) +
                    C*(y_diff)*(1-x_diff)   +  D*(x_diff*y_diff)
                    ) ;

    dstPtr[pixId] =  saturate_8u(pixVal);

}

extern "C" __global__ void scale_pkd (   unsigned char* srcPtr,
                             unsigned char* dstPtr,
                            const unsigned int source_height,
                            const unsigned int source_width,
                            const unsigned int dest_height,
                            const unsigned int dest_width,
                            const unsigned int channel,
                            const unsigned int exp_dest_height,
                            const unsigned int exp_dest_width
)
{
    int A, B, C, D, x, y, index, pixVal ;
    float x_ratio = ((float)(source_width -1 ))/exp_dest_width ;
    float y_ratio = ((float)(source_height -1 ))/exp_dest_height;
    float x_diff, y_diff, ya, yb ;

    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= dest_width || id_y >= dest_height || id_z >= channel || id_x >= exp_dest_width || id_y >= exp_dest_height) return;

    x = (int)(x_ratio * id_x) ;
    y = (int)(y_ratio * id_y) ;

    x_diff = (x_ratio * id_x) - x ;
    y_diff = (y_ratio * id_y) - y ;

    unsigned int pixId;
    pixId = id_x * channel + id_y * dest_width * channel + id_z;

    A = srcPtr[x * channel + y * source_width * channel + id_z];
    B = srcPtr[(x +1) * channel + y * source_width * channel + id_z];
    C = srcPtr[x * channel + (y+1) * source_width * channel + id_z];
    D = srcPtr[(x+1) * channel + (y+1) * source_width * channel + id_z];

    pixVal = (int)(  A*(1-x_diff)*(1-y_diff) +  B*(x_diff)*(1-y_diff) +
                  C*(y_diff)*(1-x_diff)   +  D*(x_diff*y_diff)) ;
    dstPtr[pixId] =  saturate_8u(pixVal);

}


extern "C" __global__ void scale_batch(     unsigned char* srcPtr,
                                     unsigned char* dstPtr,
                                     float* percentage,
                                     unsigned int *source_height,
                                     unsigned int *source_width,
                                     unsigned int *dest_height,
                                     unsigned int *dest_width,
                                     unsigned int *max_source_width,
                                     unsigned int *max_dest_width,
                                      int *xroi_begin,
                                     int *xroi_end,
                                      int *yroi_begin,
                                      int *yroi_end,
                                      unsigned long *source_batch_index,
                                     unsigned long *dest_batch_index,
                                      const unsigned int channel,
                                     unsigned int *source_inc, // use width * height for pln and 1 for pkd
                                     unsigned int *dest_inc,
                                    const int plnpkdindex // use 1 pln 3 for pkd
                                    )
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x, id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y, id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    int A, B, C, D, x, y, index, pixVal ;
    int expdest_height, expdest_width;
    expdest_width = percentage[id_z] * dest_width[id_z];
    expdest_height = percentage[id_z] * dest_height[id_z];

    float x_ratio = ((float)(xroi_end[id_z] - xroi_begin[id_z] -1 ))/expdest_width;
    float y_ratio = ((float)(yroi_end[id_z] - yroi_begin[id_z] -1 ))/expdest_height;
    
    float x_diff, y_diff, ya, yb ;

    int indextmp=0;
    unsigned long src_pixIdx = 0, dst_pixIdx = 0;
    if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z] || id_x >= expdest_width || id_y >=expdest_height) return;
    x = (int)(x_ratio * id_x) ;
    y = (int)(y_ratio * id_y) ;

    x_diff = (x_ratio * id_x) - x ;
    y_diff = (y_ratio * id_y) - y ;

    x = xroi_begin[id_z] + x;
    y = yroi_begin[id_z] + y;
    
    if ((x +1) < source_width[id_z] && (y+1) < source_height[id_z]){
        dst_pixIdx = dest_batch_index[id_z] + (id_x  + id_y * max_dest_width[id_z] ) * plnpkdindex;
        for(indextmp = 0; indextmp < channel; indextmp++){
            A = srcPtr[source_batch_index[id_z] + (x  + y * max_source_width[id_z]) * plnpkdindex + indextmp*source_inc[id_z]]; 
            B = srcPtr[source_batch_index[id_z] + ((x + 1)   + y * max_source_width[id_z]) * plnpkdindex + indextmp*source_inc[id_z]];
            C = srcPtr[source_batch_index[id_z] + (x  + (y + 1) * max_source_width[id_z]) * plnpkdindex + indextmp*source_inc[id_z]];
            D = srcPtr[source_batch_index[id_z] + ((x + 1)  + (y + 1) * max_source_width[id_z]) * plnpkdindex + indextmp*source_inc[id_z]];

            pixVal = (int)(  A*(1-x_diff)*(1-y_diff) +  B*(x_diff)*(1-y_diff) +
                        C*(y_diff)*(1-x_diff)   +  D*(x_diff*y_diff)) ;
            dstPtr[dst_pixIdx] =  saturate_8u(pixVal);
            dst_pixIdx += dest_inc[id_z];
        }
    }
    
}