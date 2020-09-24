#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

__kernel void occlusion_pln (  __global unsigned char* srcPtr1,
                            __global unsigned char* srcPtr2,
                            __global unsigned char* dstPtr,
                            const unsigned int source_height1,
                            const unsigned int source_width1,
                            const unsigned int dest_height,
                            const unsigned int dest_width,
                            const unsigned int x11,
                            const unsigned int y11,
                            const unsigned int x12,
                            const unsigned int y12,
                            const unsigned int x21,
                            const unsigned int y21,
                            const unsigned int x22,
                            const unsigned int y22,
                            const unsigned int channel
)
{
    int A, B, C, D, x, y, index, pixVal ;
    float x_ratio = ((float)(x12 - x11 +1 )/(x22-x21 +1)) ;
    float y_ratio = ((float)(y12 - y11 +1)/(y22 - y21 +1));
    float x_diff, y_diff;
    A = B = C = D = 0;
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    unsigned int pixId;
    pixId = id_x + id_y * dest_width + id_z * dest_width * dest_height;
    if (id_x >= dest_width || id_y >= dest_height || id_z >= channel) return;
    
    if ((id_x >= x21) && (id_x <= x22) && (id_y>= y21) && (id_y <= y22))
    {
        x =  (int)(x_ratio * (id_x - x21));
        y =  (int)(y_ratio * (id_y - y21));
        
        x_diff = (x_ratio * (id_x - x21)) - x ;
        y_diff = (y_ratio * (id_y - x21)) - y ;

    
        A = srcPtr1[(x + x11) + (y+y11) * source_width1 + id_z * source_height1 * source_width1];
        B = srcPtr1[(x + x11 + 1)  + (y+y11) * source_width1 + id_z * source_height1 * source_width1];
        C = srcPtr1[(x + x11)+ (y + y11 + 1) * source_width1 + id_z * source_height1 * source_width1];
        D = srcPtr1[(x+ x11 + 1) + (y+ y11 +1) * source_width1 + id_z * source_height1 * source_width1];
        
        pixVal = (int)(  A*(1-x_diff)*(1-y_diff) +  B*(x_diff)*(1-y_diff) +
                        C*(y_diff)*(1-x_diff)   +  D*(x_diff*y_diff)
                        ) ;

        dstPtr[pixId] =  saturate_8u(pixVal);
    }
    else
        dstPtr[pixId] =  srcPtr2[pixId];
  
}

__kernel void occlusion_pkd (  __global unsigned char* srcPtr1,
                            __global unsigned char* srcPtr2,
                            __global unsigned char* dstPtr,
                            const unsigned int source_height1,
                            const unsigned int source_width1,
                            const unsigned int dest_height,
                            const unsigned int dest_width,
                            const unsigned int x11,
                            const unsigned int y11,
                            const unsigned int x12,
                            const unsigned int y12,
                            const unsigned int x21,
                            const unsigned int y21,
                            const unsigned int x22,
                            const unsigned int y22,
                            const unsigned int channel
)
{
    int A, B, C, D, x, y, index, pixVal ;
    float x_ratio = ((float)(x12 - x11 +1 )/(x22- x21 +1)) ;
    float y_ratio = ((float)(y12 - y11 +1)/(y22 - y21 +1));
    float x_diff, y_diff;
    A = B = C = D = 0;
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    unsigned int pixId;
    pixId = id_x * channel + id_y * dest_width * channel + id_z;

    if (id_x >= dest_width || id_y >= dest_height || id_z >= channel) return;
    
    if ((id_x >= x21) && (id_x <= x22) && (id_y>= y21) && (id_y <= y22))
    {
        x =  (int)(x_ratio * (id_x - x21));
        y =  (int)(y_ratio * (id_y - y21));
        
        x_diff = (x_ratio * (id_x - x21)) - x ;
        y_diff = (y_ratio * (id_y - x21)) - y ;

        A = srcPtr1[(x + x11) * channel + (y+y11) * source_width1 * channel + id_z ];
        B = srcPtr1[(x + x11 + 1) *channel  + (y+y11) * source_width1 * channel + id_z];
        C = srcPtr1[(x + x11) * channel+ (y + y11 + 1) * source_width1 * channel + id_z ];
        D = srcPtr1[(x+ x11 + 1) * channel + (y+ y11 +1) * source_width1 * channel + id_z];
        pixVal = (int)(  A*(1-x_diff)*(1-y_diff) +  B*(x_diff)*(1-y_diff) +
                        C*(y_diff)*(1-x_diff)   +  D*(x_diff*y_diff)
                        ) ;
        dstPtr[pixId] =  saturate_8u(A);
    }
    else
        dstPtr[pixId] =  srcPtr2[pixId];
  
}


__kernel void occlusion_batch (  __global unsigned char* srcPtr1,
                            __global unsigned char* srcPtr2,
                            __global unsigned char* dstPtr,
                            __global unsigned int *source_height,
                            __global unsigned int *source_width,
                            __global unsigned int *dest_height,
                            __global unsigned int *dest_width,
                            __global unsigned int *x11,
                            __global unsigned int *y11,
                            __global unsigned int *x12,
                            __global unsigned int *y12,
                            __global unsigned int *x21,
                            __global unsigned int *y21,
                            __global unsigned int *x22,
                            __global unsigned int *y22,
                            __global unsigned int *max_source_width,
                            __global unsigned int *max_dest_width,
                            __global unsigned long *source_batch_index,
                            __global unsigned long *dest_batch_index,
                            const unsigned int channel,
                            __global unsigned int *source_inc, // use width * height for pln and 1 for pkd
                            __global unsigned int *dest_inc,
                            const int plnpkdindex // use 1 pln 3 for pkd
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    int A, B, C, D, x, y, index, pixVal ;
    float x_ratio = ((float)(x12[id_z] - x11[id_z] +1 )/(x22[id_z]- x21[id_z] +1)) ;
    float y_ratio = ((float)(y12[id_z] - y11[id_z] +1)/(y22[id_z] - y21[id_z] +1));
    float x_diff, y_diff;
    A = B = C = D = 0;
    
    int indextmp=0;
    unsigned long  dst_pixIdx = 0;

    dst_pixIdx = dest_batch_index[id_z] + (id_x  + id_y * max_dest_width[id_z] ) * plnpkdindex;
    if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z]) return;
    
    if ((id_x >= x21[id_z]) && (id_x <= x22[id_z]) && (id_y>= y21[id_z]) && (id_y <= y22[id_z]))
    {
        x =  (int)(x_ratio * (id_x - x21[id_z]));
        y =  (int)(y_ratio * (id_y - y21[id_z]));
        
        x_diff = (x_ratio * (id_x - x21[id_z])) - x ;
        y_diff = (y_ratio * (id_y - y21[id_z])) - y ;

         for(indextmp = 0; indextmp < channel; indextmp++){
            A = srcPtr1[source_batch_index[id_z] + ((x + x11[id_z])  + (y + y11[id_z]) * max_source_width[id_z]) * plnpkdindex+ indextmp*source_inc[id_z]]; 
            B = srcPtr1[source_batch_index[id_z] + ((x + x11[id_z] + 1)   + (y + y11[id_z]) * max_source_width[id_z]) * plnpkdindex+ indextmp*source_inc[id_z]];
            C = srcPtr1[source_batch_index[id_z] + ((x + x11[id_z])  + (y + y11[id_z] + 1) * max_source_width[id_z]) * plnpkdindex+ indextmp*source_inc[id_z]];
            D = srcPtr1[source_batch_index[id_z] + ((x + x11[id_z]+1)  + (y + y11[id_z] + 1) * max_source_width[id_z]) * plnpkdindex+ indextmp*source_inc[id_z]];

            pixVal = (int)(  A*(1-x_diff)*(1-y_diff) +  B*(x_diff)*(1-y_diff) +
                        C*(y_diff)*(1-x_diff)   +  D*(x_diff*y_diff)
                        ) ;
            dstPtr[dst_pixIdx] =  saturate_8u(A);
            dst_pixIdx += dest_inc[id_z];
            source_batch_index += source_inc[id_z];
         }
    }
    else
    {
        dstPtr[dst_pixIdx] =  srcPtr2[dst_pixIdx];
        dstPtr[dstPtr + dest_inc[id_z]] = srcPtr2[dstPtr + dest_inc[id_z]];
        dstPtr[dstPtr + 2*dest_inc[id_z]] = srcPtr2[dstPtr + 2*dest_inc[id_z]];
    }

}
