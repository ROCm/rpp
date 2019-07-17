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
    
    if (id_x >= dest_width || id_y >= dest_height || id_z >= channel) return;
    
    if ((id_x >= x21) && (id_x <= x22) && (id_y>= y21) && (id_y >= y22))
    {
        x =  (int)(x_ratio * (id_x - x21));
        y =  (int)(y_ratio * (id_y - y21));
        
        x_diff = (x_ratio * id_x) - (x + x21);
        y_diff = (y_ratio * id_y) - (y + y21) ;

        
        pixId = id_x + id_y * dest_width + id_z * dest_width * dest_height;
    
    
        A = srcPtr1[(x + x11) + (y+y11) * source_width + id_z * source_height * source_width];
        B = srcPtr1[(x + x11 + 1)  + (y+y11) * source_width + id_z * source_height * source_width];
        C = srcPtr1[(x + x11)+ (y + y11 + 1) * source_width + id_z * source_height * source_width];
        D = srcPtr1[(x+ x11 + 1) + (y+ y11 +1) * source_width + id_z * source_height * source_width];
        
        pixVal = (int)(  A*(1-x_diff)*(1-y_diff) +  B*(x_diff)*(1-y_diff) +
                        C*(y_diff)*(1-x_diff)   +  D*(x_diff*y_diff)
                        ) ;

        dstPtr[pixId] =  saturate_8u(pixVal);
    }
    else
        dstPtr[pixId] =  srcPtr2[pixId];
  
}

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
    
    if (id_x >= dest_width || id_y >= dest_height || id_z >= channel) return;
    
    if ((id_x >= x21) && (id_x <= x22) && (id_y>= y21) && (id_y >= y22))
    {
        x =  (int)(x_ratio * (id_x - x21));
        y =  (int)(y_ratio * (id_y - y21));
        
        x_diff = (x_ratio * id_x) - (x + x21);
        y_diff = (y_ratio * id_y) - (y + y21) ;

        
        pixId = id_x * channel + id_y * dest_width * channel + id_z;
    
    
        A = srcPtr1[(x + x11) + (y+y11) * source_width + id_z * source_height * source_width];
        B = srcPtr1[(x + x11 + 1)  + (y+y11) * source_width + id_z * source_height * source_width];
        C = srcPtr1[(x + x11)+ (y + y11 + 1) * source_width + id_z * source_height * source_width];
        D = srcPtr1[(x+ x11 + 1) + (y+ y11 +1) * source_width + id_z * source_height * source_width];
        
        pixVal = (int)(  A*(1-x_diff)*(1-y_diff) +  B*(x_diff)*(1-y_diff) +
                        C*(y_diff)*(1-x_diff)   +  D*(x_diff*y_diff)
                        ) ;

        dstPtr[pixId] =  saturate_8u(pixVal);
    }
    else
        dstPtr[pixId] =  srcPtr2[pixId];
  
}
