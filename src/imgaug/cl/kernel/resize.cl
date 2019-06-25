#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value)   ))

__kernel void resize_pln (  __global unsigned char* srcPtr,
                            __global unsigned char* dstPtr,
                            const unsigned int source_height,
                            const unsigned int source_width,
                            const unsigned int dest_height,
                            const unsigned int dest_width,
                            const unsigned int channel
)
{
    int A, B, C, D, x, y, index, pixVal ;
    float x_ratio = ((float)(source_width -1 ))/dest_width ;
    float y_ratio = ((float)(source_height -1 ))/dest_height;
    float x_diff, y_diff, ya, yb ;

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= dest_width || id_y >= dest_height || id_z >= channel) return;

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

__kernel void resize_pkd (  __global unsigned char* srcPtr,
                            __global unsigned char* dstPtr,
                            const unsigned int source_height,
                            const unsigned int source_width,
                            const unsigned int dest_height,
                            const unsigned int dest_width,
                            const unsigned int channel
)
{
    int A, B, C, D, x, y, index, pixVal ;
    float x_ratio = ((float)(source_width -1 ))/dest_width ;
    float y_ratio = ((float)(source_height -1 ))/dest_height;
    float x_diff, y_diff, ya, yb ;

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= dest_width || id_y >= dest_height || id_z >= channel) return;

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


__kernel void resize_crop_pln (  __global unsigned char* srcPtr,
                            __global unsigned char* dstPtr,
                            const unsigned int source_height,
                            const unsigned int source_width,
                            const unsigned int dest_height,
                            const unsigned int dest_width,
                            const unsigned int x1,
                            const unsigned int y1,
                            const unsigned int x2,
                            const unsigned int y2,
                            const unsigned int channel
)
{
    int A, B, C, D, x, y, index, pixVal ;
    float x_ratio = ((float)(x2 - x1 ))/dest_width ;
    float y_ratio = ((float)(y2 - y1 ))/dest_height;
    float x_diff, y_diff, ya, yb ;
    A = B = C = D = 0;
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= dest_width || id_y >= dest_height || id_z >= channel) return;

    x =  (int)(x_ratio * id_x) ;
    y =  (int)(y_ratio * id_y) ;
    
    x_diff = (x_ratio * id_x) - x;
    y_diff = (y_ratio * id_y) - y ;

    unsigned int pixId;
    pixId = id_x + id_y * dest_width + id_z * dest_width * dest_height;
       
    A = srcPtr[(x + x1) + (y+y1) * source_width + id_z * source_height * source_width];
    B = srcPtr[(x + x1 + 1)  + (y+y1) * source_width + id_z * source_height * source_width];
    C = srcPtr[(x + x1)+ (y + y1 + 1) * source_width + id_z * source_height * source_width];
    D = srcPtr[(x+ x1 + 1) + (y+ y1 +1) * source_width + id_z * source_height * source_width];
    
    pixVal = (int)(  A*(1-x_diff)*(1-y_diff) +  B*(x_diff)*(1-y_diff) +
                    C*(y_diff)*(1-x_diff)   +  D*(x_diff*y_diff)
                    ) ;

    dstPtr[pixId] =  saturate_8u(pixVal);

  
}

__kernel void resize_crop_pkd (  __global unsigned char* srcPtr,
                            __global unsigned char* dstPtr,
                            const unsigned int source_height,
                            const unsigned int source_width,
                            const unsigned int dest_height,
                            const unsigned int dest_width,
                            const unsigned int x1,
                            const unsigned int y1,
                            const unsigned int x2,
                            const unsigned int y2,
                            const unsigned int channel
)
{
    int A, B, C, D, x, y, index, pixVal ;
    float x_ratio = ((float)(x2 - x1 ))/dest_width ;
    float y_ratio = ((float)(y2 - y1 ))/dest_height;
    float x_diff, y_diff, ya, yb ;
    A = B = C = D = 0;
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= dest_width || id_y >= dest_height || id_z >= channel) return;

    x =  (int)(x_ratio * id_x) ;
    y =  (int)(y_ratio * id_y) ;
    
    x_diff = (x_ratio * id_x) - x;
    y_diff = (y_ratio * id_y) - y ;

    unsigned int pixId;
    pixId = id_x * channel + id_y * dest_width * channel + id_z;

    A = srcPtr[(x+x1) * channel + (y+y1) * source_width * channel + id_z];
    B = srcPtr[(x +x1 +1) * channel + (y+y1) * source_width * channel + id_z];
    C = srcPtr[(x+x1) * channel + (y+ y1+ 1) * source_width * channel + id_z];
    D = srcPtr[(x+x1+1) * channel + (y+y1+1) * source_width * channel + id_z];

    pixVal = (int)(  A*(1-x_diff)*(1-y_diff) +  B*(x_diff)*(1-y_diff) +
                    C*(y_diff)*(1-x_diff)   +  D*(x_diff*y_diff)
                    ) ;

    dstPtr[pixId] =  saturate_8u(pixVal);

}