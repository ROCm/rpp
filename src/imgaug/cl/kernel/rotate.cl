
#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
#define PI 3.14159265
#define RAD(deg) (deg * PI / 180)

__kernel void rotate_pln (  __global unsigned char* srcPtr,
                            __global unsigned char* dstPtr,
                            const float angleDeg;
                            const unsigned int source_height,
                            const unsigned int source_width,
                            const unsigned int dest_height,
                            const unsigned int dest_width,
                            const unsigned int minX,
                            const unsigned int minY,
                            const unsigned int channel
)
{
    float angleRad = RAD(angleDeg);
    float rotate[4];
    rotate[0] = cos(-1 * angleRad);
    rotate[1] = sin(-1 * angleRad);
    rotate[2] = -1 * sin(-1 * angleRad);
    rotate[3] = cos(-1 * angleRad);

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    
    int k -= (Rpp32s)minX;
    int l -= (Rpp32s)minY;
    k = (Rpp32s)((rotate[0] * id_y) + (rotate[1] * id_x) ;
    l = (Rpp32s)((rotate[2] * id_y) + (rotate[3] * id_x) ;

    if (l < source_width && l >=0 && k < source_height && k >=0 )
    dstPtr[(id_z * dist_height * dest_width) + (id_y * dest_width) + id_x] =
                            srcPtr[(id_z * source_height * source_width) + (k * source_width) + l];
    else
    dstPtr[(id_z * dist_height * dest_width) + (id_y * dest_width) + id_x] = 0;
    

}

__kernel void rotate_pkd (  __global unsigned char* srcPtr,
                            __global unsigned char* dstPtr,
                            const float angleDeg;
                            const unsigned int source_height,
                            const unsigned int source_width,
                            const unsigned int dest_height,
                            const unsigned int dest_width,
                            const unsigned int minX,
                            const unsigned int minY,
                            const unsigned int channel
)
{
    float angleRad = RAD(angleDeg);
    float rotate[4];
    rotate[0] = cos(-1 * angleRad);
    rotate[1] = sin(-1 * angleRad);
    rotate[2] = -1 * sin(-1 * angleRad);
    rotate[3] = cos(-1 * angleRad);

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);

    int k -= (Rpp32s)minX;
    int l -= (Rpp32s)minY;
    k = (Rpp32s)((rotate[0] * id_y) + (rotate[1] * id_x) ;
    l = (Rpp32s)((rotate[2] * id_y) + (rotate[3] * id_x) ;
    
    if (l < source_width && l >=0 && k < source_height && k >=0 )
    dstPtr[id_z + (channel * id_y * dstSize.width) + (channel * id_x)] =
                             srcPtr[id_z + (channel * k * srcSize.width) + (channel * l)];
    else
    dstPtr[(id_z * dist_height * dest_width) + (id_y * dest_width) + id_x] = 0;
    

}

