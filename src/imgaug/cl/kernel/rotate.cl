
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
    float rotate[6];
    rotate[0] = cos(angleRad);
    rotate[1] = sin(angleRad);
    rotate[2] = 0;
    rotate[3] = -1 * sin(angleRad);
    rotate[4] = cos(angleRad);
    rotate[5] = 0;

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    
    int k = (Rpp32s)((affine[0] * id_y) + (affine[1] * id_x) + (affine[2] * 1));
    int l = (Rpp32s)((affine[3] * id_y) + (affine[4] * id_x) + (affine[5] * 1));
    k -= (Rpp32s)minX;
    l -= (Rpp32s)minY;
    dstPtr[(id_z * dist_height * dest_width) + (k * dest_width) + l] =
                            srcPtr[(id_z * source_height * source_width) + (id_y * source_width) + id_x];

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
    float rotate[6] = {0};
    rotate[0] = cos(angleRad);
    rotate[1] = sin(angleRad);
    rotate[2] = 0;
    rotate[3] = -1 * sin(angleRad);
    rotate[4] = cos(angleRad);
    rotate[5] = 0;

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    
    int k = (Rpp32s)((affine[0] * id_y) + (affine[1] * id_x) + (affine[2] * 1));
    int l = (Rpp32s)((affine[3] * id_y) + (affine[4] * id_x) + (affine[5] * 1));
    k -= (Rpp32s)minX;
    l -= (Rpp32s)minY;

    dstPtr[id_z + (channel * k * dstSize.width) + (channel * l)] =
                             srcPtr[id_z + (channel * id_y * srcSize.width) + (channel * id_x)];

}


/*__kernel void smoothen_pln()
{

}*/