
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
    float rotate[6] = {0};
    rotate[0] = cos(angleRad);
    rotate[1] = sin(angleRad);
    rotate[2] = 0;
    rotate[3] = -sin(angleRad);
    rotate[4] = cos(angleRad);
    rotate[5] = 0;

    int k = (Rpp32s)((affine[0] * i) + (affine[1] * j) + (affine[2] * 1));
    int l = (Rpp32s)((affine[3] * i) + (affine[4] * j) + (affine[5] * 1));
    k -= (Rpp32s)minX;
    l -= (Rpp32s)minY;
    dstPtr[(c * dist_height * dest_width) + (k * dest_width) + l] =
                            srcPtr[(c * source_height * source_width) + (i * source_width) + j];

}


/*__kernel void smoothen_pln()
{

}*/