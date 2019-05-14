#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

__kernel void naive_convolution_planar(
	const __global unsigned char* input,
	__global  unsigned char* output,
	__global  float* filter,
    const unsigned short height,
    const unsigned short width,
    const unsigned short channel,
    const unsigned short filterSize
)
{

    unsigned short id_x = get_global_id(0);
    unsigned short id_y = get_global_id(1);
    unsigned short id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x + id_y * width + id_z * width * height;

    unsigned short hfFiltSz = filterSize/2;
    if ( (id_x < hfFiltSz) || (id_y < hfFiltSz) ||
            (id_y >= ((int) height-hfFiltSz)) || (id_x >= (width-hfFiltSz)) )
    {
        output[pixIdx] = input[pixIdx];
        return;
    }

    float sum = 0.0;
    for (int ri = -hfFiltSz , rf = 0;
            (ri < hfFiltSz) && (rf < filterSize);
                ri++, rf++)
    {
        for (int ci = -hfFiltSz , cf = 0;
                (ci < hfFiltSz) && (cf < filterSize);
                    ci++, cf++)
        {
            const int idxF = rf + cf * filterSize ;
            const int idxI = pixIdx + ri + ci * width;
            sum += filter[idxF]*input[idxI];
        }
    }

    output[pixIdx] = 1;//saturate_8u(sum);

}