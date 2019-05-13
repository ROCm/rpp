__kernel void gaussian_blur_planar(
	const __global unsigned char* input,
	__global  unsigned char* output,
	__global  float* filter,
    const size_t height,
    const size_t width,
    const size_t channel,
    const size_t filterSize
)
{

    int pixIdx = get_global_id(0) + get_global_id(1)* width +
                 get_global_id(2)* width * height;

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);

    if ((id_x ==0) || (id_y ==0) || (id_y == (height-1)) || (id_x == (width-1)))
    {
        output[pixIdx] = input[pixIdx];
        return;
    }

    float sum = 0.0;
    for (int r = 0; r < filterSize; r++)
    {
        for (int c = 0; c < filterSize; c++)
        {
            const int idxF = r * filterSize + c;
            sum += filter[idxF]*input[pixIdx];
        }
    } //for (int r = 0...

	output[pixIdx] = sum;

}