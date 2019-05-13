__kernel void flip_horizontal_planar(
	const __global unsigned char* input,
	__global  unsigned char* output,
    const size_t height,
    const size_t width,
    const size_t channel
)
{

    int oPixIdx =   get_global_id(0) +
                    get_global_id(1)* width +
                    get_global_id(2)* width * height;

    int nPixIdx =   get_global_id(0) +
                    (height-1 - get_global_id(1)) * width +
                    get_global_id(2)* width * height;

	output[nPixIdx] = input[oPixIdx];

}

__kernel void flip_vertical_planar(
	const __global unsigned char* input,
	__global  unsigned char* output,
    const size_t height,
    const size_t width,
    const size_t channel
)
{

    int oPixIdx =   get_global_id(0) +
                    get_global_id(1)* width +
                    get_global_id(2)* width * height;

    int nPixIdx =   (width-1 - get_global_id(0)) +
                    get_global_id(1) * width +
                    get_global_id(2)* width * height;

	output[nPixIdx] = input[oPixIdx];

}

__kernel void flip_bothaxis_planar(
	const __global unsigned char* input,
	__global  unsigned char* output,
    const size_t height,
    const size_t width,
    const size_t channel
)
{

    int oPixIdx =   get_global_id(0) +
                    get_global_id(1)* width +
                    get_global_id(2)* width * height;

    int nPixIdx =   (width-1 - get_global_id(0)) +
                    (height-1 - get_global_id(1)) * width +
                    get_global_id(2)* width * height;

	output[nPixIdx] = input[oPixIdx];

}
