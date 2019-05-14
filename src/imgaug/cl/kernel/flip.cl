__kernel void flip_horizontal_planar(
	const __global unsigned char* input,
	__global  unsigned char* output,
    const unsigned int height,
    const unsigned int width,
    const unsigned int channel
)
{

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int oPixIdx =   id_x + id_y * width + id_z * width * height;

    int nPixIdx =   id_x + (height-1 - id_y) * width + id_z * width * height;

	output[nPixIdx] = input[oPixIdx];

}

__kernel void flip_vertical_planar(
	const __global unsigned char* input,
	__global  unsigned char* output,
    const unsigned int height,
    const unsigned int width,
    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int oPixIdx =   id_x + id_y * width + id_z * width * height;

    // TODO:Vertical flip has to be fixed

    int nPixIdx =   (width-1 - id_x) + id_y * width + id_z * width * height;

	output[nPixIdx] = input[oPixIdx];

}

__kernel void flip_bothaxis_planar(
	const __global unsigned char* input,
	__global  unsigned char* output,
    const unsigned int height,
    const unsigned int width,
    const unsigned int channel
)
{

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int oPixIdx =   id_x + id_y * width + id_z * width * height;

    // TODO:Vertical flip has to be fixed
    //int nPixIdx =   (width-1 - id_x) + (height-1 - id_y) * width + id_z * width * height;

    int nPixIdx =   id_x + (height-1 - id_y) * width + id_z * width * height;



    output[nPixIdx] = input[oPixIdx];

}
