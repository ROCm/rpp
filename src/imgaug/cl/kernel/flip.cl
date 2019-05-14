__kernel void flip_horizontal_planar(
	const __global unsigned char* input,
	__global  unsigned char* output,
    const unsigned short height,
    const unsigned short width,
    const unsigned short channel
)
{

    unsigned short id_x = get_global_id(0);
    unsigned short id_y = get_global_id(1);
    unsigned short id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    unsigned short oPixIdx =   id_x + id_y * width + id_z * width * height;

    unsigned short nPixIdx =   id_x + (height-1 - id_y) * width + id_z * width * height;

	output[nPixIdx] = input[oPixIdx];

}

__kernel void flip_vertical_planar(
	const __global unsigned char* input,
	__global  unsigned char* output,
    const unsigned short height,
    const unsigned short width,
    const unsigned short channel
)
{
    unsigned short id_x = get_global_id(0);
    unsigned short id_y = get_global_id(1);
    unsigned short id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    unsigned short oPixIdx =   id_x + id_y * width + id_z * width * height;

    unsigned short nPixIdx =   (width-1 - id_x) + id_y * width + id_z * width * height;

	output[nPixIdx] = input[oPixIdx];

}

__kernel void flip_bothaxis_planar(
	const __global unsigned char* input,
	__global  unsigned char* output,
    const unsigned short height,
    const unsigned short width,
    const unsigned short channel
)
{

    unsigned short id_x = get_global_id(0);
    unsigned short id_y = get_global_id(1);
    unsigned short id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    unsigned short oPixIdx =   id_x + id_y * width + id_z * width * height;

    unsigned short nPixIdx =   (width-1 - id_x) + (height-1 - id_y) * width + id_z * width * height;

	output[nPixIdx] = input[oPixIdx];

}
