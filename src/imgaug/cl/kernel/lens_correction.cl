#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
__kernel void lenscorrection_pkd(
	    const __global unsigned char* input,
	    __global  unsigned char* output,
        const float strength,
        const float zoom,
        const float halfWidth,
        const float halfHeight,
        const float correctionRadius,
	    const unsigned int height,
	    const unsigned int width,
	    const unsigned int channel

)
{    
    int pix, pix_right, pix_right_down, pix_down, pixVal;
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int dstpixIdx = id_x * channel + id_y * width * channel + id_z;
    float theta;
    float newX = id_x - halfWidth;
    float newY = id_y - halfHeight;
    float r = (float)(sqrt(newX * newX + newY * newY)) / (float)correctionRadius;
    if (r == 0) 
        theta = 1.0;
    else
        theta = atan(r) / r;
    float new_idx = (halfWidth + theta * newX * zoom);
    float new_idy = (halfHeight + theta * newY * zoom);
    int x = (int) new_idx;
    int y = (int) new_idy;
    float x_diff = new_idx - x;
    float y_diff = new_idy - y;
    if ((x >= 0) && (y >= 0) && 
        (x < width - 2) && (y < height - 2))
    {
        pix = input[x * channel + y * width * channel + id_z];
        pix_right = input[(x +1) * channel + y * width * channel + id_z];
        pix_right_down = input[x * channel + (y+1) * width * channel + id_z];
        pix_down = input[(x+1) * channel + (y+1) * width * channel + id_z];

        pixVal = (int)(  pix*(1-x_diff)*(1-y_diff) +  pix_right*(x_diff)*(1-y_diff) +
                    pix_right_down*(y_diff)*(1-x_diff)   +  pix_down*(x_diff*y_diff)) ;
        output[dstpixIdx] =  saturate_8u(pixVal);
    }
    else {
        output[dstpixIdx] = 0;
    }
}

__kernel void lenscorrection_pln(
	    const __global unsigned char* input,
	    __global  unsigned char* output,
        const float strength,
        const float zoom,
        const float halfWidth,
        const float halfHeight,
        const float correctionRadius,
	    const unsigned int height,
	    const unsigned int width,
	    const unsigned int channel
)
{    
    int pix, pix_right, pix_right_down, pix_down, pixVal;
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int dstpixIdx = id_x + id_y * width + id_z  * channel;
    float newX = id_x - halfWidth;
    float newY = id_y - halfHeight;
    float r = (float)(sqrt(newX * newX + newY * newY)) / (float)correctionRadius;

    float theta;
    if (r == 0) 
        theta = 1.0;
    else
        theta = atan(r) / r;
    float new_idx = (halfWidth + theta * newX * zoom);
    float new_idy = (halfHeight + theta * newY * zoom);
    int x = (int) new_idx;
    int y = (int) new_idy;
    float x_diff = new_idx - x;
    float y_diff = new_idy - y;
    if ((x >= 0) && (y >= 0) && 
        (x < width - 2) && (y < height - 2))
    {
        pix = input[x + y * width + id_z * channel ];
        pix_right = input[(x +1)+ y * width + id_z * channel ];
        pix_right_down = input[x + (y+1) * width + id_z * channel  ];
        pix_down = input[(x+1) + (y+1) * width + id_z * channel ];

        pixVal = (int)(  pix*(1-x_diff)*(1-y_diff) +  pix_right*(x_diff)*(1-y_diff) +
                    pix_right_down*(y_diff)*(1-x_diff)   +  pix_down*(x_diff*y_diff)) ;
        output[dstpixIdx] =  saturate_8u(pixVal);
    }
    else {
        output[dstpixIdx] = 0;
    }
    // else if((x == width - 1) || (y == height - 1)){
    //     output[dstpixIdx] =  input[x + y * width + id_z * channel];;
    // }
}

