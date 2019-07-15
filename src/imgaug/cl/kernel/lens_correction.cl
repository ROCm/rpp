#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
__kernel void lenscorrection_pkd(
	    const __global unsigned char* input,
	    __global  unsigned char* output,
	    const unsigned int height,
	    const unsigned int width,
	    const unsigned int channel,
        const float strength,
        const float zoom
)
{    
    int A, B, C, D, pixVal;
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int dstpixIdx = id_x * channel + id_y * width * channel + id_z;

    float halfWidth = (float)width / 2.0;
    float halfHeight = (float)height / 2.0;
    float correctionRadius = (float)sqrt((float)width * width + height * height) / (float)strength;
     
    float newX = id_x - halfWidth;
    float newY = id_y - halfHeight;
    float r = (float)(sqrt(newX * newX + newY * newY)) / (float)correctionRadius;

    float theta;
    if (r == 0) 
        theta = 1.0;
    else
        theta = atan(r) / r;

    int x = (int) (halfWidth + theta * newX * zoom);
    int y = (int) (halfHeight + theta * newY * zoom);
    float x_diff = (halfWidth + theta * newX * zoom) - x;
    float y_diff = (halfHeight + theta * newY * zoom) - y;
    if ((x >= 0) && (y >= 0) && 
        (x < width) && (y < height))
    {
        A = input[x * channel + y * width * channel + id_z];
        B = input[(x +1) * channel + y * width * channel + id_z];
        C = input[x * channel + (y+1) * width * channel + id_z];
        D = input[(x+1) * channel + (y+1) * width * channel + id_z];

        pixVal = (int)(  A*(1-x_diff)*(1-y_diff) +  B*(x_diff)*(1-y_diff) +
                    C*(y_diff)*(1-x_diff)   +  D*(x_diff*y_diff)) ;
        output[dstpixIdx] =  saturate_8u(pixVal);
    }
}

__kernel void lenscorrection_pln(
	    const __global unsigned char* input,
	    __global  unsigned char* output,
	    const unsigned int height,
	    const unsigned int width,
	    const unsigned int channel,
        const float strength,
        const float zoom
)
{    
    int A, B, C, D, pixVal;
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int dstpixIdx = id_x + id_y * width + id_z  * channel;

    float halfWidth = (float)width / 2.0;
    float halfHeight = (float)height / 2.0;
    float correctionRadius = (float)sqrt((float)width * width + height * height) / (float)strength;
     
    float newX = id_x - halfWidth;
    float newY = id_y - halfHeight;
    float r = (float)(sqrt(newX * newX + newY * newY)) / (float)correctionRadius;

    float theta;
    if (r == 0) 
        theta = 1.0;
    else
        theta = atan(r) / r;

    int x = (int) (halfWidth + theta * newX * zoom);
    int y = (int) (halfHeight + theta * newY * zoom);
    float x_diff = (halfWidth + theta * newX * zoom) - x;
    float y_diff = (halfHeight + theta * newY * zoom) - y;
    if ((x >= 0) && (y >= 0) && 
        (x < width) && (y < height))
    {
        A = input[x + y * width + id_z * channel ];
        B = input[(x +1)+ y * width + id_z * channel ];
        C = input[x + (y+1) * width + id_z * channel  ];
        D = input[(x+1) + (y+1) * width + id_z * channel ];

        pixVal = (int)(  A*(1-x_diff)*(1-y_diff) +  B*(x_diff)*(1-y_diff) +
                    C*(y_diff)*(1-x_diff)   +  D*(x_diff*y_diff)) ;
        output[dstpixIdx] =  saturate_8u(pixVal);
    }
}

