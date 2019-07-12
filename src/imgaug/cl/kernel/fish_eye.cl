__kernel void fisheye_planar(
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

    int dstpixIdx = id_x + id_y * width  + id_z  * channel;
    float normY = ((float)(2*id_y)/(float) (height))-1; 
    float normYSquare = normY*normY;
    float normX = ((float)(2*id_x)/(float) (width))-1;
    float normXSquare = normX*normX;
    float dist = sqrt(normXSquare+normYSquare);
    if ((0.0 <= dist) && (dist <= 1.0)) {
        float newDist = sqrt(1.0-dist*dist);
        newDist = (dist + (1.0-newDist)) / 2.0;
        if (newDist <= 1.0) {
            float theta = atan2(normY,normX);
            float newX = newDist*cos(theta);
            float newY = newDist*sin(theta);
            int srcX = (int)(((newX+1)*width)/2.0);
            int srcY = (int)(((newY+1)*height)/2.0);
            int srcpixIdx = srcY * width + srcX + id_z * channel ;
            if (srcpixIdx >= 0 & srcpixIdx < width*height*channel) {
                output[dstpixIdx] = input[srcpixIdx];
            }
        }
    }
}
__kernel void fisheye_packed(
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

    int dstpixIdx = id_x * channel + id_y * width * channel + id_z;
    float normY = ((float)(2*id_y)/(float) (height))-1; 
    float normYSquare = normY*normY;
    float normX = ((float)(2*id_x)/(float) (width))-1;
    float normXSquare = normX*normX;
    float dist = sqrt(normXSquare+normYSquare);
    if ((0.0 <= dist) && (dist <= 1.0)) {
        float newDist = sqrt(1.0-dist*dist);
        newDist = (dist + (1.0-newDist)) / 2.0;
        if (newDist <= 1.0) {
            float theta = atan2(normY,normX);
            float newX = newDist*cos(theta);
            float newY = newDist*sin(theta);
            int srcX = (int)(((newX+1)*width)/2.0);
            int srcY = (int)(((newY+1)*height)/2.0);
            int srcpixIdx = srcY * width  * channel + srcX  * channel + id_z ;
            if (srcpixIdx >= 0 & srcpixIdx < width*height*channel) {
                output[dstpixIdx] = input[srcpixIdx];
            }
        }
    }
}
