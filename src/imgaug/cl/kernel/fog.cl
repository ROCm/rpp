#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
__kernel void fog_planar(  __global unsigned char* input,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const float fogValue
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    if (id_x >= width || id_y >= height) return;
    int pixId= width * id_y  + id_x;
    int c=width*height;
    float check=input[pixId]+input[pixId+c]+input[pixId+c*2];
    if(check >= (240*3) && fogValue!=0)
    {}
    else if(check>=(170*3))
    {
        float pixel = ((float) srcPtr[pixId])  * (1.5 + fogValue) - (fogValue*4) + (7*fogValue);
        srcPtr[i] = RPPPIXELCHECK(pixel);
        pixel = ((float) srcPtr[pixId+c]) * (1.5 + fogValue) + (7*fogValue);
        srcPtr[pixId+c] = RPPPIXELCHECK(pixel);
        pixel = ((float) srcPtr[pixId+c*2]) * (1.5 + fogValue) + (fogValue*4) + (7*fogValue);
        srcPtr[pixId+c*2] = RPPPIXELCHECK(pixel);
    }

    else if(check<=(85*3))
    {
        float pixel = ((float) srcPtr[pixId]) * (1.5 + pow(fogValue,2)) - (fogValue*4) + (130*fogValue);
        srcPtr[pixId] = RPPPIXELCHECK(pixel);
        pixel = ((float) srcPtr[pixId+c]) * (1.5 + pow(fogValue,2)) + (130*fogValue);
        srcPtr[pixId+c] = RPPPIXELCHECK(pixel);
        pixel = ((float) srcPtr[pixId+c*2]) * (1.5 + pow(fogValue,2)) + (fogValue*4) + 130*fogValue;
        srcPtr[pixId+c*2] = RPPPIXELCHECK(pixel);
    }
    else
    {
        float pixel = ((float) srcPtr[pixId]) * (1.5 + pow(fogValue,1.5)) - (fogValue*4) + 20 + (100*fogValue);
        srcPtr[pixId] = RPPPIXELCHECK(pixel);
        pixel = ((float) srcPtr[pixId+c]) * (1.5 + pow(fogValue,1.5)) + 20 + (100*fogValue);
        srcPtr[pixId+c] = RPPPIXELCHECK(pixel);
        pixel = ((float) srcPtr[pixId+c*2]) * (1.5 + pow(fogValue,1.5)) + (fogValue*4) + (100*fogValue);
        srcPtr[pixId+c*2] = RPPPIXELCHECK(pixel);
    }
}
__kernel void fog_packed(  __global unsigned char* input,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const float fogValue
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    if (id_x >= width || id_y >= height) return;
    int i= width * id_y * channel + id_x * channel;
    float check=input[i]+input[i+1]+input[i+2];
    if(check >= (240*3) && fogValue!=0)
    {}
    else if(check>=(170*3) && fogValue!=0)
    {
        float pixel = ((float) srcPtr[i]) * (1.5 + fogValue) - (fogValue*4) + (7*fogValue);
        srcPtr[i] = RPPPIXELCHECK(pixel);
        pixel = ((float) srcPtr[i + 1]) * (1.5 + fogValue) + (7*fogValue);
        srcPtr[i+1] = RPPPIXELCHECK(pixel);
        pixel = ((float) srcPtr[i + 2]) * (1.5 + fogValue) + (fogValue*4) + (7*fogValue);
        srcPtr[i+2] = RPPPIXELCHECK(pixel);
    }
    else if(check<=(85*3) && fogValue!=0)
    {
        float pixel = ((float) srcPtr[i]) * (1.5 + pow(fogValue,2)) - (fogValue*4) + (130*fogValue);
        srcPtr[i] = RPPPIXELCHECK(pixel);
        pixel = ((float) srcPtr[i + 1]) * (1.5 + pow(fogValue,2)) + (130*fogValue);
        srcPtr[i+1] = RPPPIXELCHECK(pixel);
        pixel = ((float) srcPtr[i + 2]) * (1.5 + pow(fogValue,2)) + (fogValue*4) + 130*fogValue;
        srcPtr[i+2] = RPPPIXELCHECK(pixel);
    }
    else if(fogValue!=0)
    {
        float pixel = ((float) srcPtr[i]) * (1.5 + pow(fogValue,1.5)) - (fogValue*4) + 20 + (100*fogValue);
        srcPtr[i] = RPPPIXELCHECK(pixel);
        pixel = ((float) srcPtr[i + 1]) * (1.5 + pow(fogValue,1.5)) + 20 + (100*fogValue);
        srcPtr[i+1] = RPPPIXELCHECK(pixel);
        pixel = ((float) srcPtr[i + 2]) * (1.5 + pow(fogValue,1.5)) + (fogValue*4) + (100*fogValue);
        srcPtr[i+2] = RPPPIXELCHECK(pixel);
    }

}