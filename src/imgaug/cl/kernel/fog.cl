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
        float pixel = ((float) input[pixId])  * (1.5 + fogValue) - (fogValue*4) + (7*fogValue);
        input[pixId] = saturate_8u(pixel);
        pixel = ((float) input[pixId+c]) * (1.5 + fogValue) + (7*fogValue);
        input[pixId+c] = saturate_8u(pixel);
        pixel = ((float) input[pixId+c*2]) * (1.5 + fogValue) + (fogValue*4) + (7*fogValue);
        input[pixId+c*2] = saturate_8u(pixel);
    }

    else if(check<=(85*3))
    {
        float pixel = ((float) input[pixId]) * (1.5 + (fogValue*fogValue)) - (fogValue*4) + (130*fogValue);
        input[pixId] = saturate_8u(pixel);
        pixel = ((float) input[pixId+c]) * (1.5 + (fogValue*fogValue)) + (130*fogValue);
        input[pixId+c] = saturate_8u(pixel);
        pixel = ((float) input[pixId+c*2]) * (1.5 + (fogValue*fogValue)) + (fogValue*4) + 130*fogValue;
        input[pixId+c*2] = saturate_8u(pixel);
    }
    else
    {
        float pixel = ((float) input[pixId]) * (1.5 + (fogValue * ( fogValue * 1.414))) - (fogValue*4) + 20 + (100*fogValue);
        input[pixId] = saturate_8u(pixel);
        pixel = ((float) input[pixId+c]) * (1.5 + (fogValue * ( fogValue * 1.414))) + 20 + (100*fogValue);
        input[pixId+c] = saturate_8u(pixel);
        pixel = ((float) input[pixId+c*2]) * (1.5 + (fogValue * ( fogValue * 1.414))) + (fogValue*4) + (100*fogValue);
        input[pixId+c*2] = saturate_8u(pixel);
    }
}

__kernel void fog_pkd(  __global unsigned char* input,
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
        float pixel = ((float) input[i]) * (1.5 + fogValue) - (fogValue*4) + (7*fogValue);
        input[i] = saturate_8u(pixel);
        pixel = ((float) input[i + 1]) * (1.5 + fogValue) + (7*fogValue);
        input[i+1] = saturate_8u(pixel);
        pixel = ((float) input[i + 2]) * (1.5 + fogValue) + (fogValue*4) + (7*fogValue);
        input[i+2] = saturate_8u(pixel);
    }
    else if(check<=(85*3) && fogValue!=0)
    {
        float pixel = ((float) input[i]) * (1.5 + (fogValue*fogValue)) - (fogValue*4) + (130*fogValue);
        input[i] = saturate_8u(pixel);
        pixel = ((float) input[i + 1]) * (1.5 + (fogValue*fogValue)) + (130*fogValue);
        input[i+1] = saturate_8u(pixel);
        pixel = ((float) input[i + 2]) * (1.5 + (fogValue*fogValue)) + (fogValue*4) + 130*fogValue;
        input[i+2] = saturate_8u(pixel);
    }
    else if(fogValue!=0)
    {
        float pixel = ((float) input[i]) * (1.5 + (fogValue * ( fogValue * 1.414))) - (fogValue*4) + 20 + (100*fogValue);
        input[i] = saturate_8u(pixel);
        pixel = ((float) input[i + 1]) * (1.5 + (fogValue * ( fogValue * 1.414))) + 20 + (100*fogValue);
        input[i+1] = saturate_8u(pixel);
        pixel = ((float) input[i + 2]) * (1.5 + (fogValue * ( fogValue * 1.414))) + (fogValue*4) + (100*fogValue);
        input[i+2] = saturate_8u(pixel);
    }

}