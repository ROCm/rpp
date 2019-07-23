#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
float gaussian(float x,float y, float stdDev) 
{
    float res,pi=3.14;
    res= 1 / (2 * pi * stdDev * stdDev);
    float exp1,exp2;
    exp1= - (x*x) / (2*stdDev*stdDev);
    exp2= - (y*y) / (2*stdDev*stdDev);
    exp1= exp1+exp2;
    exp1=exp(exp1);
    res*=exp1;
	return res;
}
__kernel void vignette(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const float stdDev
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_y*channel*width + id_x*channel + id_z;

    int x=(id_x-(width/2));
    int y=(id_y-(height/2));
    float gaussianvalue=gaussian(x,y,stdDev)/gaussian(0.0,0.0,stdDev);
    float res = ((float)input[pixIdx]) * gaussianvalue ;
    output[pixIdx] = saturate_8u(res) ;
}