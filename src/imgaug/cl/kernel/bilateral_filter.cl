#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
double gaussian(double x, double sigmaI) {
    double a = 2.0;
    return exp(-(pow(x, a))/(2 * pow(sigmaI, a))) / (2 * M_PI * pow(sigmaI, a));
}

double distance(int x1, int y1, int x2, int y2){
    double d_x = x2-x1;
    double d_y = y2-y1;
    double a = 2.0;
    double dis = sqrt(pow(d_x,a)+pow(d_y,a));
    return dis;
}

__kernel void bilateral_filter_planar(
    const __global unsigned char* input,
    __global  unsigned char* output,
    const unsigned int height,
    const unsigned int width,
    const unsigned int channel,
    const unsigned int filterSize,
    const double sigmaI,
    const double sigmaS
)
{

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x + id_y * width + id_z * width * height;

    int hfFiltSz = filterSize/2;
    if ( (id_x < hfFiltSz) || (id_y < hfFiltSz) ||
            (id_y >= (height-hfFiltSz)) || (id_x >= (width-hfFiltSz)) )
    {
        output[pixIdx] = input[pixIdx];
        //output[pixIdx] = 0;
        return;
    }
    
    double sum = 0.0;
    double w_sum = 0.0;
    for (int ri = (-1 * hfFiltSz) , rf = 0;
            (ri <= hfFiltSz) && (rf < filterSize);
                ri++, rf++)
    {
        for (int ci = (-1 * hfFiltSz) , cf = 0;
                (ci <= hfFiltSz) && (cf < filterSize);
                    ci++, cf++)
        {
            const int idxI = pixIdx + ri + ci * width;
            double gi  = gaussian(input[idxI]-input[pixIdx],sigmaI);
            double dis = distance(id_y, id_x, id_y+ri, id_x+ci);
            double gs  = gaussian(dis,sigmaS);
            double w = gi * gs;
            sum += input[idxI] * w;
            w_sum += w;
        }
    }
    int res = sum/w_sum;
    output[pixIdx] = saturate_8u(res);

}

__kernel void bilateral_filter_packed(
    const __global unsigned char* input,
    __global  unsigned char* output,
    const unsigned int height,
    const unsigned int width,
    const unsigned int channel,
    const unsigned int filterSize,
    const double sigmaI,
    const double sigmaS
)
{

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x * channel + id_y * width * channel + id_z ;

    int hfFiltSz = filterSize/2;
    if ( (id_x < hfFiltSz) || (id_y < hfFiltSz) ||
            (id_y >= (height-hfFiltSz)) || (id_x >= (width-hfFiltSz)) )
    {
        output[pixIdx] = input[pixIdx];
        return;
    }

    double sum = 0.0;
    double w_sum = 0.0;
    for (int ri = (-1 * hfFiltSz) , rf = 0;
            (ri <= hfFiltSz) && (rf < filterSize);
                ri++, rf++)
    {
        for (int ci = (-1 * hfFiltSz) , cf = 0;
                (ci <= hfFiltSz) && (cf < filterSize);
                    ci++, cf++)
        {
            const int idxI = pixIdx + ri * channel + ci * width * channel;
            
            double gi  = gaussian(input[idxI]-input[pixIdx],sigmaI);
            double dis = distance(id_y, id_x, id_y + (ri * channel) , id_x + (ci * channel));
            double gs  = gaussian(dis,sigmaS);
            double w = gi * gs;
            sum += input[idxI] * w;
            w_sum += w;
        }
    }
    int res = sum/w_sum;
    output[pixIdx] = saturate_8u(res);

}