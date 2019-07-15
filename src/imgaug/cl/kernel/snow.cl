#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
inline float HueToRGB(float v1, float v2, float vH) {
    if (vH < 0)
        vH += 1;

    if (vH > 1)
        vH -= 1;

    if ((6 * vH) < 1)
        return (v1 + (v2 - v1) * 6 * vH);

    if ((2 * vH) < 1)
        return v2;

    if ((3 * vH) < 2)
        return (v1 + (v2 - v1) * ((2.0f / 3) - vH) * 6);

    return v1;
}
__kernel void snow_pkd(
        const __global unsigned char* input,
        __global  unsigned char* output,
        const unsigned int height,
        const unsigned int width,
        const unsigned int channel,
        const float snowCoefficient
){
    // int id_x = get_global_id(0);
    // int id_y = get_global_id(1);
    // int id_z = get_global_id(2);
    // if (id_x >= width || id_y >= height || id_z >= channel) return;

    // int pixIdx = id_x * channel + id_y * width * channel + id_z;
    double snowVal = snowCoefficient;
    snowVal = (snowVal * 255 ) / 2;
    snowVal =  (snowVal + 255) / 3;
    // float pixel;
    // if(input[pixIdx] < 100)
    //     pixel = input[pixIdx] * 2.5;
    // output[pixIdx] = saturate_8u(pixel);
        //Get our global thread ID
    int id = get_global_id(0);
    double r,g,b, min, max, delta;
    double h, s, l;

    //Make sure we do not go out of bounds
    id = id * 3;
    if (id < 3 *height * width ){
        r = input[id] / 255.0;
        g = input[id + 1] / 255.0;
        b = input[id + 2]/ 255.0;

        min = (r < g && r< b)? r : ((g < b)? g: b);
        max = (r > g && r > b)? r : ((g > b)? g: b);

        delta = max - min;

        l = (min + max) / 2;
        if (delta == 0){
            h = 0;
            s = 0;
        }
        else {
            s = (l <= 0.5) ? (delta / (max + min)) : (delta / (2 - max - min));
            double hue;

            if (r == max)
            {
                hue = ((g - b) / 6) / delta;
            }
            else if (g == max)
            {
                hue = (1.0f / 3) + ((b - r) / 6) / delta;
            }
            else
            {
                hue = (2.0f / 3) + ((r - g) / 6) / delta;
            }

            if (hue < 0)
                hue += 1;
            if (hue > 1)
                hue -= 1;

            h = (int)(hue * 360);
        }
        if ( l < snowVal)
            l = l * 2.5;
        if( l > 1)
            l = 1;

        if (s <= 0){
            r = l * 255;
            g = l * 255;
            b = l * 255;
        } 
        else {
            
            float v1, v2;
            float hue = (float)h / 360;

            v2 = (l < 0.5) ? (l * (1 + s)) : ((l + s) - (l * s));
            v1 = 2 * l - v2;

            r = (unsigned char)(255 * HueToRGB(v1, v2, hue + (1.0f / 3)));
            g = (unsigned char)(255 * HueToRGB(v1, v2, hue));
            b = (unsigned char)(255 * HueToRGB(v1, v2, hue - (1.0f / 3)));
        }
        output[id] = saturate_8u(r);
        output[id + 1] = saturate_8u(g);
        output[id + 2] = saturate_8u(b);
    }

}