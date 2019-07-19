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
        return (v1 + (v2 - v1) * ((float)(2.0 / 3) - vH) * 6);

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
    int id = get_global_id(0);
    float r,g,b, min, max, delta;
    float h, s, l;

    //Make sure we do not go out of bounds
    id = id * 3;
    if (id < 3 *height * width ){
        r = (float)input[id] / 255.0;
        g = (float)input[id + 1] / 255.0;
        b = (float)input[id + 2]/ 255.0;

        min = (r < g && r< b)? r : ((g < b)? g: b);
        max = (r > g && r > b)? r : ((g > b)? g: b);

        delta = max - min;

        l = (float)(min + max) / 2.0;
        if (delta == 0){
            h = 0;
            s = 0;
        }
        else {
            s = (l <= 0.5) ? (delta / (max + min)) : (delta / (2.0 - (max - min)));
            float hue;

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
        if ( l < snowCoefficient)
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
__kernel void snow_pln(
        const __global unsigned char* input,
        __global  unsigned char* output,
        const unsigned int height,
        const unsigned int width,
        const unsigned int channel,
        const float snowCoefficient
){
    int id = get_global_id(0);
    float r,g,b, min, max, delta;
    float h, s, l;

    //Make sure we do not go out of bounds
    id = id * 3;
    if (id < 3 *height * width ){
        r = (float)input[id] / 255.0;
        g = (float)input[id + height * width] / 255.0;
        b = (float)input[id + 2 * height * width]/ 255.0;

        min = (r < g && r< b)? r : ((g < b)? g: b);
        max = (r > g && r > b)? r : ((g > b)? g: b);

        delta = max - min;

        l = (float)(min + max) / 2.0;
        if (delta == 0){
            h = 0;
            s = 0;
        }
        else {
            s = (l <= 0.5) ? (delta / (max + min)) : (delta / (2.0 - (max - min)));
            float hue;

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
        if ( l < snowCoefficient)
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
        output[id + height * width] = saturate_8u(g);
        output[id + 2 * height * width] = saturate_8u(b);
    }

}