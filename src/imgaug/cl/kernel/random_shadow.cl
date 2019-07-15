#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
__kernel void random_shadow_pkd(
    const __global unsigned char* input,
    __global  unsigned char* output,
        const unsigned int height,
        const unsigned int width,
        const unsigned int channel
){

}