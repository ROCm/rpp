#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

__kernel void brightness_contrast(  __global unsigned char* a,
                                    __global unsigned char* b,
                                    const float alpha,
                                    const int beta,
                                    const size_t height,
                                    const size_t width,
                                    const size_t channel
)
{
    int pixIdx = get_global_id(0) + get_global_id(1)* width +
                 get_global_id(2)* width * height;

    int res = a[pixIdx] * alpha + beta;
    b[pixIdx] = saturate_8u(res);
}
