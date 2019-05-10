__kernel void brightness_contrast(  __global unsigned char a,
                                    __global unsigned char b,
                                    const size_t n,
                                    const Rpp32f alpha,
                                    const Rpp32s beta)
{
    //Get our global thread ID
    int id = get_global_id(0);

    //Make sure we do not go out of bounds
    if (id < n)
        b[id] = a[id] * alpha + beta;
}
