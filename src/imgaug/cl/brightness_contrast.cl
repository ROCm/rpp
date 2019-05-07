__kernel void brightness_contrast(  __global RppPtr_t a,
                                    __global RppPtr_t b,
                                    const unsigned int n,
                                    const Rpp32f alpha,
                                    const Rpp32f beta)
{
    //Get our global thread ID
    int id = get_global_id(0);

    //Make sure we do not go out of bounds
    if (id < n)
        b[id] = a[id] * alpha + beta;
}
