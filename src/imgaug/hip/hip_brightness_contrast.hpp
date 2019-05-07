
template <typename T>
__global__ void brightness_contrast_kernel( T* inDevPtr, T* outDevPtr,
                                            const int rows,const int cols, const int chns,
                                            const Rpp32f alpha, const Rpp32f beta )
{

    int x = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x; //width
    int y = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y; //height
    int z = hipThreadIdx_z + hipBlockIdx_z * hipBlockDim_z; //depth
    if ( x >= cols || y >= rows || z >= chns)
        return;

    //Packed
    T result = inDevPtr[z + x*chns + y*rows*chns] * alpha + beta;
    outDevPtr[z + x*chns + y*rows*chns] = result;

    //Planar impl is same for 3 channel alone

}


template <typename T>
RppStatus hip_brightness_contrast ( T* inputPtr, RppiSize srcSize, T* outputPtr,
                                    Rpp32f alpha, Rpp32f beta, RppiChnFormat chnFormat )
{

    if (1)
    { // naive Kernel

        dim3 grid;
        dim3 block;
        if (srcSize.channel == 1) { block = dim3(32,32,1); grid = dim3(srcSize.width/32 +1, srcSize.height/32 +1 ,1);}
        else if(srcSize.channel == 3) { block = dim3(16,16,3); grid = dim3(srcSize.width/16 +1, srcSize.height/16 +1 ,1);}
        else if(srcSize.channel == 4) { block = dim3(16,16,4); grid = dim3(srcSize.width/16 +1, srcSize.height/16+1 ,1);}

        hipLaunchKernelGGL( (brightness_contrast_kernel<T>),
                            grid, block, 0, /*Stream*/0,
                            inputPtr, outputPtr,
                            srcSize.height, srcSize.width, srcSize.channel,
                            alpha, beta );
    }
    else if (0)
    {
        // Blas implementation
    }


    return RPP_SUCCESS;

}
