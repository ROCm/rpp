hipError_t
hipoc_brightness_contrast( void* srcPtr, RppiSize srcSize,
                                void* dstPtr,
                                Rpp32f alpha, Rpp32s beta,
                                RppiChnFormat chnFormat, unsigned int channel,
                                hipStream_t theQueue );