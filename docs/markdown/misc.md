# Miscellaneous examples

## RPP stand-alone batch processing code snippet (HOST)

    // Initializations
    int noOfImages = 32;
    int channels = 3;
    RppiSize maxSize;
    maxSize.width = 224;
    maxSize.height = 224;

    // Allocate host memory and/or obtain input data
    unsigned long long ioBufferSize = noOfImages * maxSize.width * maxSize.height * channels;
    Rpp8u *input = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    Rpp8u *output = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    // Get the data for a batch of 224x224 images into 'input' here

    // Initialize values for any necessary parameters to the RPP function being called
    Rpp32f alpha[noOfImages];
    Rpp32f beta[noOfImages];
    for (int i = 0; i < noOfImages; i++)
    {
        alpha[i] = 1.75;
        beta[i] = 50;
        srcSize[i].width = 224;
        srcSize[i].height = 224;
    }

    // Create handle
    rppHandle_t handle;
    
    // Set the number of threads to be used by OpenMP pragma for RPP batch processing on host.
    // If numThreads value passed is 0, number of OpenMP threads used by RPP will be set to batch size
    Rpp32u numThreads = 0;
    rppCreateWithBatchSize(&handle, noOfImages, numThreads);

    // Call the RPP API for the specific variant required (pkd3/pln3/pln1)
    rppi_brightness_u8_pkd3_batchPD_host(input, srcSize, maxSize, output, alpha, beta, noOfImages, handle);

### RPP stand-alone batch processing code snippet (OCL)

    // Initializations
    int noOfImages = 32;
    int channels = 3;
    RppiSize maxSize;
    maxSize.width = 224;
    maxSize.height = 224;

    // Allocate host memory and/or obtain input data
    unsigned long long ioBufferSize = noOfImages * maxSize.width * maxSize.height * channels;
    Rpp8u *input = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    Rpp8u *output = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    // Get the data for a batch of 224x224 images into 'input' here

    // OCL initializations, allocate device memory and copy input data to device
    cl_mem d_input, d_output;
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context theContext;
    cl_command_queue theQueue;
    cl_int err;
    err = clGetPlatformIDs(1, &platform_id, NULL);
    err |= clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    theContext = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    theQueue = clCreateCommandQueueWithProperties(theContext, device_id, 0, &err);
    d_input = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp8u), NULL, NULL);
    d_output = clCreateBuffer(theContext, CL_MEM_READ_ONLY, ioBufferSize * sizeof(Rpp8u), NULL, NULL);
    err |= clEnqueueWriteBuffer(theQueue, d_input, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8u), input, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(theQueue, d_output, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8u), output, 0, NULL, NULL);

    // Initialize values for any necessary parameters to the RPP function being called
    Rpp32f alpha[noOfImages];
    Rpp32f beta[noOfImages];
    for (int i = 0; i < noOfImages; i++)
    {
        alpha[i] = 1.75;
        beta[i] = 50;
        srcSize[i].width = 224;
        srcSize[i].height = 224;
    }

    // Create handle
    rppHandle_t handle;
    rppCreateWithStreamAndBatchSize(&handle, theQueue, noOfImages);

    // Call the RPP API for the specific variant required (pkd3/pln3/pln1)
    rppi_brightness_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, alpha, beta, noOfImages, handle);

    // Copy output data back to host
    clEnqueueReadBuffer(theQueue, d_output, CL_TRUE, 0, ioBufferSize * sizeof(Rpp8u), output, 0, NULL, NULL);

## RPP stand-alone batch processing code snippet (HIP)

    // Initializations
    int noOfImages = 32;
    int channels = 3;
    RppiSize maxSize;
    maxSize.width = 224;
    maxSize.height = 224;

    // Allocate host memory and/or obtain input data
    unsigned long long ioBufferSize = noOfImages * maxSize.width * maxSize.height * channels;
    Rpp8u *input = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    Rpp8u *output = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    // Get the data for a batch of 224x224 images into 'input' here

    // HIP initializations, allocate device memory and copy input data to device
    int *d_input, d_output;
    hipMalloc(&d_input, ioBufferSize * sizeof(Rpp8u));
    hipMalloc(&d_output, ioBufferSize * sizeof(Rpp8u));
    hipMemcpy(d_input, input, ioBufferSize * sizeof(Rpp8u), hipMemcpyHostToDevice);
    hipMemcpy(d_output, output, ioBufferSize * sizeof(Rpp8u), hipMemcpyHostToDevice);

    // Initialize values for any necessary parameters to the RPP function being called
    Rpp32f alpha[noOfImages];
    Rpp32f beta[noOfImages];
    for (int i = 0; i < noOfImages; i++)
    {
        alpha[i] = 1.75;
        beta[i] = 50;
        srcSize[i].width = 224;
        srcSize[i].height = 224;
    }

    // Create handle
    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    // Call the RPP API for the specific variant required (pkd3/pln3/pln1)
    rppi_brightness_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, alpha, beta, noOfImages, handle);

    // Copy output data back to host
    hipMemcpy(output, d_output, ioBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToHost);
