#ifndef RPP_CL_ARITHMETIC_DECLATAIONS_H
#define RPP_CL_ARITHMETIC_DECLATAIONS_H

//===== Utils
cl_int
cl_kernel_initializer ( cl_command_queue theQueue,
                        std::string kernelFile, std::string kernelName,
                        cl_program& theProgram, cl_kernel& theKernel);

cl_int
cl_kernel_implementer (cl_command_queue theHandle, size_t* gDim3, size_t* lDim3, cl_program& theProgram,
                        cl_kernel& theKernel  );

//===== Internal CL functions
RppStatus
cl_bitwise_AND ( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize, cl_mem dstPtr,
                 RppiChnFormat chnFormat, unsigned int channel,
                 cl_command_queue theQueue);

RppStatus
cl_bitwise_NOT ( cl_mem srcPtr1,
                 RppiSize srcSize, cl_mem dstPtr,
                 RppiChnFormat chnFormat, unsigned int channel,
                 cl_command_queue theQueue);

RppStatus
cl_exclusive_OR ( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize, cl_mem dstPtr,
                 RppiChnFormat chnFormat, unsigned int channel,
                 cl_command_queue theQueue);

RppStatus
cl_inclusive_OR ( cl_mem srcPtr1,cl_mem srcPtr2,
                 RppiSize srcSize, cl_mem dstPtr,
                 RppiChnFormat chnFormat, unsigned int channel,
                 cl_command_queue theQueue);

#endif //RPP_CL_IMGAUG_DECLATAIONS_H
