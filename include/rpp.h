/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef RPP_H
#define RPP_H

#include <export.h>

#if RPP_BACKEND_OPENCL

#define CL_TARGET_OPENCL_VERSION 220
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
typedef cl_command_queue rppAcceleratorQueue_t;

#elif RPP_BACKEND_HIP

#include <hip/hip_runtime_api.h>
typedef hipStream_t rppAcceleratorQueue_t;

#endif

// Constructs type name from a struct
#define RPP_DECLARE_OBJECT(name)    \
    struct name                     \
    {                               \
    };                              \
    typedef struct name* name##_t;

RPP_DECLARE_OBJECT(rppHandle);      // Create the rppHandle_t type
typedef rppHandle_t RppHandle_t;    // Create typedef for RppHandle_t

#if _WIN32
#define SHARED_PUBLIC __declspec(dllexport)
#else
#define SHARED_PUBLIC __attribute__ ((visibility ("default")))
#endif

#include "rppcore.h"
#include "rppdefs.h"
#include "rppi.h"
#include "rppt.h"
#include "rppversion.h"

/******************** rppAllocatorFunction ********************/

// Custom allocator function to allow for user-defined custom allocation
// *param[in] context A pointer to a context
// *param[in] sizeBytes Number of bytes to allocate
typedef void* (*rppAllocatorFunction)(void* context, size_t sizeBytes);

/******************** rppDeallocatorFunction ********************/

// Custom deallocator function to allow for user-defined custom deallocation
// *param[in] context A pointer to a context
// *param[in] memory A pointer to allocated memory
typedef void (*rppDeallocatorFunction)(void* context, void* memory);

#ifdef __cplusplus
extern "C" {
#endif

/******************** rppGetErrorString ********************/

// Returns a NULL terminated character string of the passed error code
// *param[in] error Error status of rppStatus_t type
// *returns errorString
extern "C" SHARED_PUBLIC const char* rppGetErrorString(rppStatus_t error);

/******************** rppCreate ********************/

// Function to create a rpp handle. To be called in the beginning to initialize the rpp environment
// *param[in] handle A pointer to rpp handle of type rppHandle_t
// *returns a rppStatus_t enumeration.
extern "C" SHARED_PUBLIC rppStatus_t rppCreate(rppHandle_t* handle);

/******************** rppCreateWithBatchSize ********************/

// Function to create a rpp handle for a batch. To be called in the beginning to initialize the rpp environment
// *param[in] handle A pointer to rpp handle of type rppHandle_t
// *param[in] nBatchSize Batch size
// *param[in] numThreads number of threads to be used for OpenMP pragma
// *returns a rppStatus_t enumeration.
extern "C" SHARED_PUBLIC rppStatus_t rppCreateWithBatchSize(rppHandle_t* handle, size_t nBatchSize, Rpp32u numThreads = 0);

/******************** rppDestroy ********************/

// Function to destroy a rpp handle. To be called in the end to break down the rpp environment
// *param[in] handle An rpp handle of type rppHandle_t
// *returns a rppStatus_t enumeration.
extern "C" SHARED_PUBLIC rppStatus_t rppDestroy(rppHandle_t handle);

/******************** rppDestroyHost ********************/

// Function to destroy a rpp handle's host memory allocation. To be called in the end to break down the rpp environment
// *param[in] handle An rpp handle of type rppHandle_t
// *returns a rppStatus_t enumeration.
extern "C" SHARED_PUBLIC rppStatus_t rppDestroyHost(rppHandle_t handle);

/******************** rppSetBatchSize ********************/

// Function to set batch size for handle previously created
// *param[in] handle An rpp handle of type rppHandle_t
// *param[in] batchSize Batch size
// *returns a rppStatus_t enumeration.
extern "C" SHARED_PUBLIC rppStatus_t rppSetBatchSize(rppHandle_t handle, size_t batchSize);

/******************** rppGetBatchSize ********************/

// Function to get batch size for handle previously created
// *param[in] handle An rpp handle of type rppHandle_t
// *param[in] batchSize Batch size
// *returns a rppStatus_t enumeration.
extern "C" SHARED_PUBLIC rppStatus_t rppGetBatchSize(rppHandle_t handle, size_t *batchSize);

#if GPU_SUPPORT

/******************** rppCreateWithStream ********************/

// Function to create a rpp handle with an accelerator stream. To be called in the beginning to initialize the rpp environment
// *param[in] handle A pointer to rpp handle of type rppHandle_t
// *param[in] stream An accelerator queue of type rppAcceleratorQueue_t (hipStream_t for HIP and cl_command_queue for OpenCL)
// *returns a rppStatus_t enumeration.
extern "C" SHARED_PUBLIC rppStatus_t rppCreateWithStream(rppHandle_t* handle, rppAcceleratorQueue_t stream);

/******************** rppCreateWithStreamAndBatchSize ********************/

// Function to create a rpp handle with an accelerator stream for a batch. To be called in the beginning to initialize the rpp environment
// *param[in] handle A pointer to rpp handle of type rppHandle_t
// *param[in] stream An accelerator queue of type rppAcceleratorQueue_t (hipStream_t for HIP and cl_command_queue for OpenCL)
// *param[in] nBatchSize Batch size
// *returns a rppStatus_t enumeration.
extern "C" SHARED_PUBLIC rppStatus_t rppCreateWithStreamAndBatchSize(rppHandle_t* handle, rppAcceleratorQueue_t stream, size_t nBatchSize);

/******************** rppDestroyGPU ********************/

// Function to destroy a rpp handle's device memory allocation. To be called in the end to break down the rpp environment
// *param[in] handle An rpp handle of type rppHandle_t
// *returns a rppStatus_t enumeration.
extern "C" SHARED_PUBLIC rppStatus_t rppDestroyGPU(rppHandle_t handle);

/******************** rppSetStream ********************/

// Function to set an accelerator command queue previously created
// *param[in] handle An rpp handle of type rppHandle_t
// *param[in] stream An accelerator queue of type rppAcceleratorQueue_t (hipStream_t for HIP and cl_command_queue for OpenCL)
// *returns a rppStatus_t enumeration.
extern "C" SHARED_PUBLIC rppStatus_t rppSetStream(rppHandle_t handle, rppAcceleratorQueue_t streamID);

/******************** rppGetStream ********************/

// Function to get an accelerator command queue previously created
// *param[in] handle An rpp handle of type rppHandle_t
// *param[in] stream An accelerator queue of type rppAcceleratorQueue_t (hipStream_t for HIP and cl_command_queue for OpenCL)
// *returns a rppStatus_t enumeration.
extern "C" SHARED_PUBLIC rppStatus_t rppGetStream(rppHandle_t handle, rppAcceleratorQueue_t* streamID);

/******************** rppSetAllocator ********************/

// Function to set allocator for previously created rppHandle_t
// *param[in] handle An rpp handle of type rppHandle_t
// *param[in] allocator A callback function rpp will use for internal memory allocations. The provided callback function should allocate device memory with requested size and return a pointer to this memory. Passing 0 will restore the default RPP allocator and deallocator.
// *param[in] deallocator A callback function rpp will use to for internal memory deallocation. The provided callback function should free the specified memory pointer.
// *param[in] allocatorContext User-specified pointer which is passed to allocator and deallocator. This allows the callback function to access state set by the caller to this function, for example a stateful heap allocator or a c++ class.
// *returns a rppStatus_t enumeration.
extern "C" SHARED_PUBLIC rppStatus_t rppSetAllocator(rppHandle_t handle, rppAllocatorFunction allocator, rppDeallocatorFunction deallocator, void* allocatorContext);

/******************** rppGetKernelTime ********************/

// Function to get time for last kernel launched. This function is used only when profiling mode has been enabled.
// *param[in] handle An rpp handle of type rppHandle_t
// *param[in] time Pointer to a float type to contain kernel time in milliseconds
// *returns a rppStatus_t enumeration.
extern "C" SHARED_PUBLIC rppStatus_t rppGetKernelTime(rppHandle_t handle, float* time);

/******************** rppEnableProfiling ********************/

// Function to enable profiling to retrieve kernel time
// *param[in] handle An rpp handle of type rppHandle_t
// *param[in] enable Boolean to toggle profiling
// *returns a rppStatus_t enumeration.
extern "C" SHARED_PUBLIC rppStatus_t rppEnableProfiling(rppHandle_t handle, bool enable);

#endif // GPU_SUPPORT

#ifdef __cplusplus
}
#endif    // __cplusplus

#endif    // RPP_H
