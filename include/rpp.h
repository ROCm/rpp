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

/*! \file
 * \brief RPP top-level header with RPP handle API.
 * \defgroup group_rpp RPP handle API
 * \brief RPP API to create and destroy RPP HOST/GPU handle.
 */

#include <export.h>

#if RPP_BACKEND_OPENCL

#define CL_TARGET_OPENCL_VERSION 220
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
/*! \brief Set rppAcceleratorQueue_t to cl_command_queue if RPP_BACKEND_OPENCL \ingroup group_rpp */
typedef cl_command_queue rppAcceleratorQueue_t;

#elif RPP_BACKEND_HIP

#include <hip/hip_runtime_api.h>
/*! \brief Set rppAcceleratorQueue_t to hipStream_t if RPP_BACKEND_HIP \ingroup group_rpp */
typedef hipStream_t rppAcceleratorQueue_t;

#endif

/*! \brief Construct type name from a struct
 * \ingroup group_rpp
 */
#define RPP_DECLARE_OBJECT(name)    \
    struct name                     \
    {                               \
    };                              \
    typedef struct name* name##_t;

/*! \brief RPP handle type creation.
 * \ingroup group_rpp
 */
RPP_DECLARE_OBJECT(rppHandle);      // Create the rppHandle_t type

/*! \brief RPP handle.
 * \ingroup group_rpp
 */
typedef rppHandle_t RppHandle_t;    // Create typedef for RppHandle_t

#if _WIN32
#define SHARED_PUBLIC __declspec(dllexport)
#else
#define SHARED_PUBLIC __attribute__ ((visibility ("default")))
#endif

#include "rppdefs.h"
#include "rppi.h"
#include "rppt.h"
#include "rppversion.h"

/*! \brief Handles RPP context allocations.
 * \details Custom allocator function to allow for user-defined custom allocation.
 * \param [in] context A pointer to a context.
 * \param [in] sizeBytes Number of bytes to allocate.
 * \ingroup group_rpp
 */
typedef void* (*rppAllocatorFunction)(void* context, size_t sizeBytes);

/*! \brief Handles RPP context allocations.
 * \details Custom deallocator function to allow for user-defined custom deallocation.
 * \param [in] context A pointer to a context.
 * \param [in] memory A pointer to allocated memory.
 * \ingroup group_rpp
 */
typedef void (*rppDeallocatorFunction)(void* context, void* memory);

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Returns error string.
 * \details Returns a NULL terminated character string of the passed error code.
 * \param [in] error Error status of rppStatus_t type.
 * \ingroup group_rpp
 * \return <tt>\ref const char*</tt>
 * \returns Returns an error string among those listed under \ref rppStatus_t.
 */
extern "C" SHARED_PUBLIC const char* rppGetErrorString(rppStatus_t error);

/*! \brief Creates RPP handle for HOST single image processing - To be deprecated.
 * \details Function to create a RPP handle. To be called in the beginning to initialize the RPP environment.
 * \param [in] handle A pointer to RPP handle of type <tt> \ref rppHandle_t</tt>.
 * \ingroup group_rpp
 * \return A <tt> \ref rppStatus_t</tt> enumeration.
 * \retval rppStatusSuccess
 * \retval rppStatusNotInitialized
 * \retval rppStatusInvalidValue
 * \retval rppStatusBadParm
 * \retval rppStatusAllocFailed
 * \retval rppStatusInternalError
 * \retval rppStatusNotImplemented
 * \retval rppStatusUnknownError
 * \retval rppStatusUnsupportedOp
 * \deprecated
 */
extern "C" SHARED_PUBLIC rppStatus_t rppCreate(rppHandle_t* handle);

/*! \brief Creates RPP handle for HOST batch processing.
 * \details Function to create a RPP handle for a batch. To be called in the beginning to initialize the RPP environment.
 * \param [in] handle A pointer to RPP handle of type <tt> \ref rppHandle_t</tt>.
 * \param [in] nBatchSize Batch size.
 * \param [in] numThreads Number of threads to be used for OpenMP pragma.
 * \ingroup group_rpp
 * \return A <tt> \ref rppStatus_t</tt> enumeration.
 * \retval rppStatusSuccess
 * \retval rppStatusNotInitialized
 * \retval rppStatusInvalidValue
 * \retval rppStatusBadParm
 * \retval rppStatusAllocFailed
 * \retval rppStatusInternalError
 * \retval rppStatusNotImplemented
 * \retval rppStatusUnknownError
 * \retval rppStatusUnsupportedOp
 */
extern "C" SHARED_PUBLIC rppStatus_t rppCreateWithBatchSize(rppHandle_t* handle, size_t nBatchSize, Rpp32u numThreads = 0);

/*! \brief Destory RPP handle.
 * \details Function to destroy a RPP handle. To be called in the end to break down the RPP environment.
 * \param [in] handle RPP handle of type <tt> \ref rppHandle_t</tt>.
 * \ingroup group_rpp
 * \return A <tt> \ref rppStatus_t</tt> enumeration.
 * \retval rppStatusSuccess
 * \retval rppStatusNotInitialized
 * \retval rppStatusInvalidValue
 * \retval rppStatusBadParm
 * \retval rppStatusAllocFailed
 * \retval rppStatusInternalError
 * \retval rppStatusNotImplemented
 * \retval rppStatusUnknownError
 * \retval rppStatusUnsupportedOp
 */
extern "C" SHARED_PUBLIC rppStatus_t rppDestroy(rppHandle_t handle);

/*! \brief Destory RPP HOST handle - To be deprecated.
 * \details Function to destroy a RPP handle's host memory allocation. To be called in the end to break down the rpp environment.
 * \param [in] handle RPP handle of type <tt> \ref rppHandle_t</tt>.
 * \ingroup group_rpp
 * \return A <tt> \ref rppStatus_t</tt> enumeration.
 * \retval rppStatusSuccess
 * \retval rppStatusNotInitialized
 * \retval rppStatusInvalidValue
 * \retval rppStatusBadParm
 * \retval rppStatusAllocFailed
 * \retval rppStatusInternalError
 * \retval rppStatusNotImplemented
 * \retval rppStatusUnknownError
 * \retval rppStatusUnsupportedOp
 * \deprecated
 */
extern "C" SHARED_PUBLIC rppStatus_t rppDestroyHost(rppHandle_t handle);

/*! \brief Set batch size given a RPP handle.
 * \details Function to set batch size for handle previously created.
 * \param [in] handle RPP handle of type <tt> \ref rppHandle_t</tt>.
 * \param [in] batchSize Batch size.
 * \ingroup group_rpp
 * \return A <tt> \ref rppStatus_t</tt> enumeration.
 * \retval rppStatusSuccess
 * \retval rppStatusNotInitialized
 * \retval rppStatusInvalidValue
 * \retval rppStatusBadParm
 * \retval rppStatusAllocFailed
 * \retval rppStatusInternalError
 * \retval rppStatusNotImplemented
 * \retval rppStatusUnknownError
 * \retval rppStatusUnsupportedOp
 */
extern "C" SHARED_PUBLIC rppStatus_t rppSetBatchSize(rppHandle_t handle, size_t batchSize);

/*! \brief Get batch size given a RPP Handle.
 * \details Function to get batch size for handle previously created.
 * \param [in] handle RPP handle of type <tt> \ref rppHandle_t</tt>.
 * \param [in] batchSize Batch size
 * \ingroup group_rpp
 * \return A <tt> \ref rppStatus_t</tt> enumeration.
 * \retval rppStatusSuccess
 * \retval rppStatusNotInitialized
 * \retval rppStatusInvalidValue
 * \retval rppStatusBadParm
 * \retval rppStatusAllocFailed
 * \retval rppStatusInternalError
 * \retval rppStatusNotImplemented
 * \retval rppStatusUnknownError
 * \retval rppStatusUnsupportedOp
 */
extern "C" SHARED_PUBLIC rppStatus_t rppGetBatchSize(rppHandle_t handle, size_t *batchSize);

#if GPU_SUPPORT

/*! \brief Creates RPP handle with accelerator stream for GPU single image processing - To be deprecated.
 * \details Function to create a RPP handle with an accelerator stream. To be called in the beginning to initialize the RPP environment.
 * \param [in] handle A pointer to RPP handle of type <tt> \ref rppHandle_t</tt>.
 * \param [in] stream An accelerator queue of type <tt> \ref rppAcceleratorQueue_t</tt> (hipStream_t for HIP and cl_command_queue for OpenCL).
 * \ingroup group_rpp
 * \return A <tt> \ref rppStatus_t</tt> enumeration.
 * \retval rppStatusSuccess
 * \retval rppStatusNotInitialized
 * \retval rppStatusInvalidValue
 * \retval rppStatusBadParm
 * \retval rppStatusAllocFailed
 * \retval rppStatusInternalError
 * \retval rppStatusNotImplemented
 * \retval rppStatusUnknownError
 * \retval rppStatusUnsupportedOp
 * \deprecated
 */
extern "C" SHARED_PUBLIC rppStatus_t rppCreateWithStream(rppHandle_t* handle, rppAcceleratorQueue_t stream);

/*! \brief Creates RPP handle with accelerator stream for GPU batch processing.
 * \details Function to create a RPP handle with an accelerator stream for a batch. To be called in the beginning to initialize the RPP environment.
 * \param [in] handle A pointer to RPP handle of type <tt> \ref rppHandle_t</tt>.
 * \param [in] stream An accelerator queue of type <tt> \ref rppAcceleratorQueue_t</tt> (hipStream_t for HIP and cl_command_queue for OpenCL).
 * \param [in] nBatchSize Batch size
 * \ingroup group_rpp
 * \return A <tt> \ref rppStatus_t</tt> enumeration.
 * \retval rppStatusSuccess
 * \retval rppStatusNotInitialized
 * \retval rppStatusInvalidValue
 * \retval rppStatusBadParm
 * \retval rppStatusAllocFailed
 * \retval rppStatusInternalError
 * \retval rppStatusNotImplemented
 * \retval rppStatusUnknownError
 * \retval rppStatusUnsupportedOp
 */
extern "C" SHARED_PUBLIC rppStatus_t rppCreateWithStreamAndBatchSize(rppHandle_t* handle, rppAcceleratorQueue_t stream, size_t nBatchSize);

/*! \brief Destory RPP GPU handle.
 * \details Function to destroy a RPP handle's device memory allocation. To be called in the end to break down the RPP environment.
 * \param [in] handle RPP handle of type <tt> \ref rppHandle_t</tt>.
 * \ingroup group_rpp
 * \return A <tt> \ref rppStatus_t</tt> enumeration.
 * \retval rppStatusSuccess
 * \retval rppStatusNotInitialized
 * \retval rppStatusInvalidValue
 * \retval rppStatusBadParm
 * \retval rppStatusAllocFailed
 * \retval rppStatusInternalError
 * \retval rppStatusNotImplemented
 * \retval rppStatusUnknownError
 * \retval rppStatusUnsupportedOp
 */
extern "C" SHARED_PUBLIC rppStatus_t rppDestroyGPU(rppHandle_t handle);

/*! \brief Set accelerator stream given a RPP handle.
 * \details Function to set an accelerator stream previously created.
 * \param [in] handle RPP handle of type <tt> \ref rppHandle_t</tt>.
 * \param [in] stream An accelerator queue of type <tt> \ref rppAcceleratorQueue_t</tt> (hipStream_t for HIP and cl_command_queue for OpenCL).
 * \ingroup group_rpp
 * \return A <tt> \ref rppStatus_t</tt> enumeration.
 * \retval rppStatusSuccess
 * \retval rppStatusNotInitialized
 * \retval rppStatusInvalidValue
 * \retval rppStatusBadParm
 * \retval rppStatusAllocFailed
 * \retval rppStatusInternalError
 * \retval rppStatusNotImplemented
 * \retval rppStatusUnknownError
 * \retval rppStatusUnsupportedOp
 */
extern "C" SHARED_PUBLIC rppStatus_t rppSetStream(rppHandle_t handle, rppAcceleratorQueue_t streamID);

/*! \brief Get accelerator stream given a RPP handle.
 * \details Function to get an accelerator stream previously created.
 * \param [in] handle RPP handle of type <tt> \ref rppHandle_t</tt>.
 * \param [in] stream An accelerator queue of type <tt> \ref rppAcceleratorQueue_t</tt> (hipStream_t for HIP and cl_command_queue for OpenCL).
 * \ingroup group_rpp
 * \return A <tt> \ref rppStatus_t</tt> enumeration.
 * \retval rppStatusSuccess
 * \retval rppStatusNotInitialized
 * \retval rppStatusInvalidValue
 * \retval rppStatusBadParm
 * \retval rppStatusAllocFailed
 * \retval rppStatusInternalError
 * \retval rppStatusNotImplemented
 * \retval rppStatusUnknownError
 * \retval rppStatusUnsupportedOp
 */
extern "C" SHARED_PUBLIC rppStatus_t rppGetStream(rppHandle_t handle, rppAcceleratorQueue_t* streamID);

/*! \brief Set allocator given a RPP handle.
 * \details Function to set allocator for a previously created RPP handle of type <tt> \ref rppHandle_t</tt>.
 * \param [in] handle RPP handle of type <tt> \ref rppHandle_t</tt>.
 * \param [in] allocator A callback function rpp will use for internal memory allocations. The provided callback function should allocate device memory with requested size and return a pointer to this memory. Passing 0 will restore the default RPP allocator and deallocator.
 * \param [in] deallocator A callback function rpp will use to for internal memory deallocation. The provided callback function should free the specified memory pointer.
 * \param [in] allocatorContext User-specified pointer which is passed to allocator and deallocator. This allows the callback function to access state set by the caller to this function, for example a stateful heap allocator or a c++ class.
 * \ingroup group_rpp
 * \return A <tt> \ref rppStatus_t</tt> enumeration.
 * \retval rppStatusSuccess
 * \retval rppStatusNotInitialized
 * \retval rppStatusInvalidValue
 * \retval rppStatusBadParm
 * \retval rppStatusAllocFailed
 * \retval rppStatusInternalError
 * \retval rppStatusNotImplemented
 * \retval rppStatusUnknownError
 * \retval rppStatusUnsupportedOp
 */
extern "C" SHARED_PUBLIC rppStatus_t rppSetAllocator(rppHandle_t handle, rppAllocatorFunction allocator, rppDeallocatorFunction deallocator, void* allocatorContext);

/*! \brief Get time taken by previous kernel.
 * \details Function to get time for last kernel launched. This function is used only when profiling mode has been enabled.
 * \param [in] handle RPP handle of type <tt> \ref rppHandle_t</tt>.
 * \param [in] time Pointer to a float type to contain kernel time in milliseconds.
 * \ingroup group_rpp
 * \return A <tt> \ref rppStatus_t</tt> enumeration.
 * \retval rppStatusSuccess
 * \retval rppStatusNotInitialized
 * \retval rppStatusInvalidValue
 * \retval rppStatusBadParm
 * \retval rppStatusAllocFailed
 * \retval rppStatusInternalError
 * \retval rppStatusNotImplemented
 * \retval rppStatusUnknownError
 * \retval rppStatusUnsupportedOp
 */
extern "C" SHARED_PUBLIC rppStatus_t rppGetKernelTime(rppHandle_t handle, float* time);

/*! \brief Enable Profiling.
 * \details Function to enable profiling to retrieve kernel time.
 * \param [in] handle RPP handle of type <tt> \ref rppHandle_t</tt>.
 * \param [in] enable Boolean to toggle profiling.
 * \ingroup group_rpp
 * \return A <tt> \ref rppStatus_t</tt> enumeration.
 * \retval rppStatusSuccess
 * \retval rppStatusNotInitialized
 * \retval rppStatusInvalidValue
 * \retval rppStatusBadParm
 * \retval rppStatusAllocFailed
 * \retval rppStatusInternalError
 * \retval rppStatusNotImplemented
 * \retval rppStatusUnknownError
 * \retval rppStatusUnsupportedOp
 */
extern "C" SHARED_PUBLIC rppStatus_t rppEnableProfiling(rppHandle_t handle, bool enable);

#endif // GPU_SUPPORT

#ifdef __cplusplus
}
#endif    // __cplusplus

#endif    // RPP_H
