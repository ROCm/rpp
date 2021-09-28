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
#elif RPP_BACKEND_HIP
#include <hip/hip_runtime_api.h>
#endif

/*! Constructs type name from a struct */
#define RPP_DECLARE_OBJECT(name) \
    struct name                     \
    {                               \
    };                              \
    typedef struct name* name##_t;






#if RPP_BACKEND_OPENCL
typedef cl_command_queue rppAcceleratorQueue_t;
#elif RPP_BACKEND_HIP
typedef hipStream_t rppAcceleratorQueue_t;
#endif


/*! @ingroup handle
 * @brief Creates the rppHandle_t type
 */
RPP_DECLARE_OBJECT(rppHandle);



typedef rppHandle_t RppHandle_t;

#ifdef __cplusplus
extern "C" {
#endif

#include "rppcore.h"
#include "rppdefs.h"
#include "rppi.h"
#include "rppt.h"
#include "rppversion.h"



/*! @brief Get character string for an error code.
 *
 * A function which returns a NULL terminated character string of the error code.
 *
 * @param error  rppStatus_t type error status (input)
 * @return       errorString
*/
RPP_EXPORT const char* rppGetErrorString(rppStatus_t error);

/*! @brief Custom allocator function
 *
 * This function allow for user-defined custom allocator
 *
 * @param context     A pointer a context (input)
 * @param sizeBytes   Number of bytes to allocate (input)
 *
*/
typedef void* (*rppAllocatorFunction)(void* context, size_t sizeBytes);

/*! @brief Custom deallocator function
 *
 * This function allow for user-defined custom deallocation function
 *
 * @param context     A pointer context (input)
 * @param memory      A pointer allocated memory (input)
 *
*/
typedef void (*rppDeallocatorFunction)(void* context, void* memory);

/*! @brief Method to create the MIOpen handle object.
 *
 * This function creates a MIOpen handle. This is called at the very start to initialize the MIOpen
 * environment.
 * @param handle     A pointer to a MIOpen handle type (output)
 *
 * @return           rppStatus_t
*/
RPP_EXPORT rppStatus_t rppCreate(rppHandle_t* handle);

/*! @brief Create a MIOpen handle with an accelerator stream.
 *
 * The HIP side returns a hipStream_t type for the stream, while OpenCL will return a
 * cl_command_queue.
 *
 * Create a handle with a previously created accelerator command queue.
 * @param handle     A pointer to a MIOpen handle type (input)
 * @param stream      An accelerator queue type (output)
 *
 * @return           rppStatus_t
*/
RPP_EXPORT rppStatus_t rppCreateWithBatchSize(rppHandle_t* handle, size_t nBatchSize);

RPP_EXPORT rppStatus_t rppCreateWithStream(rppHandle_t* handle,
                                                    rppAcceleratorQueue_t stream);


RPP_EXPORT rppStatus_t rppCreateWithStreamAndBatchSize(rppHandle_t* handle,
                                                    rppAcceleratorQueue_t stream, size_t nBatchSize);
/*! @brief Destroys the MIOpen handle.
 *
 * This is called when breaking down the MIOpen environment.
 * @param handle     MIOpen handle (input)
 * @return           rppStatus_t
*/
RPP_EXPORT rppStatus_t rppDestroy(rppHandle_t handle);

RPP_EXPORT rppStatus_t rppDestroyGPU(rppHandle_t handle);
RPP_EXPORT rppStatus_t rppDestroyHost(rppHandle_t handle);
/*! @brief Set accelerator command queue previously created
 *
 * Set a command queue for an accelerator device
 * @param handle     MIOpen handle (input)
 * @param streamID   An accelerator queue type (input)
 * @return           rppStatus_t
*/
RPP_EXPORT rppStatus_t rppSetStream(rppHandle_t handle,
                                             rppAcceleratorQueue_t streamID);

/*! @brief Get the previously created accelerator command queue
 *
 * Creates a command queue for an accelerator device
 * @param handle     MIOpen handle (input)
 * @param streamID   Pointer to a accelerator queue type (output)
 * @return           rppStatus_t
*/
RPP_EXPORT rppStatus_t rppGetStream(rppHandle_t handle,
                                             rppAcceleratorQueue_t* streamID);

/*! @brief Set allocator for previously created rppHandle
 *
 * Set a command queue for an accelerator device
 * @param handle     MIOpen handle
 * @param allocator  A callback function MIOpen will use for internal memory allocations.
 *      The provided callback function should allocate device memory with requested size
 *      and return a pointer to this memory.
 *      Passing 0 will restore the default MIOpen allocator and deallocator.
 * @param deallocator  A callback function MIOpen will use to for internal memory deallocation.
 *      The provided callback function should free the specified memory pointer
 * @param allocatorContext  User-specified pointer which is passed to \p allocator and \p
 * deallocator
 *      This allows the callback function to access state set by the caller to this function,
 *      for example a stateful heap allocator or a c++ class.
 * @return           rppStatus_t
*/
RPP_EXPORT rppStatus_t rppSetAllocator(rppHandle_t handle,
                                                rppAllocatorFunction allocator,
                                                rppDeallocatorFunction deallocator,
                                                void* allocatorContext);

/*! @brief Get time for last kernel launched
 *
 * This function is used only when profiling mode has been enabled.
 * Kernel timings are based on the MIOpen handle and is not thread-safe.
 * In order to use multi-threaded profiling, create an MIOpen handle for each
 * concurrent thread.
 *
 * @param handle     MIOpen handle (input)
 * @param time       Pointer to a float type to contain kernel time in milliseconds (output)
 * @return           rppStatus_t
*/
RPP_EXPORT rppStatus_t rppGetKernelTime(rppHandle_t handle, float* time);

/*! @brief Enable profiling to retrieve kernel time
 *
 * Enable or disable kernel profiling. This profiling is only for kernel time.
 * @param handle     MIOpen handle (input)
 * @param enable     Boolean to toggle profiling (input)
 * @return           rppStatus_t
*/
RPP_EXPORT rppStatus_t rppEnableProfiling(rppHandle_t handle, bool enable);
/** @} */


/*TODO: Comments for rppSetBatchSize and rppGetBatchSize*/

RPP_EXPORT rppStatus_t rppGetBatchSize(rppHandle_t handle, size_t *batchSize);

RPP_EXPORT rppStatus_t rppSetBatchSize(rppHandle_t handle , size_t batchSize);



#ifdef __cplusplus
}
#endif

#endif
