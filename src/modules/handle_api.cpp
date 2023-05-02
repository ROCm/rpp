/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <cstdio>
#include "rpp/errors.hpp"
#include "rpp/handle.hpp"

extern "C" const char* rppGetErrorString(rppStatus_t error)
{
    switch(error)
    {
    case rppStatusSuccess: return "rppStatusSuccess";

    case rppStatusNotInitialized: return "rppStatusNotInitialized";

    case rppStatusInvalidValue: return "rppStatusInvalidValue";

    case rppStatusBadParm: return "rppStatusBadParm";

    case rppStatusAllocFailed: return "rppStatusAllocFailed";

    case rppStatusInternalError: return "rppStatusInternalError";

    case rppStatusNotImplemented: return "rppStatusNotImplemented";

    case rppStatusUnknownError: return "rppStatusUnknownError";

    case rppStatusUnsupportedOp: return "rppStatusUnsupportedOp";
    }
    return "Unknown error status";
}

extern "C" rppStatus_t rppCreate(rppHandle_t* handle)
{
    return rpp::try_([&] { rpp::deref(handle) = new rpp::Handle(); });
}

extern "C" rppStatus_t rppCreateWithBatchSize(rppHandle_t* handle, size_t nBatchSize, Rpp32u numThreads)
{
    return rpp::try_([&] { rpp::deref(handle) = new rpp::Handle(nBatchSize, numThreads); });
}

extern "C" rppStatus_t rppDestroy(rppHandle_t handle)
{
    return rpp::try_([&] { rpp_destroy_object(handle); });
}

extern "C" rppStatus_t rppDestroyHost(rppHandle_t handle)
{
    return rpp::try_([&] { rpp::deref(handle).rpp_destroy_object_host(); });
}

extern "C" rppStatus_t rppSetBatchSize(rppHandle_t handle, size_t batchSize)
{
    return rpp::try_([&] { rpp::deref(handle).SetBatchSize(batchSize); });
}

extern "C" rppStatus_t rppGetBatchSize(rppHandle_t handle, size_t *batchSize)
{
    return rpp::try_([&] { rpp::deref(batchSize) = rpp::deref(handle).GetBatchSize(); });
}

#if GPU_SUPPORT

extern "C" rppStatus_t rppCreateWithStream(rppHandle_t* handle, rppAcceleratorQueue_t stream)
{
    return rpp::try_([&] { rpp::deref(handle) = new rpp::Handle(stream); });
}

extern "C" rppStatus_t rppCreateWithStreamAndBatchSize(rppHandle_t* handle, rppAcceleratorQueue_t stream, size_t nBatchSize)
{
    return rpp::try_([&] { rpp::deref(handle) = new rpp::Handle(stream, nBatchSize); });
}

extern "C" rppStatus_t rppDestroyGPU(rppHandle_t handle)
{
    return rpp::try_([&] { rpp::deref(handle).rpp_destroy_object_gpu(); });
}

extern "C" rppStatus_t rppSetStream(rppHandle_t handle, rppAcceleratorQueue_t streamID)
{
    return rpp::try_([&] { rpp::deref(handle).SetStream(streamID); });
}

extern "C" rppStatus_t rppGetStream(rppHandle_t handle, rppAcceleratorQueue_t* streamID)
{
    return rpp::try_([&] { rpp::deref(streamID) = rpp::deref(handle).GetStream(); });
}

extern "C" rppStatus_t rppSetAllocator(rppHandle_t handle, rppAllocatorFunction allocator, rppDeallocatorFunction deallocator, void* allocatorContext)
{
    return rpp::try_([&] { rpp::deref(handle).SetAllocator(allocator, deallocator, allocatorContext); });
}

extern "C" rppStatus_t rppGetKernelTime(rppHandle_t handle, float* time)
{
    return rpp::try_([&] { rpp::deref(time) = rpp::deref(handle).GetKernelTime(); });
}

extern "C" rppStatus_t rppEnableProfiling(rppHandle_t handle, bool enable)
{
    return rpp::try_([&] { rpp::deref(handle).EnableProfiling(enable); });
}

#endif // GPU_SUPPORT
