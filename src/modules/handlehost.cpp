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

#include <boost/filesystem.hpp>
#ifndef _WIN32
#include <unistd.h>
#endif

#include<thread>
#include "config.h"
#include "rpp/logger.hpp"
#include "rpp/handle.hpp"

namespace rpp {

#if !GPU_SUPPORT

struct HandleImpl
{
    size_t nBatchSize = 1;
    Rpp32u numThreads = 0;
    InitHandle* initHandle = nullptr;

    void PreInitializeBufferCPU()
    {
        this->initHandle = new InitHandle();
        this->initHandle->nbatchSize = this->nBatchSize;
        this->initHandle->mem.mcpu.maxSrcSize = (RppiSize *)malloc(sizeof(RppiSize) * this->nBatchSize);
        this->initHandle->mem.mcpu.maxDstSize = (RppiSize *)malloc(sizeof(RppiSize) * this->nBatchSize);
        this->initHandle->mem.mcpu.roiPoints = (RppiROI *)malloc(sizeof(RppiROI) * this->nBatchSize);
        this->initHandle->mem.mcpu.tempFloatmem = (Rpp32f *)malloc(sizeof(Rpp32f) * 99532800 * this->nBatchSize); // 7680 * 4320 * 3
    }
};

Handle::Handle(size_t batchSize, Rpp32u numThreads) : impl(new HandleImpl())
{
    impl->nBatchSize = batchSize;
    numThreads = std::min(numThreads, std::thread::hardware_concurrency());
    if(numThreads == 0)
        numThreads = batchSize;
    impl->numThreads = numThreads;
    impl->PreInitializeBufferCPU();
}

Handle::Handle() : impl(new HandleImpl())
{
    impl->PreInitializeBufferCPU();
    impl->numThreads = std::min(impl->numThreads, std::thread::hardware_concurrency());
    if(impl->numThreads == 0)
        impl->numThreads = impl->nBatchSize;
    RPP_LOG_I(*this);
}

Handle::~Handle() {}

void Handle::rpp_destroy_object_host()
{
    free(this->GetInitHandle()->mem.mcpu.maxSrcSize);
    free(this->GetInitHandle()->mem.mcpu.maxDstSize);
    free(this->GetInitHandle()->mem.mcpu.roiPoints);
    free(this->GetInitHandle()->mem.mcpu.tempFloatmem);
}

size_t Handle::GetBatchSize() const
{
    return this->impl->nBatchSize;
}

Rpp32u Handle::GetNumThreads() const
{
    return this->impl->numThreads;
}

void Handle::SetBatchSize(size_t bSize) const
{
    this->impl->nBatchSize = bSize;
}

InitHandle* Handle::GetInitHandle() const
{
    return impl->initHandle;
}

#endif // GPU_SUPPORT

} // namespace rpp
