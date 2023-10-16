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

#ifndef GUARD_RPP_WRITE_FILE_HPP
#define GUARD_RPP_WRITE_FILE_HPP

#include <fstream>
#include "filesystem.h"

#include "rpp/manage_ptr.hpp"

namespace rpp {

using FilePtr = RPP_MANAGE_PTR(FILE*, std::fclose);

inline void WriteFile(const std::string& content, const fs::path& name)
{
    FilePtr f{std::fopen(name.string().c_str(), "w")};
    if(std::fwrite(content.c_str(), 1, content.size(), f.get()) != content.size())
        RPP_THROW("Failed to write to src file");
}
} // namespace rpp

#endif
