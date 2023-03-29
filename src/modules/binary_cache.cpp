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

#include <fstream>
#include <iostream>
#include <boost/filesystem.hpp>

#include "rpp.h"
#include "rppversion.h"
#include "rpp/binary_cache.hpp"
#include "rpp/errors.hpp"
#include "rpp/env.hpp"
#include "rpp/stringutils.hpp"
#include "rpp/md5.hpp"

namespace rpp {

RPP_DECLARE_ENV_VAR(RPP_DISABLE_CACHE)

boost::filesystem::path ComputeCachePath()
{
#ifdef RPP_CACHE_DIR
    std::string cache_dir = RPP_CACHE_DIR;

    std::string version = std::to_string(RPP_VERSION_MAJOR) + "." +
                          std::to_string(RPP_VERSION_MINOR) + "." +
                          std::to_string(RPP_VERSION_PATCH);

    auto p = boost::filesystem::path{cache_dir} / version;
    if(!boost::filesystem::exists(p))
        boost::filesystem::create_directories(p);
    // auto p = cache_dir;
    //std::cerr<<"\n cache_dir"<<cache_dir;
    return p;
#else
    return {};
#endif
}

boost::filesystem::path GetCachePath()
{
    static const boost::filesystem::path path = ComputeCachePath();
    return path;
}

bool IsCacheDisabled()
{

#ifdef RPP_CACHE_DIR
    // std::cerr<<"\n Coming to IsCacheDisabled in binary_cache.cpp";
    return rpp::IsEnabled(RPP_DISABLE_CACHE{});
#else
    return true;
#endif
}

boost::filesystem::path GetCacheFile(const std::string& device,
                                     const std::string& name,
                                     const std::string& args,
                                     bool is_kernel_str)
{
    // std::cerr<<"\n Coming to GetCacheFile in binary_cache.cpp";
    std::string filename = (is_kernel_str ? rpp::md5(name) : name) + ".o";
    return GetCachePath() / rpp::md5(device + ":" + args) / filename;
}

std::string LoadBinary(const std::string& device,
                       const std::string& name,
                       const std::string& args,
                       bool is_kernel_str)
{
    // std::cerr<<"\n Coming to LoadBinary in binary_cache.cpp";
    if(rpp::IsCacheDisabled())
        return {};
    auto f = GetCacheFile(device, name, args, is_kernel_str);
    if(boost::filesystem::exists(f))
    {
        return f.string();
    }
    else
    {
        return {};
    }
}
void SaveBinary(const boost::filesystem::path& binary_path,
                const std::string& device,
                const std::string& name,
                const std::string& args,
                bool is_kernel_str)
{
    if(rpp::IsCacheDisabled())
    {
        boost::filesystem::remove(binary_path);
    }
    else
    {
        auto p = GetCacheFile(device, name, args, is_kernel_str);
        boost::filesystem::create_directories(p.parent_path());
        boost::filesystem::rename(binary_path, p);
    }
}

} // namespace rpp
