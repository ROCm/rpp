/*
MIT License

Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "rpp/clhelper.hpp"
#include "rpp/errors.hpp"
#include "rpp/gcn_asm_utils.hpp"
#include "rpp/hip_build_utils.hpp"
#include "rpp/kernel.hpp"
#include "rpp/kernel_warnings.hpp"
#include "rpp/stringutils.hpp"
#include "rpp/ocldeviceinfo.hpp"
#include "rpp/tmp_dir.hpp"
#include "rpp/write_file.hpp"

namespace rpp {

void ParseDevName(std::string& name)
{
    auto loc_p = name.find('+');
    if(loc_p != std::string::npos)
        name = name.substr(0, loc_p);
}

static cl_program CreateProgram(cl_context ctx, const char* char_source, size_t size)
{
    cl_int status;
    auto result = clCreateProgramWithSource(ctx, 1, &char_source, &size, &status);
    if(status != CL_SUCCESS)
        RPP_THROW_CL_STATUS(status, "Error Creating OpenCL Program (cl_program) in LoadProgram()");

    return result;
}

static void ClAssemble(cl_device_id device, std::string& source, const std::string& params)
{
    // Add device nmae
    std::string name = rpp::GetDeviceInfo<CL_DEVICE_NAME>(device);
    ParseDevName(name);
    AmdgcnAssemble(source, std::string("-mcpu=") + name + " " + params);
}

static cl_program CreateProgramWithBinary(cl_context ctx, cl_device_id device, const char* char_source, size_t size)
{
    cl_int status, binaryStatus;
    auto result = clCreateProgramWithBinary(ctx,
                                            1,
                                            &device,
                                            reinterpret_cast<const size_t*>(&size),
                                            reinterpret_cast<const unsigned char**>(&char_source),
                                            &status,
                                            &binaryStatus);

    if(status != CL_SUCCESS)
        RPP_THROW_CL_STATUS(status, "Error creating code object program (cl_program) in LoadProgramFromBinary()");

    return result;
}

static std::string BuildProgramInfo(cl_program program, cl_device_id device)
{
    std::vector<char> errorbuf(1024 * 1024);
    size_t psize;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 1024 * 1024, errorbuf.data(), &psize);
    return errorbuf.data();
}

static void BuildProgram(cl_program program, cl_device_id device, const std::string& params = "")
{
    auto status = clBuildProgram(program, 1, &device, params.c_str(), nullptr, nullptr);

    auto msg = BuildProgramInfo(program, device);
    if(!msg.empty())
        std::cerr << msg << std::endl;

    if(status != CL_SUCCESS)
        RPP_THROW_CL_STATUS(status, "Error Building OpenCL Program in BuildProgram()\n" + BuildProgramInfo(program, device));
}

ClProgramPtr LoadBinaryProgram(cl_context ctx, cl_device_id device, const std::string& source)
{
    ClProgramPtr result{CreateProgramWithBinary(ctx, device, source.data(), source.size())};
    BuildProgram(result.get(), device);
    return result;
}

ClProgramPtr LoadProgram(cl_context ctx, cl_device_id device, const std::string& program_name, std::string params, bool is_kernel_str, const std::string& kernel_src)
{
    bool is_binary = false;
    std::string source;
    if(is_kernel_str)
    {
        source = program_name;
    }
    else
    {
        if(kernel_src.empty())
            source = rpp::GetKernelSrc(program_name);
        else
            source  = kernel_src;
        auto is_asm = rpp::EndsWith(program_name, ".s");
        if(is_asm)
        { // Overwrites source (asm text) by binary results of assembly:
            ClAssemble(device, source, params);
            is_binary = true;
        }
        else
        {
            is_binary = rpp::EndsWith(program_name, ".so");
        }
    }

    if(is_binary)
    {
        return LoadBinaryProgram(ctx, device, source);
    }
    else if(!is_kernel_str && rpp::EndsWith(program_name, ".cpp"))
    {
        std::string device_name = rpp::GetDeviceInfo<CL_DEVICE_NAME>(device);
        ParseDevName(device_name);
        std::optional<rpp::TmpDir> dir(program_name);
#if RPP_BUILD_DEV
        params += " -Werror";
#ifdef __linux__
        params += HipKernelWarningsString();
#endif
#endif
        auto hsaco_file = HipBuild(dir, program_name, source, params, device_name);
        // load the hsaco file as a data stream and then load the binary
        std::string buf;
        bin_file_to_str(hsaco_file, buf);
        return LoadBinaryProgram(ctx, device, buf);
    }
    else
    {
        ClProgramPtr result{CreateProgram(ctx, source.data(), source.size())};
#if RPP_BUILD_DEV
        params += " -Werror";
#ifdef __linux__
        params += OclKernelWarningsString();
#endif
#endif
        params += " -cl-std=CL1.2";
        BuildProgram(result.get(), device, params);
        return result;
    }
}

void SaveProgramBinary(const ClProgramPtr& program, const std::string& name)
{
    size_t binary_size;
    clGetProgramInfo(program.get(), CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, nullptr);
    std::vector<char> binary(binary_size);
    char* src[1] = {binary.data()};
    clGetProgramInfo(program.get(), CL_PROGRAM_BINARIES, sizeof(src), &src, nullptr);
    std::ofstream fout(name.c_str(), std::ios::out | std::ios::binary);
    fout.write(binary.data(), binary.size());
}

ClKernelPtr CreateKernel(cl_program program, const std::string& kernel_name)
{
    cl_int status;
    ClKernelPtr result{clCreateKernel(program, kernel_name.c_str(), &status)};

    if(status != CL_SUCCESS)
    {
        RPP_THROW_CL_STATUS(status);
    }

    return result;
}

cl_device_id GetDevice(cl_command_queue q)
{
    cl_device_id device;
    cl_int status =
        clGetCommandQueueInfo(q, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, nullptr);
    if(status != CL_SUCCESS)
    {
        RPP_THROW_CL_STATUS(status, "Error Getting Device Info from Queue in GetDevice()");
    }

    return device;
}

cl_context GetContext(cl_command_queue q)
{
    cl_context context;
    cl_int status =
        clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, nullptr);
    if(status != CL_SUCCESS)
    {
        RPP_THROW_CL_STATUS(status, "Error Getting Device Info from Queue in GetDevice()");
    }
    return context;
}

} // namespace rpp
