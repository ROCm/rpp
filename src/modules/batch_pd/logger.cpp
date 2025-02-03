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

#include <cstdlib>

#ifdef __linux__
#include <unistd.h>
#include <sys/syscall.h> /* For SYS_xxx definitions */
#endif

#include "rpp/env.hpp"
#include "rpp/logger.hpp"

namespace rpp {

/// Enable logging of the most important function calls.
/// Name of envvar in a bit inadequate due to historical reasons.
RPP_DECLARE_ENV_VAR(RPP_ENABLE_LOGGING)

/// Prints driver command lines into log.
/// Works from any application which uses the library.
/// Allows to reproduce library use cases using the driver instead of the actual application.
RPP_DECLARE_ENV_VAR(RPP_ENABLE_LOGGING_CMD)

/// Prefix each log line with information which allows the user
/// to uniquiely identify log records printed from different processes
/// or threads. Useful for debugging multi-process/multi-threaded apps.
RPP_DECLARE_ENV_VAR(RPP_ENABLE_LOGGING_MPMT)

/// See LoggingLevel in the header.
RPP_DECLARE_ENV_VAR(RPP_LOG_LEVEL)

namespace {

inline bool operator!=(const int& lhs, const LoggingLevel& rhs)
{
    return lhs != static_cast<int>(rhs);
}
inline bool operator>=(const int& lhs, const LoggingLevel& rhs)
{
    return lhs >= static_cast<int>(rhs);
}

/// Returns value which uniquiely identifies current process/thread
/// and can be printed into logs for MP/MT environments.
inline int GetProcessAndThreadId()
{
#ifdef __linux__
    // LWP is fine for identifying both processes and threads.
    return syscall(SYS_gettid); // NOLINT
#else
    return 0; // Not implemented.
#endif
}

} // namespace

bool IsLoggingFunctionCalls() { return rpp::IsEnabled(RPP_ENABLE_LOGGING{}); }

bool IsLogging(const LoggingLevel level)
{
    const auto enabled_level = rpp::Value(RPP_LOG_LEVEL{});
    if(enabled_level != LoggingLevel::Default)
        return enabled_level >= level;
#ifdef NDEBUG // Simplest way.
    return LoggingLevel::Warning >= level;
#else
    return LoggingLevel::Info >= level;
#endif
}

const char* LoggingLevelToCString(const LoggingLevel level)
{
    // Intentionally straightforward.
    // The most frequently used come first.
    if(level == LoggingLevel::Error)
        return "Error";
    else if(level == LoggingLevel::Warning)
        return "Warning";
    else if(level == LoggingLevel::Info)
        return "Info";
    else if(level == LoggingLevel::Info2)
        return "Info2";
    else if(level == LoggingLevel::Trace)
        return "Trace";
    else if(level == LoggingLevel::Default)
        return "Default";
    else if(level == LoggingLevel::Quiet)
        return "Quiet";
    else if(level == LoggingLevel::Fatal)
        return "Fatal";
    else
        return "<Unknown>";
}
bool IsLoggingCmd() { return rpp::IsEnabled(RPP_ENABLE_LOGGING_CMD{}); }

std::string LoggingPrefix()
{
    std::stringstream ss;
    if(rpp::IsEnabled(RPP_ENABLE_LOGGING_MPMT{}))
    {
        ss << GetProcessAndThreadId() << ' ';
    }
    ss << "RPP";
#if OCL_COMPILE
    ss << "(OpenCL)";
#elif HIP_COMPILE
    ss << "(HIP)";
#endif
    ss << ": ";
    return ss.str();
}

/// Expected to be invoked with __func__ and __PRETTY_FUNCTION__.
std::string LoggingParseFunction(const char* func, const char* pretty_func)
{
    const std::string fname{func};
    if(fname != "operator()")
        return fname;
    // lambda
    const std::string pf{pretty_func};
    const std::string pf_tail{pf.substr(0, pf.find_first_of('('))};
    return pf_tail.substr(1 + pf_tail.find_last_of(':'));
}

} // namespace rpp
