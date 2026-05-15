/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

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

#include "logger.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>

namespace rpp {

// ─── Level names ─────────────────────────────────────────────────────────────

const char* LoggingLevelToCString(LoggingLevel level)
{
    switch(level)
    {
    case LoggingLevel::Default: return "Default";
    case LoggingLevel::Quiet:   return "Quiet";
    case LoggingLevel::Fatal:   return "Fatal";
    case LoggingLevel::Error:   return "Error";
    case LoggingLevel::Warning: return "Warning";
    case LoggingLevel::Info:    return "Info";
    case LoggingLevel::Info2:   return "Info2";
    case LoggingLevel::Trace:   return "Trace";
    default:                    return "Unknown";
    }
}

// ─── Env-var helpers ──────────────────────────────────────────────────────────

// RPP_LOG_LEVEL=N  where N matches LoggingLevel enum (0-7).
// If unset, defaults to Warning (4) for Release, Info (5) for Debug.
static LoggingLevel configured_level()
{
    const char* val = std::getenv("RPP_LOG_LEVEL");
    if(val == nullptr || val[0] == '\0')
    {
#ifdef NDEBUG
        return LoggingLevel::Warning;   // Release default
#else
        return LoggingLevel::Info;      // Debug default
#endif
    }
    int n = std::atoi(val);
    n = std::max(0, std::min(n, 7));
    return static_cast<LoggingLevel>(n);
}

// ─── Public API ───────────────────────────────────────────────────────────────

// Returns true when `level` is at or below the configured verbosity.
// Higher LoggingLevel values = more verbose.
bool IsLogging(LoggingLevel level)
{
    static LoggingLevel env_level = configured_level();
    return static_cast<int>(level) <= static_cast<int>(env_level);
}

// RPP_LOG_CMD=1 enables driver command logging.
bool IsLoggingCmd()
{
    static bool val = (std::getenv("RPP_LOG_CMD") != nullptr);
    return val;
}

// RPP_LOG_FUNCTION_CALLS=1 enables per-parameter function-call logging.
bool IsLoggingFunctionCalls()
{
    static bool val = (std::getenv("RPP_LOG_FUNCTION_CALLS") != nullptr);
    return val;
}

std::string LoggingPrefix()
{
    return "[RPP] ";
}

// Extracts the bare function name from __func__ / __PRETTY_FUNCTION__.
std::string LoggingParseFunction(const char* func, const char* /*pretty_func*/)
{
    return func;
}

} // namespace rpp
