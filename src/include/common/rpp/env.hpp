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

#ifndef GUARD_RPP_ENV_HPP
#define GUARD_RPP_ENV_HPP

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace rpp {

/// \todo Rework: Case-insensitive string compare, ODR, (?) move to .cpp

// Declare a cached environment variable
#define RPP_DECLARE_ENV_VAR(x)                 \
    struct x                                      \
    {                                             \
        static const char* value() { return #x; } \
    };

/*
 * Returns false if a feature-controlling environment variable is defined
 * and set to something which disables a feature.
 */
inline bool IsEnvvarValueDisabled(const char* name)
{
    const auto value_env_p = std::getenv(name);
    return value_env_p != nullptr &&
           (std::strcmp(value_env_p, "disable") == 0 || std::strcmp(value_env_p, "disabled") == 0 ||
            std::strcmp(value_env_p, "0") == 0 || std::strcmp(value_env_p, "no") == 0 ||
            std::strcmp(value_env_p, "false") == 0);
}

inline bool IsEnvvarValueEnabled(const char* name)
{
    const auto value_env_p = std::getenv(name);
    return value_env_p != nullptr &&
           (std::strcmp(value_env_p, "enable") == 0 || std::strcmp(value_env_p, "enabled") == 0 ||
            std::strcmp(value_env_p, "1") == 0 || std::strcmp(value_env_p, "yes") == 0 ||
            std::strcmp(value_env_p, "true") == 0);
}

// Return 0 if env is enabled else convert environment var to an int.
// Supports hexadecimal with leading 0x or decimal
inline unsigned long int EnvvarValue(const char* name)
{
    const auto value_env_p = std::getenv(name);
    if(value_env_p == nullptr)
    {
        return 0;
    }
    else
    {
        return strtoul(value_env_p, nullptr, 0);
    }
}

inline std::vector<std::string> GetEnv(const char* name)
{
    auto p = std::getenv(name);
    if(p == nullptr)
        return {};
    else
        return {{p}};
}

template <class T>
inline const char* GetStringEnv(T)
{
    static const std::vector<std::string> result = GetEnv(T::value());
    if(result.empty())
        return nullptr;
    else
        return result.front().c_str();
}

template <class T>
inline bool IsEnabled(T)
{
    static const bool result = rpp::IsEnvvarValueEnabled(T::value());
    return result;
}

template <class T>
inline bool IsDisabled(T)
{
    static const bool result = rpp::IsEnvvarValueDisabled(T::value());
    return result;
}

template <class T>
inline unsigned long int Value(T)
{
    static const auto result = rpp::EnvvarValue(T::value());
    return result;
}
} // namespace rpp

RPP_DECLARE_ENV_VAR(RPP_DEBUG_AMD_ROCM_PRECOMPILED_BINARIES)
RPP_DECLARE_ENV_VAR(HOME)

#endif
