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

#ifndef GUARD_RPP_MANAGE_PTR_HPP
#define GUARD_RPP_MANAGE_PTR_HPP

#include <memory>
#include <type_traits>

namespace rpp {

template <class F, F f>
struct manage_deleter
{
    template <class T>
    void operator()(T* x) const
    {
        if(x != nullptr)
        {
            f(x);
        }
    }
};

struct null_deleter
{
    template <class T>
    void operator()(T* /*x*/) const
    {
    }
};

template <class T, class F, F f>
using manage_ptr = std::unique_ptr<T, manage_deleter<F, f>>;

template <class T>
struct element_type
{
    using type = typename T::element_type;
};

template <class T>
using remove_ptr = typename std::conditional<std::is_pointer<T>::value,
                                             std::remove_pointer<T>,
                                             element_type<T>>::type::type;

template <class T>
using shared = std::shared_ptr<remove_ptr<T>>;

} // namespace rpp

#define RPP_MANAGE_PTR(T, F) \
    rpp::manage_ptr<typename std::remove_pointer<T>::type, decltype(&F), &F> // NOLINT

#endif
