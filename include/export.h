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

#ifndef RPP_EXPORT_H
#define RPP_EXPORT_H

#ifdef RPP_STATIC_DEFINE
#  define RPP_EXPORT
#  define RPP_NO_EXPORT
#else
#  ifndef RPP_EXPORT
#    ifdef MIOpen_EXPORTS
        /* We are building this library */
#      define RPP_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define RPP_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef RPP_NO_EXPORT
#    define RPP_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef RPP_DEPRECATED
#  define RPP_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef RPP_DEPRECATED_EXPORT
#  define RPP_DEPRECATED_EXPORT RPP_EXPORT RPP_DEPRECATED
#endif

#ifndef RPP_DEPRECATED_NO_EXPORT
#  define RPP_DEPRECATED_NO_EXPORT RPP_NO_EXPORT RPP_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef RPP_NO_DEPRECATED
#    define RPP_NO_DEPRECATED
#  endif
#endif

#endif /* RPP_EXPORT_H */
