#[[
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
]]

find_path(FFTS_INCLUDE_DIR
    NAMES ffts/ffts.h ffts.h
    PATHS
        /usr/local/include
        /usr/include
        /opt/local/include
        /opt/include
    PATH_SUFFIXES ffts
)

find_library(FFTS_LIBRARY
    NAMES ffts
    PATHS
        /usr/local/lib
        /usr/lib
        /usr/local/lib64
        /usr/lib64
        /opt/local/lib
        /opt/lib
)

# Mark the variables as advanced
mark_as_advanced(FFTS_INCLUDE_DIR FFTS_LIBRARY)

# Check if we found the library and headers
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTS
    REQUIRED_VARS FFTS_LIBRARY FFTS_INCLUDE_DIR
)

if(FFTS_FOUND)
    set(FFTS_LIBRARIES ${FFTS_LIBRARY})
    set(FFTS_INCLUDE_DIRS ${FFTS_INCLUDE_DIR})
endif()

# Create imported target
if(FFTS_FOUND AND NOT TARGET FFTS::FFTS)
    add_library(FFTS::FFTS SHARED IMPORTED)
    set_target_properties(FFTS::FFTS PROPERTIES
        IMPORTED_LOCATION "${FFTS_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${FFTS_INCLUDE_DIRS}"
    )
endif()
