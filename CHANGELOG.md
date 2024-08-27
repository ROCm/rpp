# Changelog for RPP

Full documentation for RPP is available at [https://rocm.docs.amd.com/projects/rpp/en/latest](https://rocm.docs.amd.com/projects/rpp/en/latest)
 
## RPP 1.9.1 for ROCm 6.3.0
 
### Changes
 
* Test - Prerequisites Updates
* RPP Glitch on HOST and HIP
* AMD Advanced - Build Flag
* RPP Audio Support HIP - Resample
* RPP Audio Support HIP - Pre emphasis filter
* RPP Pixelate - HOST and HIP
* RPP Audio Support HIP - Down-mixing 
* RPP Audio Support HIP - To Decibels
* RPP Audio Support HIP - Non silent region

### Removals
 
* TurboJPEG - older version
 
### Optimizations
 
* macOS - Build fix 
* Docs - changed roiTensorSrc to roiTensorPtrSrc in documentation
* Test Suite - updates 
 
### Resolved issues
 
* RPP Test Suite - augmentations fix 
* Copy - bugfix for `NCDHW` layout
* MIVisionX compatibility fix - Resample and pre-emphasis filter
* Docs - fix broken image links
 
### Known issues

* Package only supports HIP backend
 
### Upcoming changes
 
* Optimized audio augmentations 

### Tested configurations

* Linux distribution
  * Ubuntu - `20.04` / `22.04`
  * CentOS - `7`
  * RHEL - `8`/`9`
* ROCm: rocm-core - `6.3.0.60300`
* Clang - Version `5.0.1`+
* CMake - Version `3.16.3`+
* IEEE 754-based half-precision floating-point library - Version `1.12.0`


## RPP 1.8.0 for ROCm 6.2.0

### Changes

* Prerequisites - ROCm install requires only --usecase=rocm
* Use pre-allocated common scratchBufferHip everywhere in Tensor code for scratch HIP memory
* Use CHECK_RETURN_STATUS everywhere to adhere to C++17 for hip
* RPP Tensor Audio support on HOST for Spectrogram
* RPP Tensor Audio support on HOST/HIP for Slice, by modifying voxel slice kernels to now accept anchor and shape params for a more generic version
* RPP Tensor Audio support on HOST for Mel Filter Bank
* RPP Tensor Normalize ND support on HOST and HIP

### Tested configurations

* Linux distribution
  * Ubuntu - `20.04` / `22.04`
  * CentOS - `7`
  * RHEL - `8`/`9`
* ROCm: rocm-core - `6.1.0.60100`
* Clang - Version `5.0.1`
* CMake - Version `3.22.3`
* IEEE 754-based half-precision floating-point library - Version `1.12.0`

### RPP 1.5.0 for ROCm 6.1.1

### Changes

* Prerequisites

### Tested configurations

* Linux distribution
  * Ubuntu - `20.04` / `22.04`
  * CentOS - `7`
  * RHEL - `8`/`9`
* ROCm: rocm-core - `5.5.0.50500-63`
* Clang - Version `5.0.1` and above
* CMake - Version `3.22.3`
* IEEE 754-based half-precision floating-point library - Version `1.12.0`

## RPP 1.4.0 for ROCm 6.0.0

### Additions

* Updated unit tests
* Component-based packaging enabled for dev/runtime/ASan
* ASas build install/package changes added
* License file added to package
* Jenkins Groovy CI scripts enhanced to support multi-component package testing

### Optimizations

* CMakeLists

### Changes

* Documentation
* Replaced boost functions with the standard C++ library to remove boost library dependency

### Fixes

* OCL backend

### Tested configurations

* Linux
  * Ubuntu - `20.04` / `22.04`
  * CentOS - `7`
  * RHEL - `8`/`9`
* ROCm: rocm-core - `5.5.0.50500-63`
* Clang - Version `5.0.1` and above
* CMake - Version `3.22.3`
* IEEE 754-based half-precision floating-point library - Version `1.12.0`

## RPP 1.3.0 for ROCm 5.7.1

### Additions

* Updated unit tests

### Optimizations

* CMakeLists

### Changes

* Documentation

### Fixes

* OCL backend

### Tested configurations

* Linux
  * Ubuntu - `20.04` / `22.04`
  * CentOS - `7`
  * RHEL - `8`/`9`
* ROCm: rocm-core - `5.5.0.50500-63`
* Clang - Version `5.0.1` and above
* CMake - Version `3.22.3`
* Boost - Version `1.72`
* IEEE 754-based half-precision floating-point library - Version `1.12.0`

## RPP 1.2.0 for ROCm 5.7.1

### Additions

* Updated unit tests

### Optimizations

* CMakeLists

### Changes

* Documentation

### Fixes

* OCL backend
* Jenkins CI - OCL Build Test

### Tested configurations

* Linux
  * Ubuntu - `20.04` / `22.04`
  * CentOS - `7`
  * RHEL - `8`/`9`
* ROCm: rocm-core - `5.5.0.50500-63`
* Clang - Version `5.0.1` and above
* CMake - Version `3.22.3`
* Boost - Version `1.72`
* IEEE 754-based half-precision floating-point library - Version `1.12.0`

## RPP 1.1.0 for ROCm 5.7.0

### Additions

* Parameter support for OpenMP numthreads

### Optimizations

* Readme updates

### Changes

* RPP API updates

### Fixes

* Minor bugs

### Tested configurations

* Linux
  * Ubuntu - `20.04` / `22.04`
  * CentOS - `7`
  * RHEL - `8`/`9`
* ROCm: rocm-core - `5.5.0.50500-63`
* Clang - Version `5.0.1` and above
* CMake - Version `3.22.3`
* Boost - Version `1.72`
* IEEE 754-based half-precision floating-point library - Version `1.12.0`

### Known issues

* SLES - the Clang package is missing in the latest updates, which means Clang must be manually
  installed.

## RPP 1.0.0 for ROCm 5.7.0

### Additions

* Test Suite for all backends

### Optimizations

* Readme updates
* Tests
* Build and prerequisites

### Changes

* Our name has changed from *Radeon Performance Primitives* to *ROCm Performance Primitives*
* Lib name: `amd_rpp` to `rpp`

### Fixes

* Minor bugs
* Tests
* Readme

### Tested configurations

* Linux
  * Ubuntu - `20.04` / `22.04`
  * CentOS - `7`
  * RHEL - `8`/`9`
* ROCm: rocm-core - `5.4.0.50400-72`
* Clang - Version `5.0.1` and above
* CMake - Version `3.22.3`
* Boost - Version `1.72`
* IEEE 754-based half-precision floating-point library - Version `1.12.0`

### Known issues

* SLES - the Clang package is missing in the latest updates, which means Clang must be manually
  installed.

## RPP 0.99 for ROCm 5.7.0

### Additions

* Linux dockers

### Optimizations

* Readme updates

### Changes

* CMakeList

### Fixes

* Minor bugs and warnings

### Tested configurations

* Linux distribution
  * Ubuntu - `20.04` / `22.04`
  * CentOS - `7`/`8`
* ROCm: rocm-core - `5.4.0.50400-72`
* Clang - Version `5.0.1` and above
* CMake - Version `3.22.3`
* Boost - Version `1.72`
* IEEE 754-based half-precision floating-point library - Version `1.12.0`

## RPP 0.98 for ROCm 5.7.0

### Additions
* Dockers


### Optimizations

* Readme updates

### Changes

* CMakeList

### Fixes

* Minor bugs and warnings

### Tested configurations

* Linux
  * Ubuntu - `20.04` / `22.04`
  * CentOS - `7`/`8`
* ROCm: rocm-core - `5.3.0.50300-63`
* Clang - Version `5.0.1` and above
* CMake - Version `3.22.3`
* Boost - Version `1.72`
* IEEE 754-based half-precision floating-point library - Version `1.12.0`

## RPP 0.97 for ROCm 5.7.0

### Additions

* Support for CentOS 7 & SLES 15 SP2
* Support for ROCm 5.3+
* Support for Ubuntu 22.04

### Optimizations

* Readme updates

### Changes

* CMakeList updates

### Fixes

* Minor bugs and warnings

### Tested configurations

* Linux distribution
  * Ubuntu - `20.04` / `22.04`
  * CentOS - `7`/`8`
* ROCm: rocm-core - `5.3.0.50300-36`
* Clang - Version `5.0.1` and above
* CMake - Version `3.22.3`
* Boost - Version `1.72`
* IEEE 754-based half-precision floating-point library - Version `1.12.0`

## RPP 0.96 for ROCm 5.7.0

### Additions

* New tests

### Optimizations

* Readme updates

### Changes

* `CPU`/`HIP`/`OpenCL` backend updates

### Fixes

* Minor bugs and warnings

### Tested configurations

* Linux
  * Ubuntu - `18.04` / `20.04`
  * CentOS - `8`
* ROCm: rocm-core - `5.2.0.50200-65`
* Clang - Version `6.0+`
* CMake - Version `3.22.3`
* Boost - Version `1.72`
* IEEE 754-based half-precision floating-point library - Version `1.12.0`

### Known issues

* RPP is not supported on CentOS 7 and SLES SP2

## RPP 0.95 for ROCm 5.7.0

### Additions

* New tests
* CPU backend support

### Optimizations

* Readme updates

### Changes

* `HIP` is now the default backend

### Fixes

* Minor bugs and warnings

### Tested configurations

* Linux
  * Ubuntu - `18.04` / `20.04`
  * CentOS - `8`
* ROCm: rocm-core - `5.2.0.50200-65`
* Clang - Version `6.0`
* CMake - Version `3.22.3`
* Boost - Version `1.72`
* IEEE 754-based half-precision floating-point library - Version `1.12.0`

### Known issues

* ROCm reorganization: install updates no longer match ROCm specifications

## RPP 0.93 for ROCm 5.7.0

### Additions

* New tests

### Optimizations

* Readme updates

### Changes

* `HIP` is now the default backend

### Fixes

* Minor bugs and warnings

### Tested configurations

* Linux
  * Ubuntu - `18.04` / `20.04`
  * CentOS - `8`
* ROCm: rocm-core - `5.0.0.50000-49`
* Clang - Version `6.0`
* CMake - Version `3.22.3`
* Boost - Version `1.72`
* IEEE 754-based half-precision floating-point library - Version `1.12.0`

### Known issues

* `CPU` backend is not enabled
