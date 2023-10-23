# ROCm Performance Primitives Library

## Changelog

### Online Documentation
[RPP Documentation](https://gpuopen-professionalcompute-libraries.github.io/rpp/)

### RPP 1.4.0 (Unreleased)

#### Added
* Updated Unit Tests

#### Optimizations
* CMakeLists 

#### Changed
* Documentation
* Replaced the boost functions with std c++ library to remove boost library dependency

#### Fixed
* OCL Backend

### Tested Configurations
* Linux distribution
  + Ubuntu - `20.04` / `22.04`
  + CentOS - `7`
  + RHEL - `8`/`9`
* ROCm: rocm-core - `5.5.0.50500-63`
* Clang - Version `5.0.1` and above
* CMake - Version `3.22.3`
* IEEE 754-based half-precision floating-point library - Version `1.12.0`

### RPP 1.3.0

#### Added
* Updated Unit Tests

#### Optimizations
* CMakeLists 

#### Changed
* Documentation

#### Fixed
* OCL Backend

### Tested Configurations
* Linux distribution
  + Ubuntu - `20.04` / `22.04`
  + CentOS - `7`
  + RHEL - `8`/`9`
* ROCm: rocm-core - `5.5.0.50500-63`
* Clang - Version `5.0.1` and above
* CMake - Version `3.22.3`
* Boost - Version `1.72`
* IEEE 754-based half-precision floating-point library - Version `1.12.0`

### RPP 1.2.0

#### Added
* Updated Unit Tests

#### Optimizations
* CMakeLists 

#### Changed
* Documentation

#### Fixed
* OCL Backend
* Jenkins CI - OCL Build Test

### Tested Configurations
* Linux distribution
  + Ubuntu - `20.04` / `22.04`
  + CentOS - `7`
  + RHEL - `8`/`9`
* ROCm: rocm-core - `5.5.0.50500-63`
* Clang - Version `5.0.1` and above
* CMake - Version `3.22.3`
* Boost - Version `1.72`
* IEEE 754-based half-precision floating-point library - Version `1.12.0`

### Known issues
* 

### RPP 1.1.0

#### Added
* OpenMP - parameter support for OpenMP numthreads

#### Optimizations
* Readme Updates

#### Changed
* RPP - API Updated

#### Fixed
* Minor bugs

### Tested Configurations
* Linux distribution
  + Ubuntu - `20.04` / `22.04`
  + CentOS - `7`
  + RHEL - `8`/`9`
* ROCm: rocm-core - `5.5.0.50500-63`
* Clang - Version `5.0.1` and above
* CMake - Version `3.22.3`
* Boost - Version `1.72`
* IEEE 754-based half-precision floating-point library - Version `1.12.0`

### Known issues
* SLES - Clang package missing with latest updates. Need manual Clang install.

### RPP 1.0.0

#### Added
* Test Suite for all backends

#### Optimizations
* Readme Updates
* Tests
* Build & Prerequisites 

#### Changed
* Radeon Performance Primitives to ROCm Performance Primitives
* Lib Name - `amd_rpp` to `rpp`

#### Fixed
* Minor bugs
* Tests
* Readme

### Tested Configurations
* Linux distribution
  + Ubuntu - `20.04` / `22.04`
  + CentOS - `7`
  + RHEL - `8`/`9`
* ROCm: rocm-core - `5.4.0.50400-72`
* Clang - Version `5.0.1` and above
* CMake - Version `3.22.3` 
* Boost - Version `1.72`
* IEEE 754-based half-precision floating-point library - Version `1.12.0`

### Known issues
* SLES - Clang package missing with latest updates. Need manual Clang install.

### RPP 0.99

#### Added
* Linux Dockers

#### Optimizations
* Readme Updates

#### Changed
* CMakeList

#### Fixed
* Minor bugs and warnings

### Tested Configurations
* Linux distribution
  + Ubuntu - `20.04` / `22.04`
  + CentOS - `7`/`8`
* ROCm: rocm-core - `5.4.0.50400-72`
* Clang - Version `5.0.1` and above
* CMake - Version `3.22.3` 
* Boost - Version `1.72`
* IEEE 754-based half-precision floating-point library - Version `1.12.0`

### Known issues
* 

### RPP 0.98

#### Added
* Dockers

#### Optimizations
* Readme Updates

#### Changed
* CMakeList

#### Fixed
* Minor bugs and warnings

### Tested Configurations
* Linux distribution
  + Ubuntu - `20.04` / `22.04`
  + CentOS - `7`/`8`
* ROCm: rocm-core - `5.3.0.50300-63`
* Clang - Version `5.0.1` and above
* CMake - Version `3.22.3` 
* Boost - Version `1.72`
* IEEE 754-based half-precision floating-point library - Version `1.12.0`

### Known issues
* 

### RPP 0.97

#### Added
* Support for CentOS 7 & SLES 15 SP2
* Support for ROCm 5.3+ 
* Support for Ubuntu 22.04

#### Optimizations
* Readme Updates

#### Changed

* CMakeList Updates

#### Fixed
* Minor bugs and warnings

### Tested Configurations
* Linux distribution
  + Ubuntu - `20.04` / `22.04`
  + CentOS - `7`/`8`
* ROCm: rocm-core - `5.3.0.50300-36`
* Clang - Version `5.0.1` and above
* CMake - Version `3.22.3` 
* Boost - Version `1.72`
* IEEE 754-based half-precision floating-point library - Version `1.12.0`

### Known issues
* 

### RPP 0.96

#### Added
* New Tests

#### Optimizations
* Readme Updates

#### Changed

* `CPU`/`HIP`/`OpenCL` Backend Updates

#### Fixed
* Minor bugs and warnings

### Tested Configurations
* Linux distribution
  + Ubuntu - `18.04` / `20.04`
  + CentOS - `8`
* ROCm: rocm-core - `5.2.0.50200-65`
* Clang - Version `6.0+`
* CMake - Version `3.22.3`
* Boost - Version `1.72`
* IEEE 754-based half-precision floating-point library - Version `1.12.0`

### Known issues
* RPP Build on CentOS 7 & SLES SP2 Not Supported

### RPP 0.95

#### Added
* New Tests

#### Optimizations
* Readme Updates

#### Changed
* **Backend** - Default Backend set to `HIP`
* CPU Backend Added

#### Fixed
* Minor bugs and warnings

### Tested Configurations
* Linux distribution
  + Ubuntu - `18.04` / `20.04`
  + CentOS - `8`
* ROCm: rocm-core - `5.2.0.50200-65`
* Clang - Version `6.0`
* CMake - Version `3.22.3`
* Boost - Version `1.72`
* IEEE 754-based half-precision floating-point library - Version `1.12.0`

### Known issues
* `ROCm reorganization` - install updates does not match ROCm specification 

### RPP 0.93

#### Added
* New Tests

#### Optimizations
* Readme Updates

#### Changed
* **Backend** - Default Backend set to `HIP`

#### Fixed
* Minor bugs and warnings

### Tested Configurations
* Linux distribution
  + Ubuntu - `18.04` / `20.04`
  + CentOS - `8`
* ROCm: rocm-core - `5.0.0.50000-49`
* Clang - Version `6.0`
* CMake - Version `3.22.3`
* Boost - Version `1.72`
* IEEE 754-based half-precision floating-point library - Version `1.12.0`

### Known issues
* `CPU` only backend not enabled
