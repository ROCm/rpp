[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![doc](https://img.shields.io/badge/doc-readthedocs-blueviolet)](https://gpuopen-professionalcompute-libraries.github.io/rpp/)

# ROCm Performance Primitives library

AMD ROCm Performance Primitives (RPP) library is a comprehensive, high-performance computer
vision library for AMD processors that have `HIP`, `OpenCL`, or `CPU` backends.

<p align="center"><img width="50%" src="https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp/raw/master/docs/data/rpp_structure_4.png" /></p>

The latest RPP release is: [![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/GPUOpen-ProfessionalCompute-Libraries/rpp?style=for-the-badge)](https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp/releases)

## Supported functionalities and variants

<p align="center"><img width="90%" src="https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp/raw/master/docs/data/supported_functionalities.png" /></p>

<p align="center"><img width="90%" src="https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp/raw/master/docs/data/supported_functionalities_samples.jpg" /></p>

## Documentation

You can build our documentation locally using the following code:

* Sphinx

  ```bash
  cd docs
  pip3 install -r .sphinx/requirements.txt
  python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
  ```

* Doxygen

  ```bash
  doxygen .Doxyfile
  ```

## Prerequisites

Refer to our
[Linux GPU and OS support](https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html)
page to see if your system is supported.

To use RPP, you must have installed the following:

* ROCm
  For ROCm installation instructions, see
  [Linux quick-start](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html).

* Clang Version `5.0.1` and above

  * Ubuntu `20`/`22`

    ```bash
    sudo apt-get install clang
    ```

  * CentOS `7`

    ```bash
    sudo yum install llvm-toolset-7-clang llvm-toolset-7-clang-analyzer llvm-toolset-7-clang-tools-extra
    scl enable llvm-toolset-7 bash
    ```

  * RHEL `8`/`9`

    ```bash
    sudo yum install clang
    ```

  * SLES `15-SP4` (use `ROCm LLVM Clang`)

    ```bash
    zypper -n --no-gpg-checks install clang
    update-alternatives --install /usr/bin/clang clang /opt/rocm-*/llvm/bin/clang 100
    update-alternatives --install /usr/bin/clang++ clang++ /opt/rocm-*/llvm/bin/clang++ 100
    ```

* CMake Version `3.5` and above

* IEEE 754-based half-precision floating-point library (half.hpp)

  * Use the `half` package with ROCm

    ```bash
    sudo apt-get install half
    ```

   Note that you must use the correct package management utility (`zypper`/`yum`).

  * Install from source

  ```bash
  wget https://sourceforge.net/projects/half/files/half/1.12.0/half-1.12.0.zip
  unzip half-1.12.0.zip -d half-files
  sudo mkdir /usr/local/include/half
  sudo cp half-files/include/half.hpp /usr/local/include/half
  ```

* Compiler with support for C++ Version `17` and above
* OpenMP
* Threads

### Test suite prerequisites

* OpenCV `3.4.0`/`4.5.5`

  * Install OpenCV prerequisites:

    ```bash
    sudo apt-get update
    sudo -S apt-get -y --allow-unauthenticated install build-essential libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy
    sudo -S apt-get -y --allow-unauthenticated install libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev unzip wget
    ```

  * Download OpenCV `3.4.0` /`4.5.5`:

    ```bash
    wget https://github.com/opencv/opencv/archive/3.4.0.zip
    unzip 3.4.0.zip
    cd opencv-3.4.0/
    ```

    OR

    ```bash
    wget https://github.com/opencv/opencv/archive/4.5.5.zip
    unzip 4.5.5.zip
    cd opencv-4.5.5/
    ```

  * Install OpenCV:

    ```bash
    mkdir build
    cd build
    cmake -D WITH_GTK=ON -D WITH_JPEG=ON -D BUILD_JPEG=ON -D WITH_OPENCL=OFF -D WITH_OPENCLAMDFFT=OFF -D WITH_OPENCLAMDBLAS=OFF -D WITH_VA_INTEL=OFF -D WITH_OPENCL_SVM=OFF -D CMAKE_INSTALL_PREFIX=/usr/local ..
    sudo -S make -j128 <Or other number of threads to use>
    sudo -S make install
    sudo -S ldconfig
    ```

* Install TurboJpeg:

  ```bash
  sudo apt-get install nasm
  sudo apt-get install wget
  git clone -b 2.0.6.1 https://github.com/rrawther/libjpeg-turbo.git
  cd libjpeg-turbo
  mkdir build
  cd build
  cmake -DCMAKE_INSTALL_PREFIX=/usr \
        -DCMAKE_BUILD_TYPE=RELEASE  \
        -DENABLE_STATIC=FALSE       \
        -DCMAKE_INSTALL_DOCDIR=/usr/share/doc/libjpeg-turbo-2.0.3 \
        -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib  \
        ..
  make -j$nproc
  sudo make install
  ```

## Build and install RPP

To build and install RPP, run the code shown for your backend:

* HIP (default)

  ```bash
  git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp.git
  mkdir build && cd build
  cmake -DBACKEND=HIP ../rpp
  make -j8
  sudo make install
  ```

* OpenCL

  ```bash
  git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp.git
  mkdir build && cd build
  cmake -DBACKEND=OCL ../rpp
  make -j8
  sudo make install
  ```

* CPU

  ```bash
  git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp.git
  mkdir build && cd build
  cmake -DBACKEND=CPU ../rpp
  make -j8
  sudo make install
  ```

## Test Functionalities

To test the functionalities of RPP, run the code shown for your backend:

* HIP

  ```bash
    cd rpp/utilities/rpp-unittests/HIP_NEW
    ./testAllScript.sh
  ```

* OpenCL

  ```bash
    cd rpp/utilities/rpp-unittests/OCL_NEW
    ./testAllScript.sh
  ```

  * CPU

  ```bash
    cd rpp/utilities/rpp-unittests/HOST_NEW
    ./testAllScript.sh
  ```

## MIVisionX support - OpenVX extension

[MIVisionX](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX) RPP extension
[vx_rpp](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/amd_openvx_extensions/amd_rpp#amd-rpp-extension) supports RPP functionality through the OpenVX Framework.

## Technical support

For RPP questions and feedback, you can contact us at `mivisionx.support@amd.com`.

To submit feature requests and bug reports, use our
[GitHub issues](https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp/issues) page.

## Release notes

All notable changes for each release are added to our [changelog](CHANGELOG.md).

## Tested configurations

* Linux distribution
  * Ubuntu - `20.04` / `22.04`
  * CentOS - `7`
  * RedHat - `8` / `9`
  * SLES - `15-SP4`
* ROCm: rocm-core - `5.7.0.50700-63`
* OpenCV - [4.6.0](https://github.com/opencv/opencv/releases/tag/4.6.0)
