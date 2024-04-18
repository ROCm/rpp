.. meta::
  :description: rocAL documentation and API reference library
  :keywords: rocAL, ROCm, API, documentation

.. _install:

********************************************************************
Installation
********************************************************************

This chapter provides information about the installation of RPP and related packages.  

Prerequisites
=============================

* Linux distribution

  - Ubuntu 20.04 or 22.04
  - CentOS 7
  - RedHat 8 or 9
  - SLES 15-SP4

* `ROCm supported hardware <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_

* Install ROCm with `amdgpu-install <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html>`_ with ``--usecase=graphics,rocm --no-32``

* Clang Version `5.0.1` and above

  * Ubuntu `20`/`22`

    .. code-block:: shell

      sudo apt-get install clang


  * CentOS `7`

    .. code-block:: shell

      sudo yum install llvm-toolset-7-clang llvm-toolset-7-clang-analyzer llvm-toolset-7-clang-tools-extra
      scl enable llvm-toolset-7 bash


  * RHEL `8`/`9`

    .. code-block:: shell

      sudo yum install clang


  * SLES `15-SP4` (use `ROCm LLVM Clang`)

    .. code-block:: shell

      zypper -n --no-gpg-checks install clang
      update-alternatives --install /usr/bin/clang clang /opt/rocm-*/llvm/bin/clang 100
      update-alternatives --install /usr/bin/clang++ clang++ /opt/rocm-*/llvm/bin/clang++ 100


* CMake Version `3.5` and above

* IEEE 754-based half-precision floating-point library (half.hpp)

  * `half` package install

    .. code-block:: shell

      sudo apt-get install half

    .. note::
        Use appropriate package manager depending on the OS 

* Compiler with support for C++ Version `17` and above

* OpenMP

* Threads

Build and install instructions
================================

The installation process uses the following steps: 

.. _package-install:

Package install
-------------------------------

Install RPP runtime, development, and test packages. 

* Runtime package - `rpp` only provides the rpp library `librpp.so`
* Development package - `rpp-dev`/`rpp-devel` provides the library, header files, and samples
* Test package - `rpp-test` provides ctest to verify installation

.. note::
  Package installation will auto install all dependencies.

On Ubuntu
^^^^^^^^^^^^^^^

.. code-block:: shell

  sudo apt install rpp rpp-dev rpp-test

On RHEL
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

  sudo yum install rpp rpp-devel rpp-test


On SLES
^^^^^^^^^^^^^^

.. code-block:: shell

  sudo zypper install rpp rpp-devel rpp-test


.. _source-install:

Source build and install
---------------------------

The process for installing is as follows:

* Clone RPP git repository

  .. code-block:: shell
    
    git clone https://github.com/ROCm/rpp.git

  .. note::
      RPP has support for two GPU backends: **OPENCL** and **HIP**

* Instructions for building RPP with the **HIP** GPU backend (default GPU backend):

  .. code-block:: shell

      mkdir build-hip
      cd build-hip
      cmake ../rpp
      make -j8
      sudo make install


  + Run tests - `test option instructions <https://github.com/ROCm/MIVisionX/wiki/CTest>`_

    .. code-block:: shell

        make test

    .. note::
        `make test` requires installation of `test suite prerequisites <../utilities/test_suite/README.md>`_

* Instructions for building RPP with **OPENCL** GPU backend

  .. code-block:: shell

      mkdir build-ocl
      cd build-ocl
      cmake -DBACKEND=OCL ../rpp
      make -j8
      sudo make install

Verify installation
=========================

The installer will copy

* Libraries into `/opt/rocm/lib`
* Header files into `/opt/rocm/include/rpp`
* Samples folder into `/opt/rocm/share/rpp`
* Documents folder into `/opt/rocm/share/doc/rpp`

.. note::
  Installation of `test suite prerequisites <../utilities/test_suite/README.md>`_ is required to run tests

Verify with `rpp-test` package
--------------------------------------------

Test package will install `ctest` module to test `rpp`. Use the following steps to test the installation:

.. code-block:: shell

  mkdir rpp-test && cd rpp-test
  cmake /opt/rocm/share/rpp/test/
  ctest -VV

Test RPP functionalities
--------------------------------------------

To test the functionalities of `rpp`, run the code shown for your backend:

* HIP

  .. code-block:: shell

      cd rpp/utilities/rpp-unittests/HIP_NEW
      ./testAllScript.sh


* OpenCL

  .. code-block:: shell

      cd rpp/utilities/rpp-unittests/OCL_NEW
      ./testAllScript.sh


* CPU

  .. code-block:: shell

      cd rpp/utilities/rpp-unittests/HOST_NEW
      ./testAllScript.sh



