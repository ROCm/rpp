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

  - Ubuntu 22.04 or 24.04
  - RedHat 8 or 9
  - SLES 15-SP5

* `ROCm supported hardware <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_

* Install ROCm with `amdgpu-install <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html>`_ with ``--usecase=rocm``

* CMake Version `3.10` and above

    .. code-block:: shell

      sudo apt-get install cmake

* AMD Clang++ Version 18.0.0 or later - installed with ROCm

  * NOTE: For CPU only backend use Clang Version `5.0.1` and above

      .. code-block:: shell

        sudo apt-get install clang

* IEEE 754-based half-precision floating-point library (half.hpp)

    .. code-block:: shell

      sudo apt-get install half

* Compiler with support for C++ Version `17` and above

    .. code-block:: shell

      sudo apt-get install libstdc++-12-dev

* OpenMP

    .. code-block:: shell

      sudo apt-get install libomp-dev

* Threads

    .. note::
        Use appropriate package manager depending on the OS 

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
        `make test` requires installation of `test suite prerequisites <https://github.com/ROCm/rpp/blob/develop/utilities/test_suite/README.md>`_

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

Verify with `rpp-test` package
--------------------------------------------

Test package will install `ctest` module to test `rpp`. Use the following steps to test the installation:

.. code-block:: shell

  mkdir rpp-test && cd rpp-test
  cmake /opt/rocm/share/rpp/test/
  ctest -VV

.. note::
  Installation of `test suite prerequisites <https://github.com/ROCm/rpp/blob/develop/utilities/test_suite/README.md>`_ are required to run tests

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
