# Prerequisites

* **OS**
  + Ubuntu `20.04`/`22.04`
  + CentOS `7`/`8`
  + RHEL `8`/`9`
  + SLES - `15-SP3`

* [ROCm supported hardware](https://docs.amd.com/bundle/Hardware_and_Software_Reference_Guide/page/Hardware_and_Software_Support.html)

* [ROCm](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.4.3/page/How_to_Install_ROCm.html) `5.4.3` and above

* Clang Version `5.0.1` and above

  + Ubuntu `20`/`22`
    ```
    sudo apt-get install clang
    ```

  + CentOS `7`
    ```
    sudo yum install llvm-toolset-7-clang llvm-toolset-7-clang-analyzer llvm-toolset-7-clang-tools-extra
    scl enable llvm-toolset-7 bash
    ```

  + CentOS `8` and RHEL `8`/`9`
    ```
    sudo yum install clang
    ```

  + SLES `15-SP3`
    ```
    sudo zypper install llvm-clang
    ```

* CMake Version `3.5` and above

* IEEE 754-based half-precision floating-point library - half.hpp

  ```
  wget https://sourceforge.net/projects/half/files/half/1.12.0/half-1.12.0.zip
  unzip half-1.12.0.zip -d half-files
  sudo mkdir /usr/local/include/half
  sudo cp half-files/include/half.hpp /usr/local/include/half
  ```
