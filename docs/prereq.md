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

* Boost Version `1.72` and above
  ```
  wget https://boostorg.jfrog.io/artifactory/main/release/1.72.0/source/boost_1_72_0.tar.gz
  tar -xzvf boost_1_72_0.tar.gz
  cd boost_1_72_0
  ./bootstrap.sh
  ./b2 install
  ```
  + **NOTE:** [Install from source](https://www.boost.org/doc/libs/1_72_0/more/getting_started/unix-variants.html#easy-build-and-install)

* IEEE 754-based half-precision floating-point library - half.hpp

  ```
  wget https://sourceforge.net/projects/half/files/half/1.12.0/half-1.12.0.zip
  unzip half-1.12.0.zip -d half-files
  sudo mkdir /usr/local/include/half
  sudo cp half-files/include/half.hpp /usr/local/include/half
  ```
