FROM ubuntu:22.04

ARG ROCM_INSTALLER_REPO=https://repo.radeon.com/amdgpu-install/5.4.1/ubuntu/jammy/amdgpu-install_5.4.50401-1_all.deb
ARG ROCM_INSTALLER_PACKAGE=amdgpu-install_5.4.50401-1_all.deb

ENV RPP_DEPS_ROOT=/rpp-deps
WORKDIR $RPP_DEPS_ROOT

RUN apt-get update -y
# install rpp base dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install gcc g++ clang cmake git wget unzip libbz2-dev python3-dev libssl-dev libomp-dev bzip2
# install ROCm for rpp OpenCL/HIP dependency
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install initramfs-tools libnuma-dev wget sudo keyboard-configuration libstdc++-12-dev &&  \
        sudo apt-get -y clean && dpkg --add-architecture i386 && \
        wget ${ROCM_INSTALLER_REPO} && \
        sudo apt-get install -y ./${ROCM_INSTALLER_PACKAGE} && \
        sudo apt-get update -y && \
        sudo amdgpu-install -y --usecase=graphics,rocm
# install rpp dependencies - half.hpp & boost
RUN wget https://sourceforge.net/projects/half/files/half/1.12.0/half-1.12.0.zip && \
        unzip half-1.12.0.zip -d half-files && mkdir -p /usr/local/include/half && cp half-files/include/half.hpp /usr/local/include/half
RUN apt-get -y install sqlite3 libsqlite3-dev libtool build-essential && \
    wget https://boostorg.jfrog.io/artifactory/main/release/1.80.0/source/boost_1_80_0.tar.bz2 && tar xjvf boost_1_80_0.tar.bz2 && \
    cd boost_1_80_0 && ./bootstrap.sh --prefix=/usr/local --with-python=python3 && \
    ./b2 stage -j16 threading=multi link=shared cxxflags="-std=c++11" && \
    sudo ./b2 install threading=multi link=shared --with-system --with-filesystem && \
    ./b2 stage -j16 threading=multi link=static cxxflags="-std=c++11 -fpic" cflags="-fpic" && \
    sudo ./b2 install threading=multi link=static --with-system --with-filesystem

ENV RPP_WORKSPACE=/workspace
WORKDIR $RPP_WORKSPACE

# install RPP
RUN git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp.git && mkdir build && cd build && \
        cmake ../rpp && make -j8 && make install