FROM ubuntu:22.04

ENV RPP_DEPS_ROOT=/opt/rpp-deps
WORKDIR $RPP_DEPS_ROOT

RUN apt-get update -y
# install rpp base dependencies - Level 1
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install gcc g++ clang cmake git wget unzip libbz2-dev python3-dev git && \
        wget https://sourceforge.net/projects/half/files/half/1.12.0/half-1.12.0.zip && \
        unzip half-1.12.0.zip -d half-files && mkdir -p /usr/local/include/half && cp half-files/include/half.hpp /usr/local/include/half && \
        wget https://boostorg.jfrog.io/artifactory/main/release/1.72.0/source/boost_1_72_0.tar.gz && tar -xzvf boost_1_72_0.tar.gz && \
        cd boost_1_72_0 && ./bootstrap.sh --prefix=/usr/local --with-python=python3 && \
        ./b2 stage -j16 threading=multi link=shared && ./b2 install threading=multi link=shared --with-system --with-filesystem && \
        ./b2 install threading=multi link=static --with-system --with-filesystem
# install ROCm for rpp OpenCL/HIP dependency - Level 2
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install initramfs-tools libnuma-dev wget sudo keyboard-configuration &&  \
        wget https://repo.radeon.com/amdgpu-install/5.3/ubuntu/jammy/amdgpu-install_5.3.50300-1_all.deb && \
        sudo apt-get install -y ./amdgpu-install_5.3.50300-1_all.deb && \
        sudo apt-get update -y && \
        sudo amdgpu-install -y --usecase=graphics,rocm

WORKDIR /workspace

# install RPP
RUN git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp.git && mkdir build && cd build && \
        cmake ../rpp && make -j8 && make install