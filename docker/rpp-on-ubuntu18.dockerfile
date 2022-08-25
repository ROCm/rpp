FROM ubuntu:18.04

RUN apt-get update -y
# install ROCm
RUN apt-get -y install libnuma-dev wget sudo gnupg2 kmod &&  \
        wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add - && \
        echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list && \
        sudo apt-get update -y && \
        sudo apt-get -y install rocm-dev
# install rpp dependencies
RUN apt-get -y install gcc g++ clang cmake unzip libbz2-dev python3-dev git && \
        wget https://sourceforge.net/projects/half/files/half/1.12.0/half-1.12.0.zip && \
        unzip half-1.12.0.zip -d half-files && sudo mkdir /usr/local/include/half && sudo cp half-files/include/half.hpp /usr/local/include/half && \
        wget https://dl.bintray.com/boostorg/release/1.72.0/source/boost_1_72_0.tar.bz2 && tar xjvf boost_1_72_0.tar.bz2 && \
        cd boost_1_72_0 && ./bootstrap.sh --prefix=/usr/local --with-python=python3 && \
        ./b2 stage -j16 threading=multi link=shared && sudo ./b2 install threading=multi link=shared --with-system --with-filesystem && \
        sudo ./b2 install threading=multi link=static --with-system --with-filesystem