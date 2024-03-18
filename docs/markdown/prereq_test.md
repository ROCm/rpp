# Prerequisites for Test Suite

* OpenCV `3.4.0`/`4.5.5` - **pre-requisites**
  ```
  sudo apt-get update
  sudo -S apt-get -y --allow-unauthenticated install build-essential libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy
  sudo -S apt-get -y --allow-unauthenticated install libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev unzip wget
  ```

* OpenCV `3.4.0` /`4.5.5` - **download**
  ```
  wget https://github.com/opencv/opencv/archive/3.4.0.zip
  unzip 3.4.0.zip
  cd opencv-3.4.0/
  ```
  OR
  ```
  wget https://github.com/opencv/opencv/archive/4.5.5.zip
  unzip 4.5.5.zip
  cd opencv-4.5.5/
  ```

* OpenCV `3.4.0`/`4.5.5` - **installation**
  ```
  mkdir build
  cd build
  cmake -D WITH_GTK=ON -D WITH_JPEG=ON -D BUILD_JPEG=ON -D WITH_OPENCL=OFF -D WITH_OPENCLAMDFFT=OFF -D WITH_OPENCLAMDBLAS=OFF -D WITH_VA_INTEL=OFF -D WITH_OPENCL_SVM=OFF -D CMAKE_INSTALL_PREFIX=/usr/local ..
  sudo -S make -j128 <Or other number of threads to use>
  sudo -S make install
  sudo -S ldconfig
  ```

* TurboJpeg installation
  ```
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
