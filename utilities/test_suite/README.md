# AMD ROCm Performance Primitives (RPP) Test Suite

This repository contains three test suites for the AMD ROCm Performance Primitives (RPP) library: for image/voxel/audio processing. These test suites can be used to validate the functionality and performance of the AMD ROCm Performance Primitives (RPP) image/voxel/audio libraries.

## Test suite prerequisites

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
* Libsndfile installation
  ```
  sudo apt-get update
  sudo apt-get install libsndfile1-dev
  ```

* Imagemagick
  ```
  sudo apt-get install imagemagick
  ```

* Nifti-Imaging nifti_clib
  ```
  git clone git@github.com:NIFTI-Imaging/nifti_clib.git
  cd nifti_clib
  mkdir build
  cd build
  cmake ..
  sudo make -j$nproc install
  ```

* Openpyxl
  ```
  pip install openpyxl
  ```

## Rpp Image Test Suite
The image test suite can be executed under 2 backend scenarios - (HOST/HIP):
-   HOST backend - (On a CPU with HOST backend)
-   HIP backend - (On a GPU with HIP backend)

### Command Line Arguments (RPP Image Test Suite)
The image test suite accepts the following command line arguments:
-   input_path1: The path to the input folder 1. Default is $cwd/../TEST_IMAGES/three_images_mixed_src1
-   input_path2: The path to the input folder 2. Default is $cwd/../TEST_IMAGES/three_images_mixed_src2
-   case_start: The starting case number for the test range (0-89). Default is 0
-   case_end: The ending case number for the test range (0-89). Default is 89
-   test_type: The type of test to run (0 = Unit tests, 1 = Performance tests). Default is 0
-   case_list: A list of specific case numbers to run. Must be used in conjunction with --test_type
-   profiling: Run the tests with a profiler (YES/NO). Default is NO. This option is only available with HIP backend
-   qa_mode: Output images from tests will be compared with golden outputs - (0 / 1). Default is 0
-   decoder_type: Type of Decoder to decode the input data - (0 = TurboJPEG / 1 = OpenCV). Default is 0
-   num_runs: Specifies the number of runs for running the performance tests
-   preserve_output: preserves the output images or performance logs generated from the previous test suite run - (0 = remove output images or performance logs / 1 = preserve output images or performance logs). Default is 1
-   batch_size: Specifies the batch size to use for running tests. Default is 1

### Running the Tests for HOST Backend (RPP Image Test Suite)
The test suite can be run with the following command:
``` python
python runTests.py --input_path1 <input_path1> --input_path2 <input_path2> --case_start <case_start> --case_end <case_end> --test_type <test_type>
```

### Running the Tests for HIP Backend (RPP Image Test Suite)
The test suite can be run with the following command:
``` python
python runTests.py --input_path1 <input_path1> --input_path2 <input_path2> --case_start <case_start> --case_end <case_end> --test_type <test_type> --profiling <profiling>
```

## Modes of operation (Rpp Image Test Suite)
-   QA mode (Unit tests) - Tolerance based PASS/FAIL tests for RPP HIP/HOST functionalities checking pixelwise match between C/SSE/AVX/HIP versions after comparison to preset golden outputs. Please note that QA mode is only supported with a batch size of 3.
Note: QA mode is not supported for case 84 due to run-to-run variation of outputs.
``` python
python runTests.py --case_start 0 --case_end 89 --test_type 0 --qa_mode 1 --batch_size 3
```
-   QA mode (Performance tests) - Tolerance based PASS/FAIL tests for RPP HIP/HOST functionalities checking achieved improvement in performance percentage over BatchPD versions after comparison to a threshold percentage of improvement
``` python
python runTests.py --case_list 21 36 63 --test_type 1 --qa_mode 1 --batch_size 8 --num_runs 100
```
-   QA mode (Performance tests) - Tolerance based PASS/FAIL tests for RPP HIP/HOST functionalities checking achieved improvement in performance percentage over BatchPD versions after comparison to a threshold percentage of improvement
``` python
python runTests.py --case_list 21 36 63 --test_type 1 --qa_mode 1 --batch_size 8 --num_runs 100
```
-   Unit test mode - Unit tests allowing users to pass a path to a folder containing images, to execute the desired functionality and variant once, report RPP execution wall time, save and view output images
Note: For testcase 82(RICAP) Please use images of same resolution and Batchsize > 1
      RICAP dataset path: rpp/utilities/test_suite/TEST_IMAGES/three_images_150x150_src1
``` python
python runTests.py --case_start 0 --case_end 89 --test_type 0 --qa_mode 0
```
-   Performance test mode - Performance tests that execute the desired functionality and variant 100 times by default, and report max/min/avg RPP execution wall time, or optionally, AMD rocprof kernel profiler max/min/avg time for HIP backend variants.
Note: For testcase 82(RICAP) Please use images of same resolution and Batchsize > 1
      RICAP dataset path: rpp/utilities/test_suite/TEST_IMAGES/three_images_150x150_src1
``` python
python runTests.py --case_start 0 --case_end 89 --test_type 1
```

To run the unit tests / performance tests for specific case numbers. please case use case_list parameter. Example as below

-   To run unittests for case numbers 0, 2, 4
``` python
python runTests.py --case_list 0 2 4 --test_type 0
```
-   To run performance tests for case numbers 0, 2, 4
``` python
python runTests.py --case_list 0 2 4 --test_type 1
```

To run performance tests with AMD rocprof kernel profiler for HIP backend variants. This will generate profiler times for HIP backend variants
``` python
python runTests.py --test_type 1 --profiling YES
```

### Summary of features (RPP Image Test Suite)
The image test suite includes:
-   Unit tests that execute the desired functionality and variant once, report RPP execution wall time and save output images
-   Performance tests that execute the desired functionality and variant 100 times by default, and report max/min/avg RPP execution wall time, or optionally, AMD rocprof kernel profiler max/min/avg time for HIP backend variants.
-   Unit and Performance tests are included for three layouts - PLN1 (1 channel planar NCHW), PLN3 (3 channel planar NCHW) and PKD3 (3 channel packed/interrleaved NHWC).
-   Unit and Performance tests are included for various input/output bitdepths including U8/F32/F16/I8.
-   Support for pixelwise output referencing against golden outputs, and functionality validation checking, by tolerance-based pass/fail criterions for each variant. (Current support only for U8 variants)
-   Support for TurboJPEG and OpenCV decoder for decoding input images

## RPP Voxel Test Suite
The 3D Voxel test suite can be executed under 2 backend scenarios - (HOST/HIP):
-   HOST backend - (On a CPU with HOST backend)
-   HIP backend - (On a GPU with HIP backend)

### Command Line Arguments (RPP Voxel Test Suite)
The voxel test suite accepts the following command line arguments:
-   header_path: Path to the nii header
-   data_path: Path to the nii data file
-   case_start: The starting case number for the test range (0-38). Default is 0
-   case_end: The ending case number for the test range (0-38). Default is 4
-   test_type: The type of test to run (0 = Unit tests, 1 = Performance tests). Default is 0
-   case_list: A list of specific case numbers to run. Must be used in conjunction with --test_type
-   profiling: Run the tests with a profiler (YES/NO). Default is NO. This option is only available with HIP backend
-   qa_mode: Output images from tests will be compared with golden outputs - (0 / 1). Default is 0
-   num_runs: Specifies the number of runs for running the performance tests
-   preserve_output: preserves the output images or performance logs generated from the previous test suite run - (0 = remove output images or performance logs / 1 = preserve output images or performance logs). Default is 1
-   batch_size: Specifies the batch size to use for running tests. Default is 1

### Running the Tests for HOST Backend (RPP Voxel Test Suite)
The test suite can be run with the following command:
``` python
python runTests_voxel.py --header_path <header_path> --data_path <data_path> --case_start <case_start> --case_end <case_end> --test_type <test_type>
```

### Running the Tests for HIP Backend (RPP Voxel Test Suite)
The test suite can be run with the following command:
``` python
python runTests_voxel.py --header_path <header_path> --data_path <data_path> --case_start <case_start> --case_end <case_end> --test_type <test_type> --profiling <profiling>
```

### Modes of operation (RPP Voxel Test Suite)
-   QA mode - Tolerance based PASS/FAIL tests for RPP HIP/HOST functionalities checking pixelwise match between C/SSE/AVX/HIP versions after comparison to preset golden outputs.
``` python
python runTests_voxel.py --case_start 0 --case_end 4 --test_type 0 --qa_mode 1 --batch_size 3
```
-   Unit test mode - Unit tests allowing users to pass a path to a folder containing nii fikes, to execute the desired functionality and variant once, report RPP execution wall time, save and view output images, gifs and nii files
``` python
python runTests_voxel.py --case_start 0 --case_end 4 --test_type 0 --qa_mode 0
```
-   Performance test mode - Performance tests that execute the desired functionality and variant 100 times by default, and report max/min/avg RPP execution wall time, or optionally, AMD rocprof kernel profiler max/min/avg time for HIP backend variants.
``` python
python runTests_voxel.py --case_start 0 --case_end 4 --test_type 1
```

To run the unit tests / performance tests for specific case numbers. please case use case_list parameter. Example as below

-   To run unittests for case numbers 0, 2, 4
``` python
python runTests_voxel.py --case_list 0 2 4 --test_type 0
```
-   To run performance tests for case numbers 0, 2, 4
``` python
python runTests_voxel.py --case_list 0 2 4 --test_type 1
```

To run performance tests with AMD rocprof kernel profiler for HIP backend variants. This will generate profiler times for HIP backend variants
``` python
python runTests_voxel.py --test_type 1 --profiling YES
```

### Summary of features (RPP Voxel Test Suite)
The image test suite includes:
-   Unit tests that execute the desired functionality and variant once, report RPP execution wall time and save output images, gifs and nii files
-   Performance tests that execute the desired functionality and variant 100 times by default, and report max/min/avg RPP execution wall time, or optionally, AMD rocprof kernel profiler max/min/avg time for HIP backend variants.
-   Unit and Performance tests are included for three layouts - PLN1 (1 channel planar NCHW), PLN3 (3 channel planar NCHW) and PKD3 (3 channel packed/interrleaved NHWC).
-   Unit and Performance tests are included for one input/output bitdepth F32.
-   Support for pixelwise output referencing against golden outputs, and functionality validation checking, by tolerance-based pass/fail criterions for each variant.

### References (RPP Voxel Test Suite)
RPP test suite uses a sample ".nii" image usage from the BraTS2020 Dataset at https://www.kaggle.com/code/rastislav/3d-mri-brain-tumor-segmentation-u-net/input as per the following Data Usage Agreement present at the dataset link above.

Data Usage Agreement / Citations:
You are free to use and/or refer to the BraTS datasets in your own research, provided that you always cite the following three manuscripts:

-   B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694
-   S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117
-   S. Bakas, M. Reyes, A. Jakab, S. Bauer, M. Rempfler, A. Crimi, et al., "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge", arXiv preprint arXiv:1811.02629 (2018)

## RPP Audio Test Suite
The audio test suite can be executed to validate the functionality and performance of the AMD ROCm Performance Primitives (RPP) audio library.
-   HOST backend - (On a CPU with HOST backend)
-   F32 Bit Depth

### Command Line Arguments (RPP Audio Test Suite)
The audio test suite accepts the following command line arguments:
-   input_path: The path to the input folder. Default is $cwd/../TEST_AUDIO_FILES/eight_samples_single_channel_src1
-   case_start: The starting case number for the test range (0-0). Default is 0
-   case_end: The ending case number for the test range (0-0). Default is 0
-   test_type: The type of test to run (0 = QA tests, 1 = Performance tests). Default is 0
-   qa_mode: Output audio data from tests will be compared with golden outputs - (0 / 1). Default is 0
-   case_list: A list of specific case numbers to run. Must be used in conjunction with --test_type
-   num_runs: Specifies the number of runs for running the performance tests
-   preserve_output: preserves the output or performance logs generated from the previous test suite run - (0 = remove output or performance logs / 1 = preserve output or performance logs). Default is 1
-   batch_size: Specifies the batch size to use for running tests. Default is 1

### Running the Tests for HOST Backend (RPP Audio Test Suite)
The test suite can be run with the following command:
``` python
python runAudioTests.py --input_path <input_path> --case_start <case_start> --case_end <case_end> --test_type <test_type>
```

### Modes of operation (RPP Audio Test Suite)
-   QA mode - Tolerance based PASS/FAIL tests for RPP AUDIO HOST functionalities checking match between output and preset golden outputs. Please note that QA mode is only supported with a batch size of 3.
``` python
python runAudioTests.py --case_start 0 --case_end 0 --qa_mode 1 --batch_size 3
```

-   Performance test mode - Performance tests that execute the desired functionality and variant 100 times by default, and report max/min/avg RPP execution wall time.
``` python
python runAudioTests.py --case_start 0 --case_end 0 --test_type 1
```

To run the QA tests / performance tests for specific case numbers. please case use case_list parameter. Example as below

-   To run QA tests for case numbers 0, 1, 2
``` python
python runTests.py --case_list 0 1 2 --qa_mode 1 --batch_size 3
```
-   To run performance tests for case numbers 0, 1, 2
``` python
python runTests.py --case_list 0 1 2 --test_type 1
```

### Summary of features (RPP Audio Test Suite)
The audio test suite includes:
-   Performance tests that execute the desired functionality and variant 100 times by default, and report max/min/avg RPP execution wall time.
-   QA and Performance tests are included for one input/output bitdepth F32.
-   Support for output referencing against golden outputs, and functionality validation checking, by tolerance-based pass/fail criterions for each variant.
