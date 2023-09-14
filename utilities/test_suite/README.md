# AMD ROCm Performance Primitives (RPP) Test Suite

This repository contains two test suites for the AMD ROCm Performance Primitives (RPP) library: one for image processing and one for audio processing. These test suites can be used to validate the functionality and performance of the AMD ROCm Performance Primitives (RPP) vision and audio libraries.

## Rpp Image Test Suite

The image test suite can be executed under 2 backend scenarios - (HOST/HIP):
-   HOST backend - (On a CPU with HOST backend)
-   HIP backend - (On a GPU with HIP backend)

## Command Line Arguments (Rpp Image Test Suite)
The image test suite accepts the following command line arguments:
-   input_path1: The path to the input folder 1. Default is $cwd/../TEST_IMAGES/three_images_mixed_src1
-   input_path2: The path to the input folder 2. Default is $cwd/../TEST_IMAGES/three_images_mixed_src2
-   case_start: The starting case number for the test range (0-38). Default is 0
-   case_end: The ending case number for the test range (0-38). Default is 38
-   test_type: The type of test to run (0 = Unit tests, 1 = Performance tests). Default is 0
-   case_list: A list of specific case numbers to run. Must be used in conjunction with --test_type
-   profiling: Run the tests with a profiler (YES/NO). Default is NO. This option is only available with HIP backend
-   qa_mode: Output images from tests will be compared with golden outputs - (0 / 1). Default is 0
-   decoder_type: Type of Decoder to decode the input data - (0 = TurboJPEG / 1 = OpenCV). Default is 0
-   num_runs: Specifies the number of runs for running the performance tests
-   preserve_output: preserves the output images or performance logs generated from the previous test suite run - (0 = remove output images or performance logs / 1 = preserve output images or performance logs). Default is 1
-   batch_size: Specifies the batch size to use for running tests. Default is 1

## Running the Tests for HOST Backend (Rpp Image Test Suite)
The test suite can be run with the following command:
python runTests.py --input_path1 <input_path1> --input_path2 <input_path2> --case_start <case_start> --case_end <case_end> --test_type <test_type>

## Running the Tests for HIP Backend (Rpp Image Test Suite)
The test suite can be run with the following command:
python runTests.py --input_path1 <input_path1> --input_path2 <input_path2> --case_start <case_start> --case_end <case_end> --test_type <test_type> --profiling <profiling>

## Modes of operation (Rpp Image Test Suite)
-   QA mode - Tolerance based PASS/FAIL tests for RPP HIP/HOST functionalities checking pixelwise match between C/SSE/AVX/HIP versions after comparison to preset golden outputs. Please note that QA mode is only supported with a batch size of 3.
Note: QA mode is not supported for case 84 due to run-to-run variation of outputs.
``` python
python runTests.py --case_start 0 --case_end 38 --test_type 0 --qa_mode 1 --batch_size 3
```
-   Unit test mode - Unit tests allowing users to pass a path to a folder containing images, to execute the desired functionality and variant once, report RPP execution wall time, save and view output images
``` python
python runTests.py --case_start 0 --case_end 38 --test_type 0 --qa_mode 0
```
-   Performance test mode - Performance tests that execute the desired functionality and variant 100 times by default, and report max/min/avg RPP execution wall time, or optionally, AMD rocprof kernel profiler max/min/avg time for HIP backend variants.
``` python
python runTests.py --case_start 0 --case_end 38 --test_type 1
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

## Features (Rpp Image Test Suite)
The image test suite includes:
-   Unit tests that execute the desired functionality and variant once, report RPP execution wall time and save output images
-   Performance tests that execute the desired functionality and variant 100 times by default, and report max/min/avg RPP execution wall time, or optionally, AMD rocprof kernel profiler max/min/avg time for HIP backend variants.
-   Unit and Performance tests are included for three layouts - PLN1 (1 channel planar NCHW), PLN3 (3 channel planar NCHW) and PKD3 (3 channel packed/interrleaved NHWC).
-   Unit and Performance tests are included for various input/output bitdepths including U8/F32/F16/I8.
-   Support for pixelwise output referencing against golden outputs, and functionality validation checking, by tolerance-based pass/fail criterions for each variant. (Current support only for U8 variants)
-   Support for TurboJPEG and OpenCV decoder for decoding input images

## Rpp Audio Test Suite
The audio test suite can be executed to validate the functionality and performance of the AMD ROCm Performance Primitives (RPP) audio library.
-   HOST backend - (On a CPU with HOST backend)

## Command Line Arguments (Rpp Audio Test Suite)
The audio test suite accepts the following command line arguments:
-   input_path: The path to the input folder. Default is $cwd/../TEST_AUDIO_FILES/eight_samples_single_channel_src1
-   case_start: The starting case number for the test range (0-0). Default is 0
-   case_end: The ending case number for the test range (0-0). Default is 0
-   test_type: The type of test to run (0 = QA tests, 1 = Performance tests). Default is 0
-   case_list: A list of specific case numbers to run. Must be used in conjunction with --test_type
-   num_runs: Specifies the number of runs for running the performance tests
-   preserve_output: preserves the output or performance logs generated from the previous test suite run - (0 = remove output or performance logs / 1 = preserve output or performance logs). Default is 1
-   batch_size: Specifies the batch size to use for running tests. Default is 1

## Running the Tests for HOST Backend (Rpp Audio Test Suite)
The test suite can be run with the following command:
python runAudioTests.py --input_path <input_path> --case_start <case_start> --case_end <case_end> --test_type <test_type>

## Modes of operation (Rpp Audio Test Suite)
-   QA mode - Tolerance based PASS/FAIL tests for RPP AUDIO HOST functionalities checking match between output and preset golden outputs. Please note that QA mode is only supported with a batch size of 8.
``` python
python runAudioTests.py --case_start 0 --case_end 0 --test_type 0 --batch_size 8
```

-   Performance test mode - Performance tests that execute the desired functionality and variant 100 times by default, and report max/min/avg RPP execution wall time.
``` python
python runAudioTests.py --case_start 0 --case_end 0 --test_type 1
```

To run the QA tests / performance tests for specific case numbers. please case use case_list parameter. Example as below

-   To run QA tests for case numbers 0, 1, 2
``` python
python runTests.py --case_list 0 1 2 --test_type 0 --batch_size 8
```
-   To run performance tests for case numbers 0, 1, 2
``` python
python runTests.py --case_list 0 1 2 --test_type 1
```

## Features (Rpp Audio Test Suite)
The audio test suite includes:
-   Performance tests that execute the desired functionality and variant 100 times by default, and report max/min/avg RPP execution wall time.
-   QA and Performance tests are included for one input/output bitdepth F32.
-   Support for output referencing against golden outputs, and functionality validation checking, by tolerance-based pass/fail criterions for each variant.