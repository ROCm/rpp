# AMD Radeon Performance Primitives (RPP) Test Suite
The Radeon Performance Primitives library provides a test suite, that are used to validate the functionality and performance of the AMD Radeon Performance Primitives vision library. It can be executed under 2 backend scenarios - (HOST/HIP):
- HOST backend - (On a CPU with HOST backend)
- HIP backend - (On a GPU with HIP backend)

## Features:
The suite includes:
* Unit tests that execute the desired functionality and variant once, report RPP execution wall time and save output images
* Performance tests that execute the desired functionality and variant 100 times by default, and report max/min/avg RPP execution wall time, or optionally, AMD rocprof kernel profiler max/min/avg time for HIP backend variants.
* Unit and Performance tests are included for three layouts - PLN1 (1 channel planar NCHW), PLN3 (3 channel planar NCHW) and PKD3 (3 channel packed/interrleaved NHWC).
* Unit and Performance tests are included for various input/output bitdepths including U8/F32/F16/I8.
* Support for pixelwise output referencing against golden outputs, and functionality validation checking, by tolerance-based pass/fail criterions for each variant. (Current support for U8 PKD3 variants)
* Support for TurboJPEG and OpenCV decoder for decoding input images

## Command Line Arguments
The test suite accepts the following command line arguments:
* input_path1: The path to the input data. Default is $cwd/../TEST_IMAGES/three_images_mixed_src1
* input_path2: The path to the input data. Default is $cwd/../TEST_IMAGES/three_images_mixed_src2
* case_start: The starting case number for the test range (0-38). Default is 0
* case_end: The ending case number for the test range (0-38). Default is 38
* test_type: The type of test to run (0 = Unit tests, 1 = Performance tests). Default is 0
* case_list: A list of specific case numbers to run. Must be used in conjunction with --test_type
* profiling: Run the tests with a profiler (YES/NO). Default is NO

## Running the Tests for HOST Backend
The test suite can be run with the following command:
python runTests.py --input_path1 <input_path1> --input_path2 <input_path2> --case_start <case_start> --case_end <case_end> --test_type <test_type>

## Running the Tests for HIP Backend
The test suite can be run with the following command:
python runTests.py --input_path1 <input_path1> --input_path2 <input_path2> --case_start <case_start> --case_end <case_end> --test_type <test_type> --profiling <profiling>

## Modes of operation
* QA mode
```
python runTests.py --case_start 0 --case_end 38 --test_type 0 --qa_mode 1
```
* Generic unittests
```
python runTests.py --case_start 0 --case_end 38 --test_type 0 --qa_mode 0
```
* Generic performance tests
```
python runTests.py --case_start 0 --case_end 38 --test_type 1
```

To run the unit tests / performance tests for specific case numbers. please case use case_list parameter. Example as below

* To run unittests for case numbers 0 2 4
```
python runTests.py --case_list 0 2 4 --test_type 0
```
* To run performance tests for case numbers 0, 2, 4
```
python runTests.py --case_list 0 2 4 --test_type 1
```

