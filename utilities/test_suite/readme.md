AMD Radeon Performance Primitives (RPP) Test Suite

The RPP Test Suite is a set of tests that are used to validate the functionality and performance of the AMD Radeon Performance Primitives vision library.

Features:
The suite includes:
* Unit tests that execute the desired functionality and variant once, report RPP execution wall timee and save output images.
* Performance tests that execute the desired functionality and variant 100 times by default, and report max/min/avg RPP execution wall time, or optionally, AMD rocprof kernel profiler max/min/avg time for HIP backend variants.
* Unit and Performance tests are included for three layouts - PLN1 (1 channel planar NCHW), PLN3 (3 channel planar NCHW) and PKD3 (3 channel packed/interrleaved NHWC).
* Unit and Performance tests are included for various input/output bitdepths including U8/F32/F16/I8.
* Support for pixelwise output referencing against golden outputs, and functionality validation checking, by tolerance-based pass/fail criterions for each variant. (Current support for U8 PKD3 variants)

Command Line Arguments

The test suite accepts the following command line arguments:
* input_path: The path to the input data. Default is $cwd/../TEST_IMAGES/three_images_mixed_src1
* output_path: The path to the output data. Default is $cwd/../TEST_IMAGES/three_images_mixed_src2
* case_start: The starting case number for the test range (0-86). Default is 0
* case_end: The ending case number for the test range (0-86). Default is 86
* test_type: The type of test to run (0 = Unit tests, 1 = Performance tests). Default is 0
* case_list: A list of specific case numbers to run. Must be used in conjunction with --test_type.
* profiling: Run the tests with a profiler (YES/NO). Default is NO

Running the Tests

The test suite can be run with the following command:
python runTests.py --input_path <input_path> --output_path <output_path> --case_start <case_start> --case_end <case_end> --test_type <test_type> --case_list <case_list> --profiling <profiling>
