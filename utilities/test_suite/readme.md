RPP Test Suite

The RPP Test Suite is a set of tests that are used to validate the functionality and performance of the RPP application. The suite includes both unit tests and performance tests, and allows the user to test specific cases or a range of cases. This test suite includes  all three layouts .The test suite also provides an option to run the tests with a profiler for performance analysis. This test suite also contains validation part to validate the functionalities, currently the validation part is only restricted to input depth 0 and outputToggle value 0

Command Line Arguments

The test suite accepts the following command line arguments:
--input_path: The path to the input data. Default is $cwd/../TEST_IMAGES/three_images_mixed_src1
--output_path: The path to the output data. Default is $cwd/../TEST_IMAGES/three_images_mixed_src2
--case_start: The starting case number for the test range (0-86). Default is 0
--case_end: The ending case number for the test range (0-86). Default is 86
--test_type: The type of test to run (0 = Unit tests, 1 = Performance tests). Default is 0
--case_list: A list of specific case numbers to run. Must be used in conjunction with --test_type.
--profiling: Run the tests with a profiler (YES/NO). Default is NO

Running the Tests

The test suite can be run with the following command:
python main.py --input_path <input_path> --output_path <output_path> --case_start <case_start> --case_end <case_end> --test_type <test_type> --case_list <case_list> --profiling <profiling>