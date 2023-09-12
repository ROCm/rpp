# AMD ROCm Performance Primitives (RPP) Test Suite
The AMD ROCm Performance Primitives (RPP) audio test suite can be used to validate the functionality and performance of the AMD ROCm Performance Primitives (RPP) audio augmentations.
-   HOST backend - (On a CPU with HOST backend)

## Command Line Arguments
The test suite accepts the following command line arguments:
-   input_path: The path to the input folder. Default is $cwd/../TEST_AUDIO_FILES/eight_samples_single_channel_src1
-   case_start: The starting case number for the test range (0-0). Default is 0
-   case_end: The ending case number for the test range (0-0). Default is 0
-   test_type: The type of test to run (0 = Unit tests, 1 = Performance tests). Default is 0
-   case_list: A list of specific case numbers to run. Must be used in conjunction with --test_type
-   qa_mode: Outputs of audio augmentations from tests will be compared with golden outputs - (0 / 1). Default is 0
-   num_runs: Specifies the number of runs for running the performance tests
-   preserve_output: preserves the output or performance logs generated from the previous test suite run - (0 = remove output or performance logs / 1 = preserve output or performance logs). Default is 1
-   batch_size: Specifies the batch size to use for running tests. Default is 1

## Running the Tests for HOST Backend
The test suite can be run with the following command:
python runAudioTests.py --input_path <input_path> --case_start <case_start> --case_end <case_end> --test_type <test_type>

## Modes of operation
-   QA mode - Tolerance based PASS/FAIL tests for RPP AUDIO HOST functionalities checking match between output and preset golden outputs. Please note that QA mode is only supported with a batch size of 8.
``` python
python runAudioTests.py --case_start 0 --case_end 0 --test_type 0 --qa_mode 1 --batch_size 8
```
-   Unit test mode - Unit tests allowing users to pass a path to a folder containing audio files, to execute the desired functionality and variant once, report RPP execution wall time, save and view output
``` python
python runAudioTests.py --case_start 0 --case_end 0 --test_type 0 --qa_mode 0
```
-   Performance test mode - Performance tests that execute the desired functionality and variant 100 times by default, and report max/min/avg RPP execution wall time.
``` python
python runAudioTests.py --case_start 0 --case_end 0 --test_type 1
```

To run the unit tests / performance tests for specific case numbers. please case use case_list parameter. Example as below

-   To run unittests for case numbers 0, 1, 2
``` python
python runTests.py --case_list 0 1 2 --test_type 0
```
-   To run performance tests for case numbers 0, 1, 1
``` python
python runTests.py --case_list 0 1 2 --test_type 1
```

## Features
The suite includes:
-   Unit tests that execute the desired functionality and variant once, report RPP execution wall time and save output
-   Performance tests that execute the desired functionality and variant 100 times by default, and report max/min/avg RPP execution wall time.
-   Unit and Performance tests are included for one input/output bitdepth F32.
-   Support for output referencing against golden outputs, and functionality validation checking, by tolerance-based pass/fail criterions for each variant.
