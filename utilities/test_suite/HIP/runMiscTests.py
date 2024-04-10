"""
MIT License

Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import subprocess  # nosec
import argparse
import sys
import datetime
import shutil

# Set the timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
scriptPath = os.path.dirname(os.path.realpath(__file__))
outFolderPath = os.getcwd()
buildFolderPath = os.getcwd()
caseMin = 2
caseMax = 2

# Get a list of log files based on a flag for preserving output
def get_log_file_list():
    return [
        outFolderPath + "/OUTPUT_PERFORMANCE_MISC_LOGS_HIP_" + timestamp + "/Tensor_misc_hip_raw_performance_log.txt",
    ]

def run_unit_test(numDims, case, numRuns, testType, toggle, bitDepth, batchSize, outFilePath):
    print("\n\n\n\n")
    print("--------------------------------")
    print("Running a New Functionality...")
    print("--------------------------------")
    print(f"./Tensor_misc_hip {case} {testType} {toggle} {numDims} {batchSize} {numRuns}")
    result = subprocess.run([buildFolderPath + "/build/Tensor_misc_hip", str(case), str(testType), str(toggle), str(numDims), str(batchSize), str(numRuns), outFilePath, scriptPath], stdout=subprocess.PIPE)    # nosec
    print(result.stdout.decode())

    print("------------------------------------------------------------------------------------------")

def run_performance_test(loggingFolder, numDims, case, numRuns, testType, toggle, bitDepth, batchSize, outFilePath):
    print("\n\n\n\n")
    print("--------------------------------")
    print("Running a New Functionality...")
    print("--------------------------------")
    with open("{}/Tensor_misc_hip_raw_performance_log.txt".format(loggingFolder), "a") as log_file:
        print(f"./Tensor_misc_hip {case} {testType} {toggle} {numDims} {batchSize} {numRuns}")
        process = subprocess.Popen([buildFolderPath + "/build/Tensor_misc_hip", str(case), str(testType), str(toggle), str(numDims), str(batchSize), str(numRuns), outFilePath, scriptPath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)    # nosec
        while True:
            output = process.stdout.readline()
            if not output and process.poll() is not None:
                break
            print(output.strip())
            log_file.write(output)
    print("------------------------------------------------------------------------------------------")

def run_performance_test_with_profiler(loggingFolder, numDims, case, numRuns, testType, toggle, bitDepth, batchSize, outFilePath):
    print("\n\n\n\n")
    print("--------------------------------")
    print("Running a New Functionality...")
    print("--------------------------------")
    print(loggingFolder)

    if not os.path.exists(f"{outFilePath}/case_{case}"):
        os.mkdir(f"{outFilePath}/case_{case}")

    with open("{}/Tensor_misc_hip_raw_performance_log.txt".format(loggingFolder), "a") as log_file:
        print(f"\nrocprof --basenames on --timestamp on --stats -o {outFilePath}/case_{case}/output_case{case}.csv ./Tensor_misc_hip {case} {testType} {toggle} {numDims} {batchSize} {numRuns}")
        process = subprocess.Popen([ 'rocprof', '--basenames', 'on', '--timestamp', 'on', '--stats', '-o', f"{outFilePath}/case_{case}/output_case{case}.csv", "./Tensor_misc_hip", str(case), str(testType), str(toggle), str(numDims), str(batchSize), str(numRuns), outFilePath, scriptPath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)    # nosec
        while True:
            output = process.stdout.readline()
            if not output and process.poll() is not None:
                break
            print(output.strip())
            log_file.write(output)
    print("------------------------------------------------------------------------------------------")

# Parse and validate command-line arguments for the RPP test suite
def rpp_test_suite_parser_and_validator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_start", type = int, default = caseMin, help = "Testing start case # - Range must be in [" + str(caseMin) + ":" + str(caseMax) + "]")
    parser.add_argument("--case_end", type = int, default = caseMax, help = "Testing start case # - Range must be in [" + str(caseMin) + ":" + str(caseMax) + "]")
    parser.add_argument('--test_type', type = int, default = 0, help = "Type of Test - (0 = QA tests / 1 = Performance tests)")
    parser.add_argument('--toggle', type = int, default = 0, help = "Toggle outputs")
    parser.add_argument('--case_list', nargs = "+", help = "List of case numbers to test", required = False)
    parser.add_argument("--num_dims", type = int, default = 2, help = "Number of dimensions for input")
    parser.add_argument('--num_runs', type = int, default = 1, help = "Specifies the number of runs for running the performance tests")
    parser.add_argument('--qa_mode', type = int, default = 0, help = "Run with qa_mode? Outputs from tests will be compared with golden outputs - (0 / 1)", required = False)
    parser.add_argument('--profiling', type = str , default = 'NO', help = 'Run with profiler? - (YES/NO)', required = False)
    parser.add_argument('--batch_size', type = int, default = 1, help = "Specifies the batch size to use for running tests. Default is 1.")
    parser.add_argument('--preserve_output', type = int, default = 1, help = "preserves the output of the program - (0 = override output / 1 = preserve output )" )
    args = parser.parse_args()

    # validate the parameters passed by user
    if ((args.case_start < caseMin or args.case_start > caseMax) or (args.case_end < caseMin or args.case_end > caseMax)):
        print("Starting case# and Ending case# must be in the 0:1 range. Aborting!")
        exit(0)
    elif args.case_end < args.case_start:
        print("Ending case# must be greater than starting case#. Aborting!")
        exit(0)
    elif args.test_type < 0 or args.test_type > 1:
        print("Test Type# must be in the 0 / 1. Aborting!")
        exit(0)
    elif args.case_list is not None and args.case_start > caseMin and args.case_end < caseMax:
        print("Invalid input! Please provide only 1 option between case_list, case_start and case_end")
        exit(0)
    elif args.num_runs <= 0:
        print("Number of Runs must be greater than 0. Aborting!")
        exit(0)
    elif args.batch_size <= 0:
        print("Batch size must be greater than 0. Aborting!")
        exit(0)
    elif args.profiling != 'YES' and args.profiling != 'NO':
        print("Profiling option value must be either 'YES' or 'NO'.")
        exit(0)

    if args.case_list is None:
        args.case_list = range(args.case_start, args.case_end + 1)
        args.case_list = [str(x) for x in args.case_list]
    else:
        for case in args.case_list:
            if int(case) < caseMin or int(case) > caseMax:
                print("The case# must be in [" + str(caseMin) + ":" + str(caseMax) + "]")
                exit(0)
    return args

args = rpp_test_suite_parser_and_validator()
caseStart = args.case_start
caseEnd = args.case_end
testType = args.test_type
toggle = args.toggle
caseList = args.case_list
numDims = args.num_dims
numRuns = args.num_runs
profilingOption = args.profiling
batchSize = args.batch_size
qaMode = args.qa_mode
if qaMode:
    testType = 0
preserveOutput = args.preserve_output
bitDepth = 2 # Current audio test suite only supports bit depth 2
outFilePath = " "

if testType == 0 and batchSize != 3:
    print("QA mode can only run with a batch size of 3.")
    exit(0)
if preserveOutput == 0:
    validate_and_remove_folders(outFolderPath, "QA_RESULTS_MISC_HIP")
    validate_and_remove_folders(outFolderPath, "OUTPUT_PERFORMANCE_MISC_LOGS_HIP")

if(testType == 0):
    outFilePath = outFolderPath + '/QA_RESULTS_MISC_HIP_' + timestamp
    numRuns = 1
elif(testType == 1):
    if "--num_runs" not in sys.argv:
        numRuns = 100   #default numRuns for running performance tests
    outFilePath = outFolderPath + '/OUTPUT_PERFORMANCE_MISC_LOGS_HIP_' + timestamp
else:
    print("Invalid TEST_TYPE specified. TEST_TYPE should be 0/1 (0 = QA tests / 1 = Performance tests)")
    exit(0)

os.mkdir(outFilePath)
loggingFolder = outFilePath
dstPath = outFilePath

# Enable extglob
if os.path.exists(buildFolderPath + "/build"):
    shutil.rmtree(buildFolderPath + "/build")
os.makedirs(buildFolderPath + "/build")
os.chdir(buildFolderPath + "/build")

# Run cmake and make commands
subprocess.run(["cmake", scriptPath], cwd=".")   # nosec
subprocess.run(["make", "-j16"], cwd=".")    # nosec

supportedCaseList = ['2']
if testType == 0:
    for case in caseList:
        if case not in supportedCaseList:
            continue
        if toggle == 1:
            print("Only Toggle variant is QA tested for test Type 0. Aborting!")
            exit(0)

        run_unit_test(numDims, case, numRuns, testType, toggle, bitDepth, batchSize, outFilePath)
elif (testType == 1 and profilingOption == "NO"):
    for case in caseList:
        if case not in supportedCaseList:
            continue

        run_performance_test(loggingFolder, numDims, case, numRuns, testType, toggle, bitDepth, batchSize, outFilePath)
elif (testType == 1 and profilingOption == "YES"):
    for case in caseList:
        if case not in supportedCaseList:
            continue

    run_performance_test_with_profiler(loggingFolder, numDims, case, numRuns, testType, toggle, bitDepth, batchSize, outFilePath)

    RESULTS_DIR = outFolderPath + "/OUTPUT_PERFORMANCE_MISC_LOGS_HIP_" + timestamp
    print("RESULTS_DIR = " + RESULTS_DIR)
    CONSOLIDATED_FILE = RESULTS_DIR + "/consolidated_results.stats.csv"

    CASE_NUM_LIST = caseList
    BIT_DEPTH_LIST = [2]
    OFT_LIST = [0]

    # Open csv file
    new_file = open(CONSOLIDATED_FILE, 'w')
    new_file.write('"HIP Kernel Name","Calls","TotalDurationNs","AverageNs","Percentage"\n')

    # Loop through cases
    for CASE_NUM in CASE_NUM_LIST:
        # Set results directory
        CASE_RESULTS_DIR = RESULTS_DIR + "/case_" + str(CASE_NUM)
        print("CASE_RESULTS_DIR = " + CASE_RESULTS_DIR)

        # Loop through bit depths
        for BIT_DEPTH in BIT_DEPTH_LIST:
            # Loop through output format toggle cases
            for OFT in OFT_LIST:
                # Write into csv file
                CASE_FILE_PATH = CASE_RESULTS_DIR + "/output_case" + str(CASE_NUM) + ".stats.csv"
                print("CASE_FILE_PATH = " + CASE_FILE_PATH)
                fileCheck = case_file_check(CASE_FILE_PATH, new_file)
                if fileCheck == False:
                    continue

    new_file.close()
    subprocess.call(['chown', '{}:{}'.format(os.getuid(), os.getgid()), CONSOLIDATED_FILE])  # nosec
    try:
        generate_performance_reports(RESULTS_DIR)
    except ImportError:
        print("\nPandas not available! Results of GPU profiling experiment are available in the following files:\n" + \
                CONSOLIDATED_FILE + "\n")
    except IOError:
        print("Unable to open results in " + CONSOLIDATED_FILE)


# print the results of qa tests
nonQACaseList = []
supportedCases = 0
for num in caseList:
    if num in supportedCaseList:
        supportedCases += 1
caseInfo = "Tests are run for " + str(supportedCases) + " supported cases out of the " + str(len(caseList)) + " cases requested"
if qaMode and testType == 0:
    qaFilePath = os.path.join(outFilePath, "QA_results.txt")
    checkFile = os.path.isfile(qaFilePath)
    if checkFile:
        print("---------------------------------- Results of QA Test - Tensor_misc_hip ----------------------------------\n")
        print_qa_tests_summary(qaFilePath, supportedCaseList, nonQACaseList)

# Performance tests
if (testType == 1 and profilingOption == "NO"):
    log_file_list = get_log_file_list()

    functionality_group_list = [
        "arithmetic_operations",
    ]

    for log_file in log_file_list:
        print_performance_tests_summary(log_file, functionality_group_list, numRuns)