# Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import subprocess  # nosec
import argparse
import sys
import datetime
import shutil

# Set the timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

scriptPath = os.path.dirname(os.path.realpath(__file__))
inFilePath = scriptPath + "/../TEST_AUDIO_FILES/three_samples_single_channel_src1"
outFolderPath = os.getcwd()
buildFolderPath = os.getcwd()
caseMin = 0
caseMax = 2

# Checks if the folder path is empty, or is it a root folder, or if it exists, and remove its contents
def validate_and_remove_files(path):
    if not path:  # check if a string is empty
        print("Folder path is empty.")
        exit(0)

    elif path == "/*":  # check if the root directory is passed to the function
        print("Root folder cannot be deleted.")
        exit(0)

    elif os.path.exists(path):  # check if the folder exists
        # Get a list of files and directories within the specified path
        items = os.listdir(path)

        if items:
            # The directory is not empty, delete its contents
            for item in items:
                item_path = os.path.join(path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)     # Delete the directory if it exists

    else:
        print("Path is invalid or does not exist.")
        exit(0)

# Check if the folder is the root folder or exists, and remove the specified subfolders
def validate_and_remove_folders(path, folder):
    if path == "/*":  # check if the root directory is passed to the function
        print("Root folder cannot be deleted.")
        exit(0)
    if path and os.path.isdir(path):  # checks if directory string is not empty and it exists
        output_folders = [folder_name for folder_name in os.listdir(path) if folder_name.startswith(folder)]

        # Loop through each directory and delete it only if it exists
        for folder_name in output_folders:
            folder_path = os.path.join(path, folder_name)
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)  # Delete the directory if it exists
                print("Deleted directory:", folder_path)
            else:
                print("Directory not found:", folder_path)

# Validate if a path exists and is a directory
def validate_path(input_path):
    if not os.path.exists(input_path):
        raise ValueError("path " + input_path +" does not exist.")
    if not os.path.isdir(input_path):
        raise ValueError("path " + input_path + " is not a directory.")

# Get a list of log files based on a flag for preserving output
def get_log_file_list():
    return [
        outFolderPath + "/OUTPUT_PERFORMANCE_AUDIO_LOGS_HOST_" + timestamp + "/Tensor_host_audio_raw_performance_log.txt",
    ]

def run_unit_test(srcPath, case, numRuns, testType, batchSize, outFilePath):
    print("\n\n\n\n")
    print("--------------------------------")
    print("Running a New Functionality...")
    print("--------------------------------")
    print(f"./Tensor_host_audio {srcPath} {case} {numRuns} {testType} {numRuns} {batchSize}")
    result = subprocess.run([buildFolderPath + "/build/Tensor_host_audio", srcPath, str(case), str(testType), str(numRuns), str(batchSize), outFilePath, scriptPath], stdout=subprocess.PIPE)    # nosec
    print(result.stdout.decode())

    print("------------------------------------------------------------------------------------------")

def run_performance_test(loggingFolder, srcPath, case, numRuns, testType, batchSize, outFilePath):
    print("\n\n\n\n")
    print("--------------------------------")
    print("Running a New Functionality...")
    print("--------------------------------")
    with open("{}/Tensor_host_audio_raw_performance_log.txt".format(loggingFolder), "a") as log_file:
        print(f"./Tensor_host_audio {srcPath} {case} {numRuns} {testType} {numRuns} {batchSize} ")
        process = subprocess.Popen([buildFolderPath + "/build/Tensor_host_audio", srcPath, str(case), str(testType), str(numRuns), str(batchSize), outFilePath, scriptPath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)    # nosec
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
    parser.add_argument("--input_path", type = str, default = inFilePath, help = "Path to the input folder")
    parser.add_argument("--case_start", type = int, default = caseMin, help = "Testing start case # - Range must be in [" + str(caseMin) + ":" + str(caseMax) + "]")
    parser.add_argument("--case_end", type = int, default = caseMax, help = "Testing end case # - Range must be in [" + str(caseMin) + ":" + str(caseMax) + "]")
    parser.add_argument('--test_type', type = int, default = 0, help = "Type of Test - (0 = QA tests / 1 = Performance tests)")
    parser.add_argument('--qa_mode', type = int, default = 0, help = "Run with qa_mode? Output audio data from tests will be compared with golden outputs - (0 / 1)", required = False)
    parser.add_argument('--case_list', nargs = "+", help = "List of case numbers to test", required = False)
    parser.add_argument('--num_runs', type = int, default = 1, help = "Specifies the number of runs for running the performance tests")
    parser.add_argument('--preserve_output', type = int, default = 1, help = "preserves the output of the program - (0 = override output / 1 = preserve output )")
    parser.add_argument('--batch_size', type = int, default = 1, help = "Specifies the batch size to use for running tests. Default is 1.")
    args = parser.parse_args()

    # check if the folder exists
    validate_path(args.input_path)

    # validate the parameters passed by user
    if ((args.case_start < caseMin or args.case_start > caseMax) or (args.case_end < caseMin or args.case_end > caseMax)):
        print(f"Starting case# and Ending case# must be in the {caseMin}:{caseMax} range. Aborting!")
        exit(0)
    elif args.case_end < args.case_start:
        print("Ending case# must be greater than starting case#. Aborting!")
        exit(0)
    elif args.test_type < 0 or args.test_type > 1:
        print("Test Type# must be in the 0 / 1. Aborting!")
        exit(0)
    elif args.case_list is not None and args.case_start != caseMin and args.case_end != caseMax:
        print("Invalid input! Please provide only 1 option between case_list, case_start and case_end")
        exit(0)
    elif args.qa_mode < 0 or args.qa_mode > 1:
        print("QA mode must be in the 0 / 1. Aborting!")
        exit(0)
    elif args.num_runs <= 0:
        print("Number of Runs must be greater than 0. Aborting!")
        exit(0)
    elif args.batch_size <= 0:
        print("Batch size must be greater than 0. Aborting!")
        exit(0)
    elif args.preserve_output < 0 or args.preserve_output > 1:
        print("Preserve Output must be in the 0/1 (0 = override / 1 = preserve). Aborting")
        exit(0)
    elif args.test_type == 0 and args.input_path != inFilePath:
        print("Invalid input path! QA mode can run only with path:", inFilePath)
        exit(0)

    if args.case_list is None:
        args.case_list = range(args.case_start, args.case_end + 1)
        args.case_list = [str(x) for x in args.case_list]
    else:
        for case in args.case_list:
            if int(case) < caseMin or int(case) > caseMax:
                print(f"Invalid case number {case}! Case number must be in the {caseMin}:{caseMax} range. Aborting!")
                exit(0)
    return args

args = rpp_test_suite_parser_and_validator()
srcPath = args.input_path
caseStart = args.case_start
caseEnd = args.case_end
testType = args.test_type
caseList = args.case_list
qaMode = args.qa_mode
numRuns = args.num_runs
preserveOutput = args.preserve_output
batchSize = args.batch_size
outFilePath = " "

# Override testType to 0 if testType is 1 and qaMode is 1
if testType == 1 and qaMode == 1:
    print("WARNING: QA Mode cannot be run with testType = 1 (performance tests). Resetting testType to 0")
    testType = 0

# set the output folders and number of runs based on type of test (unit test / performance test)
if(testType == 0):
    outFilePath = outFolderPath + "/QA_RESULTS_AUDIO_HOST_" + timestamp
    numRuns = 1
elif(testType == 1):
    if "--num_runs" not in sys.argv:
        numRuns = 100   #default numRuns for running performance tests
    outFilePath = outFolderPath + "/OUTPUT_PERFORMANCE_AUDIO_LOGS_HOST_" + timestamp
else:
    print("Invalid TEST_TYPE specified. TEST_TYPE should be 0/1 (0 = QA tests / 1 = Performance tests)")
    exit(0)

if preserveOutput == 0:
    validate_and_remove_folders(outFolderPath, "QA_RESULTS_AUDIO_HOST")
    validate_and_remove_folders(outFolderPath, "OUTPUT_PERFORMANCE_AUDIO_LOGS_HOST")

os.mkdir(outFilePath)
loggingFolder = outFilePath
dstPath = outFilePath

# Validate DST_FOLDER
validate_and_remove_files(dstPath)

# Enable extglob
if os.path.exists(buildFolderPath + "/build"):
    shutil.rmtree(buildFolderPath + "/build")
os.makedirs(buildFolderPath + "/build")
os.chdir(buildFolderPath + "/build")

# Run cmake and make commands
subprocess.run(["cmake", scriptPath], cwd=".")   # nosec
subprocess.run(["make", "-j16"], cwd=".")    # nosec

if testType == 0:
    if batchSize != 3:
        print("QA tests can only run with a batch size of 3.")
        exit(0)

    for case in caseList:
        run_unit_test(srcPath, case, numRuns, testType, batchSize, outFilePath)
else:
    for case in caseList:
        run_performance_test(loggingFolder, srcPath, case, numRuns, testType, batchSize, outFilePath)

# print the results of qa tests
supportedCaseList = ['0', '1', '2']
nonQACaseList = [] # Add cases present in supportedCaseList, but without QA support

if testType == 0:
    qaFilePath = os.path.join(outFilePath, "QA_results.txt")
    checkFile = os.path.isfile(qaFilePath)
    if checkFile:
        f = open(qaFilePath, 'r+')
        print("---------------------------------- Results of QA Test - Tensor_host_audio -----------------------------------\n")
        numLines = 0
        numPassed = 0
        for line in f:
            sys.stdout.write(line)
            numLines += 1
            if "PASSED" in line:
                numPassed += 1
            sys.stdout.flush()
        resultsInfo = "\n\nFinal Results of Tests:"
        resultsInfo += "\n    - Total test cases including all subvariants REQUESTED = " + str(numLines)
        resultsInfo += "\n    - Total test cases including all subvariants PASSED = " + str(numPassed)
        resultsInfo += "\n\nGeneral information on Tensor test suite availability:"
        resultsInfo += "\n    - Total augmentations supported in Tensor audio test suite = " + str(len(supportedCaseList))
        resultsInfo += "\n    - Total augmentations with golden output QA test support = " + str(len(supportedCaseList) - len(nonQACaseList))
        resultsInfo += "\n    - Total augmentations without golden ouput QA test support (due to randomization involved) = " + str(len(nonQACaseList))
        f.write(resultsInfo)
        print("\n-------------------------------------------------------------------" + resultsInfo + "\n\n-------------------------------------------------------------------")

# Performance tests
if (testType == 1):
    log_file_list = get_log_file_list()

    try:
        f = open(log_file_list[0], "r")
        print("\n\n\nOpened log file -> "+ log_file_list[0])
    except IOError:
        print("Skipping file -> "+ log_file_list[0])
        exit(0)

    # Initialize data structures to store the parsed data
    functions = []
    max_wall_times = []
    min_wall_times = []
    avg_wall_times = []
    prev_line = ""
    funcCount = 0

    for line in f:
            if "max,min,avg wall times in ms/batch" in line:
                split_word_start = "Running "
                split_word_end = " " + str(numRuns)
                prev_line = prev_line.partition(split_word_start)[2].partition(split_word_end)[0]
                if prev_line not in functions:
                    functions.append(prev_line)
                    split_word_start = "max,min,avg wall times in ms/batch = "
                    split_word_end = "\n"
                    stats = line.partition(split_word_start)[2].partition(split_word_end)[0].split(",")
                    max_wall_times.append(float(stats[0]))
                    min_wall_times.append(float(stats[1]))
                    avg_wall_times.append(float(stats[2]))
                    funcCount += 1

            if line != "\n":
                prev_line = line

    # Print log lengths
    print("Functionalities - "+ str(funcCount))

    # Print the summary in a well-formatted table
    print("\n\nFunctionality\t\t\t\t\t\tnumRuns\t\tmax(ms/batch)\t\tmin(ms/batch)\t\tavg(ms/batch)\n")

    if len(functions) > 0:
        max_func_length = max(len(func) for func in functions)

        for i, func in enumerate(functions):
            print("{func}\t\t\t\t{numRuns}\t{:<15.6f}\t{:<15.6f}\t{:<15.6f}".format(
                max_wall_times[i], min_wall_times[i], avg_wall_times[i], func=func, numRuns=numRuns))
    else:
        print("No functionality data found in the log file.")