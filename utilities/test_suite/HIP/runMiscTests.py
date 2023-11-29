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
cwd = os.getcwd()

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
    if path and os.path.isdir(path + "/.."):  # checks if directory string is not empty and it exists
        output_folders = [folder_name for folder_name in os.listdir(path + "/..") if folder_name.startswith(folder)]

        # Loop through each directory and delete it only if it exists
        for folder_name in output_folders:
            folder_path = os.path.join(path, "..", folder_name)
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)  # Delete the directory if it exists
                print("Deleted directory:", folder_path)
            else:
                print("Directory not found:", folder_path)

# Get a list of log files based on a flag for preserving output
def get_log_file_list():
    return [
        "../../OUTPUT_PERFORMANCE_MISC_LOGS_HIP_" + timestamp + "/Tensor_misc_hip_raw_performance_log.txt",
    ]

def case_file_check(CASE_FILE_PATH, new_file):
    try:
        case_file = open(CASE_FILE_PATH,'r')
        for line in case_file:
            print(line)
            if not(line.startswith('"Name"')):
                new_file.write(line)
        case_file.close()
        return True
    except IOError:
        print("Unable to open case results")
        return False

# Generate performance reports based on counters and a list of types
def generate_performance_reports(RESULTS_DIR):
    import pandas as pd
    pd.options.display.max_rows = None
    # Generate performance report
    df = pd.read_csv(RESULTS_DIR + "/consolidated_results.stats.csv")
    df["AverageMs"] = df["AverageNs"] / 1000000
    dfPrint = df.drop(['Percentage'], axis = 1)
    dfPrint["HIP Kernel Name"] = dfPrint.iloc[:,0].str.lstrip("Hip_")
    dfPrint_noIndices = dfPrint.astype(str)
    dfPrint_noIndices.replace(['0', '0.0'], '', inplace = True)
    dfPrint_noIndices = dfPrint_noIndices.to_string(index = False)
    print(dfPrint_noIndices)

def run_unit_test(numDims, case, numRuns, testType, toggle, bitDepth, batchSize, outFilePath):
    print("\n\n\n\n")
    print("--------------------------------")
    print("Running a New Functionality...")
    print("--------------------------------")
    print(f"./Tensor_misc_hip {case} {testType} {toggle} {numDims} {batchSize} {numRuns}")
    result = subprocess.run(["./Tensor_misc_hip", str(case), str(testType), str(toggle), str(numDims), str(batchSize), str(numRuns), outFilePath], stdout=subprocess.PIPE)    # nosec
    print(result.stdout.decode())

    print("------------------------------------------------------------------------------------------")

def run_performance_test(loggingFolder, numDims, case, numRuns, testType, toggle, bitDepth, batchSize, outFilePath):
    print("\n\n\n\n")
    print("--------------------------------")
    print("Running a New Functionality...")
    print("--------------------------------")
    with open("{}/Tensor_misc_hip_raw_performance_log.txt".format(loggingFolder), "a") as log_file:
        print(f"./Tensor_misc_hip {case} {testType} {toggle} {numDims} {batchSize} {numRuns}")
        process = subprocess.Popen(["./Tensor_misc_hip", str(case), str(testType), str(toggle), str(numDims), str(batchSize), str(numRuns) , outFilePath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)    # nosec
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
        process = subprocess.Popen([ 'rocprof', '--basenames', 'on', '--timestamp', 'on', '--stats', '-o', f"{outFilePath}/case_{case}/output_case{case}.csv", "./Tensor_misc_hip", str(case), str(testType), str(toggle), str(numDims), str(batchSize), str(numRuns) , outFilePath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)    # nosec
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
    parser.add_argument("--case_start", type = int, default = 1, help = "Testing range starting case # - (1:1)")
    parser.add_argument("--case_end", type = int, default = 1, help = "Testing range ending case # - (1:1)")
    parser.add_argument('--test_type', type = int, default = 0, help = "Type of Test - (0 = QA tests / 1 = Performance tests)")
    parser.add_argument('--toggle', type = int, default = 0, help = "Toggle outputs")
    parser.add_argument('--case_list', nargs = "+", help = "List of case numbers to test", required = False)
    parser.add_argument("--num_dims", type = int, default = 2, help = "Number of dimensions for input")
    parser.add_argument('--num_runs', type = int, default = 1, help = "Specifies the number of runs for running the performance tests")
    parser.add_argument('--profiling', type = str , default = 'NO', help = 'Run with profiler? - (YES/NO)', required = False)
    parser.add_argument('--batch_size', type = int, default = 1, help = "Specifies the batch size to use for running tests. Default is 1.")
    parser.add_argument('--preserve_output', type = int, default = 1, help = "preserves the output of the program - (0 = override output / 1 = preserve output )" )
    args = parser.parse_args()

    # validate the parameters passed by user
    if ((args.case_start < 1 or args.case_start > 1) or (args.case_end < 1 or args.case_end > 1)):
        print("Starting case# and Ending case# must be in the 1:1 range. Aborting!")
        exit(0)
    elif args.case_end < args.case_start:
        print("Ending case# must be greater than starting case#. Aborting!")
        exit(0)
    elif args.test_type < 0 or args.test_type > 1:
        print("Test Type# must be in the 0 / 1. Aborting!")
        exit(0)
    elif args.case_list is not None and args.case_start > 1 and args.case_end < 1:
        print("Invalid input! Please provide only 1 option between case_list, case_start and case_end")
        exit(0)
    elif args.num_runs <= 0:
        print("Number of Runs must be greater than 0. Aborting!")
        exit(0)
    elif args.batch_size <= 0:
        print("Batch size must be greater than 0. Aborting!")
        exit(0)
    elif args.test_type == 0 and args.num_dims != 3:
        print("Invalid Input! QA mode is supported only for num_dims = 3!")
        exit(0)
    elif args.profiling != 'YES' and args.profiling != 'NO':
        print("Profiling option value must be either 'YES' or 'NO'.")
        exit(0)

    if args.case_list is None:
        args.case_list = range(args.case_start, args.case_end + 1)
        args.case_list = [str(x) for x in args.case_list]
    else:
        for case in args.case_list:
            if int(case) != 1:
                 print("The case# must be 1!")
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
preserveOutput = args.preserve_output
bitDepth = 2 # Current audio test suite only supports bit depth 2
outFilePath = " "

if preserveOutput == 0:
    validate_and_remove_folders(cwd, "QA_RESULTS_MISC_HIP")
    validate_and_remove_folders(cwd, "OUTPUT_PERFORMANCE_MISC_LOGS_HIP")

if(testType == 0):
    outFilePath = os.path.join(os.path.dirname(cwd), 'QA_RESULTS_MISC_HIP_' + timestamp)
    numRuns = 1
elif(testType == 1):
    if "--num_runs" not in sys.argv:
        numRuns = 100   #default numRuns for running performance tests
    outFilePath = os.path.join(os.path.dirname(cwd), 'OUTPUT_PERFORMANCE_MISC_LOGS_HIP_' + timestamp)
else:
    print("Invalid TEST_TYPE specified. TEST_TYPE should be 0/1 (0 = QA tests / 1 = Performance tests)")
    exit(0)

os.mkdir(outFilePath)
loggingFolder = outFilePath
dstPath = outFilePath

# Enable extglob
if os.path.exists("build"):
    shutil.rmtree("build")
os.makedirs("build")
os.chdir("build")

# Run cmake and make commands
subprocess.run(["cmake", ".."], cwd=".")   # nosec
subprocess.run(["make", "-j16"], cwd=".")    # nosec

if testType == 0:
    for case in caseList:
        if batchSize != 3:
            print("QA tests can only run with a batch size of 3")
            exit(0)
        if toggle == 1:
            print("Only Toggle variant is QA tested for test Type 0. Aborting!")
            exit(0)
        if int(case) != 1:
            print(f"Invalid case number {case}. Case number must be 1!")
            continue

        run_unit_test(numDims, case, numRuns, testType, toggle, bitDepth, batchSize, outFilePath)
elif (testType == 1 and profilingOption == "NO"):
    for case in caseList:
        if int(case) != 1:
            print(f"Invalid case number {case}. Case number must be 1!")
            continue

        run_performance_test(loggingFolder, numDims, case, numRuns, testType, toggle, bitDepth, batchSize, outFilePath)
elif (testType == 1 and profilingOption == "YES"):
    for case in caseList:
        if int(case) != 1:
            print(f"Invalid case number {case}. Case number must be 1!")
            continue

    run_performance_test_with_profiler(loggingFolder, numDims, case, numRuns, testType, toggle, bitDepth, batchSize, outFilePath)

    RESULTS_DIR = ""
    RESULTS_DIR = "../../OUTPUT_PERFORMANCE_MISC_LOGS_HIP_" + timestamp
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
supportedCaseList = ['1']
supportedCases = 0
for num in caseList:
    if num in supportedCaseList:
        supportedCases += 1
caseInfo = "Tests are run for " + str(supportedCases) + " supported cases out of the " + str(len(caseList)) + " cases requested"
if testType == 0:
    qaFilePath = os.path.join(outFilePath, "QA_results.txt")
    f = open(qaFilePath, 'r+')
    print("---------------------------------- Results of QA Test ----------------------------------\n")
    for line in f:
        sys.stdout.write(line)
        sys.stdout.flush()
    f.write(caseInfo)
print("\n-------------- " + caseInfo + " --------------")

# Performance tests
if (testType == 1 and profilingOption == "NO"):
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