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
headerFilePath = scriptPath + "/../TEST_QA_IMAGES_VOXEL"
dataFilePath = scriptPath + "/../TEST_QA_IMAGES_VOXEL"
qaInputFile = scriptPath + "/../TEST_QA_IMAGES_VOXEL"
outFolderPath = os.getcwd()
buildFolderPath = os.getcwd()
caseMin = 0
caseMax = 6

# Check if folder path is empty, if it is the root folder, or if it exists, and remove its contents
def validate_and_remove_contents(path):
    if not path:  # check if a string is empty
        print("Folder path is empty.")
        exit()
    elif path == "/*":  # check if the root directory is passed to the function
        print("Root folder cannot be deleted.")
        exit()
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
        exit()

# Check if the folder is the root folder or exists, and remove the specified subfolders
def validate_and_remove_folders(path, folder):
    if path == "/*":  # check if the root directory is passed to the function
        print("Root folder cannot be deleted.")
        exit()
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

def case_file_check(CASE_FILE_PATH, TYPE, TENSOR_TYPE_LIST, new_file):
    try:
        case_file = open(CASE_FILE_PATH,'r')
        for line in case_file:
            print(line)
            if not(line.startswith('"Name"')):
                if TYPE in TENSOR_TYPE_LIST:
                    new_file.write(line)
                    d_counter[TYPE] = d_counter[TYPE] + 1
        case_file.close()
        return True
    except IOError:
        print("Unable to open case results")
        return False

# Generate a directory name based on certain parameters
def directory_name_generator(qaMode, affinity, layoutType, case, path):
    if qaMode == 0:
        functionality_group = func_group_finder(int(case))
        dst_folder_temp = "{}/rpp_{}_{}_{}".format(path, affinity, layoutType, functionality_group)
    else:
        dst_folder_temp = path

    return dst_folder_temp

# Process the layout based on the given parameters and generate the directory name and log file layout.
def process_layout(layout, qaMode, case, dstPath):
    if layout == 0:
        dstPathTemp = directory_name_generator(qaMode, "hip", "pkd3", case, dstPath)
        logFileLayout = "pkd3"
    elif layout == 1:
        dstPathTemp = directory_name_generator(qaMode, "hip", "pln3", case, dstPath)
        logFileLayout = "pln3"
    elif layout == 2:
        dstPathTemp = directory_name_generator(qaMode, "hip", "pln1", case, dstPath)
        logFileLayout = "pln1"

    return dstPathTemp, logFileLayout

# Validate if a path exists and is a directory
def validate_path(input_path):
    if not os.path.exists(input_path):
        raise ValueError("path " + input_path +" does not exist.")
    if not os.path.isdir(input_path):
        raise ValueError("path " + input_path + " is not a directory.")

# Create layout directories within a destination path based on a layout dictionary
def create_layout_directories(dst_path, layout_dict):
    for layout in range(3):
        current_layout = layout_dict[layout]
        try:
            os.makedirs(dst_path + '/' + current_layout)
        except FileExistsError:
            pass
        folder_list = [f for f in os.listdir(dst_path) if current_layout.lower() in f]
        for folder in folder_list:
            os.rename(dst_path + '/' + folder, dst_path + '/' + current_layout +  '/' + folder)

def get_log_file_list(preserveOutput):
    return [
        outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HIP_VOXEL_" + timestamp + "/Tensor_voxel_hip_pkd3_raw_performance_log.txt",
        outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HIP_VOXEL_" + timestamp + "/Tensor_voxel_hip_pln3_raw_performance_log.txt",
        outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HIP_VOXEL_" + timestamp + "/Tensor_voxel_hip_pln1_raw_performance_log.txt"
    ]

# Functionality group finder
def func_group_finder(case_number):
    if case_number == 0:
        return "arithmetic_operations"
    elif case_number == 1:
        return "geometric_augmentations"
    else:
        return "miscellaneous"

# Generate performance reports based on counters and a list of types
def generate_performance_reports(d_counter, TYPE_LIST):
    import pandas as pd
    pd.options.display.max_rows = None
    # Generate performance report
    for TYPE in TYPE_LIST:
        print("\n\n\nKernels tested - ", d_counter[TYPE], "\n\n")
        df = pd.read_csv(RESULTS_DIR + "/consolidated_results_" + TYPE + ".stats.csv")
        df["AverageMs"] = df["AverageNs"] / 1000000
        dfPrint = df.drop(['Percentage'], axis = 1)
        dfPrint["HIP Kernel Name"] = dfPrint.iloc[:,0].str.lstrip("Hip_")
        dfPrint_noIndices = dfPrint.astype(str)
        dfPrint_noIndices.replace(['0', '0.0'], '', inplace = True)
        dfPrint_noIndices = dfPrint_noIndices.to_string(index = False)
        print(dfPrint_noIndices)

def run_unit_test(headerPath, dataPath, dstPathTemp, layout, case, numRuns, testType, qaMode, batchSize):
    print("\n\n\n\n")
    print("--------------------------------")
    print("Running a New Functionality...")
    print("--------------------------------")
    bitDepths = [0, 2]
    for bitDepth in bitDepths:
        print("\n\n\nRunning New Bit Depth...\n-------------------------\n\n")
        print(f"./Tensor_voxel_hip {headerPath} {dataPath} {dstPathTemp} {layout} {case} {numRuns} {testType} {qaMode} {batchSize} {bitDepth}")
        result = subprocess.run([buildFolderPath + "/build/Tensor_voxel_hip", headerPath, dataPath, dstPathTemp, str(layout), str(case), str(numRuns), str(testType), str(qaMode), str(batchSize), str(bitDepth), scriptPath], stdout=subprocess.PIPE) # nosec
        print(result.stdout.decode())
        print("------------------------------------------------------------------------------------------")

def run_performance_test(loggingFolder, logFileLayout, headerPath, dataPath, dstPathTemp, layout, case, numRuns, testType, qaMode, batchSize):
    print("\n\n\n\n")
    print("--------------------------------")
    print("Running a New Functionality...")
    print("--------------------------------")
    bitDepths = [0, 2]
    for bitDepth in bitDepths:
        with open(f"{loggingFolder}/Tensor_voxel_hip_{logFileLayout}_raw_performance_log.txt", "a") as log_file:
            print(f"./Tensor_voxel_hip {headerPath} {dataPath} {dstPathTemp} {layout} {case}{numRuns} {testType} {qaMode} {batchSize} {bitDepth}")
            process = subprocess.Popen([buildFolderPath + "/build/Tensor_voxel_hip", headerPath, dataPath, dstPathTemp, str(layout), str(case), str(numRuns), str(testType), str(qaMode), str(batchSize), str(bitDepth), scriptPath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) # nosec
            while True:
                output = process.stdout.readline()
                if not output and process.poll() is not None:
                    break
                print(output.strip())
                if "Running" in output or "max,min,avg wall times" in output:
                    cleaned_output = ''.join(char for char in output if 32 <= ord(char) <= 126)  # Remove control characters
                    cleaned_output = cleaned_output.strip()  # Remove leading/trailing whitespace
                    log_file.write(cleaned_output + '\n')
                    if "max,min,avg wall times" in output:
                        log_file.write("\n")
        print("------------------------------------------------------------------------------------------")

def run_performance_test_with_profiler(loggingFolder, logFileLayout, dstPath, headerPath, dataPath, dstPathTemp, layout, case, numRuns, testType, qaMode, batchSize):
    print("\n\n\n\n")
    print("--------------------------------")
    print("Running a New Functionality...")
    print("--------------------------------")
    bitDepths = [0, 2]
    for bitDepth in bitDepths:
        if layout == 0:
            if not os.path.exists(f"{dstPath}/Tensor_PKD3/case_{case}"):
                os.mkdir(f"{dstPath}/Tensor_PKD3/case_{case}")
            with open(f"{loggingFolder}/Tensor_voxel_hip_{logFileLayout}_raw_performance_log.txt", "a") as log_file:
                print(f"\nrocprof --basenames on --timestamp on --stats -o {dstPathTemp}/Tensor_PKD3/case_{case}/output_case{case}.csv ./Tensor_voxel_hip {headerPath} {dataPath} {dstPathTemp}  {layout} {case}{numRuns} {testType} {qaMode} {batchSize} {bitDepth}")
                process = subprocess.Popen([ 'rocprof', '--basenames', 'on', '--timestamp', 'on', '--stats', '-o', f"{dstPath}/Tensor_PKD3/case_{case}/output_case{case}.csv", buildFolderPath + "/build/Tensor_voxel_hip", headerPath, dataPath, dstPathTemp, str(layout), str(case), str(numRuns), str(testType), str(qaMode), str(batchSize), str(bitDepth), scriptPath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT) # nosec
                while True:
                    output = process.stdout.readline()
                    if not output and process.poll() is not None:
                        break
                    print(output.strip())
                    log_file.write(output.decode('utf-8'))
        elif layout == 1:
            if not os.path.exists(f"{dstPath}/Tensor_PLN3/case_{case}"):
                os.mkdir(f"{dstPath}/Tensor_PLN3/case_{case}")
            with open(f"{loggingFolder}/Tensor_voxel_hip_{logFileLayout}_raw_performance_log.txt", "a") as log_file:
                print(f"\nrocprof --basenames on --timestamp on --stats -o {dstPathTemp}/Tensor_PLN3/case_{case}/output_case{case}.csv ./Tensor_voxel_hip {headerPath} {dataPath} {dstPathTemp}  {layout} {case}{numRuns} {testType} {qaMode} {batchSize} {bitDepth}")
                process = subprocess.Popen([ 'rocprof', '--basenames', 'on', '--timestamp', 'on', '--stats', '-o', f"{dstPath}/Tensor_PLN3/case_{case}/output_case{case}.csv", buildFolderPath + "/build/Tensor_voxel_hip", headerPath, dataPath, dstPathTemp, str(layout), str(case), str(numRuns), str(testType), str(qaMode), str(batchSize), str(bitDepth), scriptPath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT) # nosec
                while True:
                    output = process.stdout.readline()
                    if not output and process.poll() is not None:
                        break
                    print(output.strip())
                    log_file.write(output.decode('utf-8'))
        elif layout == 2:
            if not os.path.exists(f"{dstPath}/Tensor_PLN1/case_{case}"):
                os.mkdir(f"{dstPath}/Tensor_PLN1/case_{case}")
            with open(f"{loggingFolder}/Tensor_voxel_hip_{logFileLayout}_raw_performance_log.txt", "a") as log_file:
                print(f"\nrocprof --basenames on --timestamp on --stats -o {dstPathTemp}/Tensor_PLN1/case_{case}/output_case{case}.csv ./Tensor_voxel_hip {headerPath} {dataPath} {dstPathTemp}  {layout} {case}{numRuns} {testType} {qaMode} {batchSize} {bitDepth}")
                process = subprocess.Popen([ 'rocprof', '--basenames', 'on', '--timestamp', 'on', '--stats', '-o', f"{dstPath}/Tensor_PLN1/case_{case}/output_case{case}.csv", buildFolderPath + "/build/Tensor_voxel_hip", headerPath, dataPath, dstPathTemp, str(layout), str(case), str(numRuns), str(testType), str(qaMode), str(batchSize), str(bitDepth), scriptPath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT) # nosec
                while True:
                    output = process.stdout.readline()
                    if not output and process.poll() is not None:
                        break
                    print(output.strip())
                    log_file.write(output.decode('utf-8'))
    print("------------------------------------------------------------------------------------------")

# Parse and validate command-line arguments for the RPP test suite
def rpp_test_suite_parser_and_validator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--header_path", type = str, default = headerFilePath, help = "Path to the nii header")
    parser.add_argument("--data_path", type = str, default = dataFilePath, help = "Path to the nii data file")
    parser.add_argument("--case_start", type = int, default = caseMin, help = "Testing start case # - Range must be in [" + str(caseMin) + ":" + str(caseMax) + "]")
    parser.add_argument("--case_end", type = int, default = caseMax, help = "Testing start case # - Range must be in [" + str(caseMin) + ":" + str(caseMax) + "]")
    parser.add_argument('--test_type', type = int, default = 0, help = "Type of Test - (0 = Unit tests / 1 = Performance tests)")
    parser.add_argument('--case_list', nargs = "+", help = "List of case numbers to list", required = False)
    parser.add_argument('--profiling', type = str , default = 'NO', help = 'Run with profiler? - (YES/NO)', required = False)
    parser.add_argument('--qa_mode', type = int, default = 0, help = "Run with qa_mode? Output images from tests will be compared with golden outputs - (0 / 1)", required = False)
    parser.add_argument('--num_runs', type = int, default = 1, help = "Specifies the number of runs for running the performance tests")
    parser.add_argument('--preserve_output', type = int, default = 1, help = "preserves the output of the program - (0 = override output / 1 = preserve output )" )
    parser.add_argument('--batch_size', type = int, default = 1, help = "Specifies the batch size to use for running tests. Default is 1.")
    args = parser.parse_args()

    # check if the folder exists
    validate_path(args.header_path)
    validate_path(args.data_path)
    validate_path(qaInputFile)

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
    elif args.qa_mode < 0 or args.qa_mode > 1:
        print("QA mode must be in the 0 / 1. Aborting!")
        exit(0)
    elif args.case_list is not None and args.case_start > caseMin and args.case_end < caseMax:
        print("Invalid input! Please provide only 1 option between case_list, case_start and case_end")
        exit(0)
    elif args.num_runs <= 0:
        print("Number of Runs must be greater than 0. Aborting!")
        exit(0)
    elif args.profiling != 'YES' and args.profiling != 'NO':
        print("Profiling option value must be either 'YES' or 'NO'.")
        exit(0)
    elif args.batch_size <= 0:
        print("Batch size must be greater than 0. Aborting!")
        exit(0)
    elif args.preserve_output < 0 or args.preserve_output > 1:
        print("Preserve Output must be in the 0/1 (0 = override / 1 = preserve). Aborting")
        exit(0)

    if args.case_list is None:
        args.case_list = range(args.case_start, args.case_end + 1)
        args.case_list = [str(x) for x in args.case_list]
    else:
        for case in args.case_list:
            if int(case) < caseMin or int(case) > caseMax:
                print("The case# must be in [" + str(caseMin) + ":" + str(caseMax) + "]")
                exit(0)

    # if QA mode is enabled overwrite the input folders with the folders used for generating golden outputs
    if args.qa_mode:
        args.header_path = headerFilePath
        args.data_path = dataFilePath

    return args

args = rpp_test_suite_parser_and_validator()
headerPath = args.header_path
dataPath = args.data_path
caseStart = args.case_start
caseEnd = args.case_end
testType = args.test_type
caseList = args.case_list
profilingOption = args.profiling
qaMode = args.qa_mode
numRuns = args.num_runs
preserveOutput = args.preserve_output
batchSize = args.batch_size

if qaMode and os.path.abspath(qaInputFile) != os.path.abspath(headerPath):
    print("QA mode should only run with the given Input path: ", qaInputFile)
    exit(0)

if qaMode and batchSize != 3:
    print("QA mode can only run with a batch size of 3.")
    exit(0)

# set the output folders and number of runs based on type of test (unit test / performance test)
if(testType == 0):
    if qaMode:
        outFilePath = outFolderPath + "/QA_RESULTS_HIP_VOXEL_" + timestamp
    else:
        outFilePath = outFolderPath + "/OUTPUT_VOXEL_HIP_" + timestamp
    numRuns = 1
elif(testType == 1):
    if "--num_runs" not in sys.argv:
        numRuns = 100 #default numRuns for running performance tests
    outFilePath = outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HIP_VOXEL_" + timestamp
else:
    print("Invalid TEST_TYPE specified. TEST_TYPE should be 0/1 (0 = Unittests / 1 = Performancetests)")
    exit()

if preserveOutput == 0:
    validate_and_remove_folders(outFolderPath, "OUTPUT_VOXEL_HIP")
    validate_and_remove_folders(outFolderPath, "QA_RESULTS_HIP_VOXEL")
    validate_and_remove_folders(outFolderPath, "OUTPUT_PERFORMANCE_LOGS_HIP_VOXEL")

os.mkdir(outFilePath)
loggingFolder = outFilePath
dstPath = outFilePath

# Validate DST_FOLDER
validate_and_remove_contents(dstPath)

# Enable extglob
if os.path.exists(buildFolderPath + "/build"):
    shutil.rmtree(buildFolderPath + "/build")
os.makedirs(buildFolderPath + "/build")
os.chdir(buildFolderPath + "/build")

# Run cmake and make commands
subprocess.run(["cmake", scriptPath], cwd=".")   # nosec
subprocess.run(["make", "-j16"], cwd=".")  # nosec

# Create folders based on testType and profilingOption
if testType == 1 and profilingOption == "YES":
    os.makedirs(f"{dstPath}/Tensor_PKD3")
    os.makedirs(f"{dstPath}/Tensor_PLN1")
    os.makedirs(f"{dstPath}/Tensor_PLN3")

print("\n\n\n\n\n")
print("##########################################################################################")
print("Running all layout Inputs...")
print("##########################################################################################")

if testType == 0:
    for case in caseList:
        for layout in range(3):
            dstPathTemp, logFileLayout = process_layout(layout, qaMode, case, dstPath)
            if qaMode == 0:
                if not os.path.isdir(dstPathTemp):
                    os.mkdir(dstPathTemp)

            run_unit_test(headerPath, dataPath, dstPathTemp, layout, case, numRuns, testType, qaMode, batchSize)
elif (testType == 1 and profilingOption == "NO"):
    for case in caseList:
        for layout in range(3):
            dstPathTemp, logFileLayout = process_layout(layout, qaMode, case, dstPath)
            run_performance_test(loggingFolder, logFileLayout, headerPath, dataPath, dstPathTemp, layout, case, numRuns, testType, qaMode, batchSize)
elif (testType == 1 and profilingOption == "YES"):
    NEW_FUNC_GROUP_LIST = [0, 1]
    for case in caseList:
        for layout in range(3):
            dstPathTemp, logFileLayout = process_layout(layout, qaMode, case, dstPath)
            run_performance_test_with_profiler(loggingFolder, logFileLayout, dstPath, headerPath, dataPath, dstPathTemp, layout, case, numRuns, testType, qaMode, batchSize)

        RESULTS_DIR = ""
        RESULTS_DIR = outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HIP_VOXEL_" + timestamp
        print("RESULTS_DIR = " + RESULTS_DIR)
        CONSOLIDATED_FILE_TENSOR_PKD3 = RESULTS_DIR + "/consolidated_results_Tensor_PKD3.stats.csv"
        CONSOLIDATED_FILE_TENSOR_PLN1 = RESULTS_DIR + "/consolidated_results_Tensor_PLN1.stats.csv"
        CONSOLIDATED_FILE_TENSOR_PLN3 = RESULTS_DIR + "/consolidated_results_Tensor_PLN3.stats.csv"

        TYPE_LIST = ["Tensor_PKD3", "Tensor_PLN1", "Tensor_PLN3"]
        TENSOR_TYPE_LIST = ["Tensor_PKD3", "Tensor_PLN1", "Tensor_PLN3"]
        CASE_NUM_LIST = caseList
        BIT_DEPTH_LIST = [0, 2]
        OFT_LIST = [0]
        d_counter = {"Tensor_PKD3":0, "Tensor_PLN1":0, "Tensor_PLN3":0}

        for TYPE in TYPE_LIST:
            # Open csv file
            new_file = open(RESULTS_DIR + "/consolidated_results_" + TYPE + ".stats.csv",'w')
            new_file.write('"HIP Kernel Name","Calls","TotalDurationNs","AverageNs","Percentage"\n')

            prev = ""

            # Loop through cases
            for CASE_NUM in CASE_NUM_LIST:

                # Add functionality group header
                if CASE_NUM in NEW_FUNC_GROUP_LIST:
                    FUNC_GROUP = func_group_finder(CASE_NUM)
                    new_file.write("0,0,0,0,0\n")
                    new_file.write(FUNC_GROUP + ",0,0,0,0\n")
                    new_file.write("0,0,0,0,0\n")

                # Set results directory
                CASE_RESULTS_DIR = RESULTS_DIR + "/" + TYPE + "/case_" + str(CASE_NUM)
                print("CASE_RESULTS_DIR = " + CASE_RESULTS_DIR)

                # Loop through bit depths
                for BIT_DEPTH in BIT_DEPTH_LIST:
                    # Loop through output format toggle cases
                    for OFT in OFT_LIST:
                        # Write into csv file
                        CASE_FILE_PATH = CASE_RESULTS_DIR + "/output_case" + str(CASE_NUM) + ".stats.csv"
                        print("CASE_FILE_PATH = " + CASE_FILE_PATH)
                        fileCheck = case_file_check(CASE_FILE_PATH, TYPE, TENSOR_TYPE_LIST, new_file)
                        if fileCheck == False:
                            continue

            new_file.close()
            subprocess.call(['chown', '{}:{}'.format(os.getuid(), os.getgid()), RESULTS_DIR + "/consolidated_results_" + TYPE + ".stats.csv"])  # nosec
        try:
            generate_performance_reports(d_counter, TYPE_LIST)

        except ImportError:
            print("\nPandas not available! Results of GPU profiling experiment are available in the following files:\n" + \
                    CONSOLIDATED_FILE_TENSOR_PKD3 + "\n" + \
                        CONSOLIDATED_FILE_TENSOR_PLN1 + "\n" + \
                            CONSOLIDATED_FILE_TENSOR_PLN3 + "\n")

        except IOError:
            print("Unable to open results in " + RESULTS_DIR + "/consolidated_results_" + TYPE + ".stats.csv")

# print the results of qa tests
supportedCaseList = ['0', '1', '2', '3', '5']
nonQACaseList = [] # Add cases present in supportedCaseList, but without QA support

if qaMode and testType == 0:
    qaFilePath = os.path.join(outFilePath, "QA_results.txt")
    checkFile = os.path.isfile(qaFilePath)
    if checkFile:
        f = open(qaFilePath, 'r+')
        print("---------------------------------- Results of QA Test - Tensor_voxel_hip ----------------------------------\n")
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
        resultsInfo += "\n\nGeneral information on Tensor voxel test suite availability:"
        resultsInfo += "\n    - Total augmentations supported in Tensor test suite = " + str(len(supportedCaseList))
        resultsInfo += "\n    - Total augmentations with golden output QA test support = " + str(len(supportedCaseList) - len(nonQACaseList))
        resultsInfo += "\n    - Total augmentations without golden ouput QA test support (due to randomization involved) = " + str(len(nonQACaseList))
        f.write(resultsInfo)
    print("\n-------------------------------------------------------------------" + resultsInfo + "\n\n-------------------------------------------------------------------")

layoutDict = {0:"PKD3", 1:"PLN3", 2:"PLN1"}
if (testType == 0 and qaMode == 0): # Unit tests
    create_layout_directories(dstPath, layoutDict)
elif (testType == 1 and profilingOption == "NO"): # Performance tests
    log_file_list = get_log_file_list(preserveOutput)

    functionality_group_list = [
        "arithmetic_operations",
        "geometric_augmentations",
    ]

    for log_file in log_file_list:
        # Opening log file
        try:
            f = open(log_file,"r")
            print("\n\n\nOpened log file -> "+ log_file)
        except IOError:
            print("Skipping file -> "+ log_file)
            continue

        stats = []
        maxVals = []
        minVals = []
        avgVals = []
        functions = []
        frames = []
        prevLine = ""
        funcCount = 0

        # Loop over each line
        for line in f:
            for functionality_group in functionality_group_list:
                if functionality_group in line:
                    functions.extend([" ", functionality_group, " "])
                    frames.extend([" ", " ", " "])
                    maxVals.extend([" ", " ", " "])
                    minVals.extend([" ", " ", " "])
                    avgVals.extend([" ", " ", " "])

            if "max,min,avg wall times in ms/batch" in line:
                split_word_start = "Running "
                split_word_end = " " +str(numRuns)
                prevLine = prevLine.partition(split_word_start)[2].partition(split_word_end)[0]
                if prevLine not in functions:
                    functions.append(prevLine)
                    frames.append(numRuns)
                    split_word_start = "max,min,avg wall times in ms/batch = "
                    split_word_end = "\n"
                    stats = line.partition(split_word_start)[2].partition(split_word_end)[0].split(",")
                    maxVals.append(stats[0])
                    minVals.append(stats[1])
                    avgVals.append(stats[2])
                    funcCount += 1

            if line != "\n":
                prevLine = line

        # Print log lengths
        print("Functionalities - "+ str(funcCount))

        # Print summary of log
        print("\n\nFunctionality\t\t\t\t\t\tFrames Count\t\tmax(ms/batch)\t\tmin(ms/batch)\t\tavg(ms/batch)\n")
        if len(functions) != 0:
            maxCharLength = len(max(functions, key = len))
            functions = [x + (' ' * (maxCharLength - len(x))) for x in functions]
            for i, func in enumerate(functions):
                print(func + "\t\t\t\t\t\t\t\t" + str(frames[i]) + "\t\t" + str(maxVals[i]) + "\t\t" + str(minVals[i]) + "\t\t" + str(avgVals[i]))
        else:
            print("No variants under this category")

        # Closing log file
        f.close()