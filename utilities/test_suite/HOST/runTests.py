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
inFilePath1 = os.path.join(os.path.dirname(cwd), 'TEST_IMAGES', 'three_images_mixed_src1')
inFilePath2 = os.path.join(os.path.dirname(cwd), 'TEST_IMAGES', 'three_images_mixed_src2')
qaInputFile = os.path.join(os.path.dirname(cwd), 'TEST_IMAGES', 'three_images_mixed_src1')

# Checks if the folder path is empty, or is it a root folder, or if it exists, and remove its contents
def validate_and_remove_files(path):
    if not path:  # check if a string is empty
        print("Folder path is empty.")
        exit()

    elif path == "/*":  # check if the root directory is passed to the function
        print("Root folder cannot be deleted.")
        exit()

    elif os.path.exists(path):  # check if the folder exists
        os.system("rm -rvf {}/*".format(path))  # Delete the directory if it exists

    else:
        print("Path is invalid or does not exist.")
        exit()

# Check if the folder is the root folder or exists, and remove the specified subfolders
def validate_and_remove_folders(path, folder):
    if path == "/*":  # check if the root directory is passed to the function
        print("Root folder cannot be deleted.")
        exit()
    if path and os.path.isdir(path + "/.."):  # checks if directory string is not empty and it exists
        output_folders = [folder_name for folder_name in os.listdir(path + "/..") if folder_name.startswith(folder)]

        # Loop through each directory and delete it only if it exists
        for folder_name in output_folders:
            folder_path = os.path.join(path, "..", folder_name)
            if os.path.isdir(folder_path):
                os.system("rm -rf {}".format(folder_path))  # Delete the directory if it exists
                print("Deleted directory:", folder_path)
            else:
                print("Directory not found:", folder_path)

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

# Validate if a path exists and is a directory
def validate_path(input_path):
    if not os.path.exists(input_path):
        raise ValueError("path " + input_path +" does not exist.")
    if not os.path.isdir(input_path):
        raise ValueError("path " + input_path + " is not a directory.")

# Get a list of log files based on a flag for preserving output
def get_log_file_list(preserveOutput):
    return [
        "../../OUTPUT_PERFORMANCE_LOGS_HOST_" + timestamp + "/Tensor_host_pkd3_raw_performance_log.txt",
        "../../OUTPUT_PERFORMANCE_LOGS_HOST_" + timestamp + "/Tensor_host_pln3_raw_performance_log.txt",
        "../../OUTPUT_PERFORMANCE_LOGS_HOST_" + timestamp + "/Tensor_host_pln1_raw_performance_log.txt"
    ]

# Functionality group finder
def func_group_finder(case_number):
    if case_number < 5 or case_number == 13 or case_number == 36 or case_number == 31:
        return "color_augmentations"
    elif case_number == 8 or case_number == 30 or case_number == 83 or case_number == 84:
        return "effects_augmentations"
    elif case_number < 40:
        return "geometric_augmentations"
    elif case_number <= 86:
        return "data_exchange_operations"
    else:
        return "miscellaneous"

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
        dstPathTemp = directory_name_generator(qaMode, "host", "pkd3", case, dstPath)
        log_file_layout = "pkd3"
    elif layout == 1:
        dstPathTemp = directory_name_generator(qaMode, "host", "pln3", case, dstPath)
        log_file_layout = "pln3"
    elif layout == 2:
        dstPathTemp = directory_name_generator(qaMode, "host", "pln1", case, dstPath)
        log_file_layout = "pln1"

    return dstPathTemp, log_file_layout

def run_unit_test(srcPath1, srcPath2, dstPathTemp, bitDepth, outputFormatToggle, case, additionalParam, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList):
    result = subprocess.run(["./Tensor_host", srcPath1, srcPath2, dstPathTemp, str(bitDepth), str(outputFormatToggle), str(case), str(additionalParam), str(numRuns), str(testType), str(layout), "0", str(qaMode), str(decoderType), str(batchSize)] + roiList, stdout=subprocess.PIPE)
    print(result.stdout.decode())

def run_performance_test(loggingFolder, log_file_layout, srcPath1, srcPath2, dstPath, bitDepth, outputFormatToggle, case, additionalParam, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList):
    with open("{}/Tensor_host_{}_raw_performance_log.txt".format(loggingFolder, log_file_layout), "a") as log_file:
        print(f"./Tensor_host {srcPath1} {srcPath2} {dstPath} {bitDepth} {outputFormatToggle} {case} {additionalParam} 0 ")
        process = subprocess.Popen(["./Tensor_host", srcPath1, srcPath2, dstPath, str(bitDepth), str(outputFormatToggle), str(case), str(additionalParam), str(numRuns), str(testType), str(layout), "0", str(qaMode), str(decoderType), str(batchSize)] + roiList, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        while True:
            output = process.stdout.readline()
            if not output and process.poll() is not None:
                break
            print(output.strip())
            log_file.write(output)

# Parse and validate command-line arguments for the RPP test suite
def rpp_test_suite_parser_and_validator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path1", type = str, default = inFilePath1, help = "Path to the input folder 1")
    parser.add_argument("--input_path2", type = str, default = inFilePath2, help = "Path to the input folder 2")
    parser.add_argument("--case_start", type = int, default = 0, help = "Testing range starting case # - (0:86)")
    parser.add_argument("--case_end", type = int, default = 86, help = "Testing range ending case # - (0:86)")
    parser.add_argument('--test_type', type = int, default = 0, help = "Type of Test - (0 = Unit tests / 1 = Performance tests)")
    parser.add_argument('--case_list', nargs = "+", help = "List of case numbers to list", required = False)
    parser.add_argument('--qa_mode', type = int, default = 0, help = "Run with qa_mode? Output images from tests will be compared with golden outputs - (0 / 1)", required = False)
    parser.add_argument('--decoder_type', type = int, default = 0, help = "Type of Decoder to decode the input data - (0 = TurboJPEG / 1 = OpenCV)")
    parser.add_argument('--num_runs', type = int, default = 1, help = "Specifies the number of runs for running the performance tests")
    parser.add_argument('--preserve_output', type = int, default = 1, help = "preserves the output of the program - (0 = override output / 1 = preserve output )" )
    parser.add_argument('--batch_size', type = int, default = 1, help = "Specifies the batch size to use for running tests. Default is 1.")
    parser.add_argument('--roi', nargs = 4, help = "specifies the roi values", required = False)
    args = parser.parse_args()

    # check if the folder exists
    validate_path(args.input_path1)
    validate_path(args.input_path2)
    validate_path(qaInputFile)

    # validate the parameters passed by user
    if ((args.case_start < 0 or args.case_start > 86) or (args.case_end < 0 or args.case_end > 86)):
        print("Starting case# and Ending case# must be in the 0:86 range. Aborting!")
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
    elif args.decoder_type < 0 or args.decoder_type > 1:
        print("Decoder Type must be in the 0/1 (0 = OpenCV / 1 = TurboJPEG). Aborting")
        exit(0)
    elif args.case_list is not None and args.case_start > 0 and args.case_end < 86:
        print("Invalid input! Please provide only 1 option between case_list, case_start and case_end")
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
    elif args.roi is not None and any(int(val) < 0 for val in args.roi[:2]):
        print(" Invalid ROI. Aborting")
        exit(0)
    elif args.roi is not None and any(int(val) <= 0 for val in args.roi[2:]):
        print(" Invalid ROI. Aborting")
        exit(0)


    if args.case_list is None:
        args.case_list = range(args.case_start, args.case_end + 1)
        args.case_list = [str(x) for x in args.case_list]
    else:
        for case in args.case_list:
            if int(case) < 0 or int(case) > 86:
                 print("The case# must be in the 0:86 range!")
                 exit(0)
    # if QA mode is enabled overwrite the input folders with the folders used for generating golden outputs
    if args.qa_mode:
        args.input_path1 = inFilePath1
        args.input_path2 = inFilePath2

    return args

args = rpp_test_suite_parser_and_validator()
srcPath1 = args.input_path1
srcPath2 = args.input_path2
caseStart = args.case_start
caseEnd = args.case_end
testType = args.test_type
caseList = args.case_list
qaMode = args.qa_mode
decoderType = args.decoder_type
numRuns = args.num_runs
preserveOutput = args.preserve_output
batchSize = args.batch_size
roiList = ['0', '0', '0', '0'] if args.roi is None else args.roi

if preserveOutput == 0:
    validate_and_remove_folders(cwd, "OUTPUT_IMAGES_HOST")
    validate_and_remove_folders(cwd, "QA_RESULTS_HOST")
    validate_and_remove_folders(cwd, "OUTPUT_PERFORMANCE_LOGS_HOST")

if qaMode and os.path.abspath(qaInputFile) != os.path.abspath(srcPath1):
    print("QA mode should only run with the given Input path: ", qaInputFile)
    exit(0)

if qaMode and batchSize != 3:
    print("QA mode can only run with a batch size of 3.")
    exit(0)

if(testType == 0):
    if qaMode:
        outFilePath = os.path.join(os.path.dirname(cwd), 'QA_RESULTS_HOST_' + timestamp)
    else:
        outFilePath = os.path.join(os.path.dirname(cwd), 'OUTPUT_IMAGES_HOST_' + timestamp)
    numRuns = 1
elif(testType == 1):
    if "--num_runs" not in sys.argv:
        numRuns = 1000 #default numRuns for running performance tests
    outFilePath = os.path.join(os.path.dirname(cwd), 'OUTPUT_PERFORMANCE_LOGS_HOST_' + timestamp)
else:
    print("Invalid TEST_TYPE specified. TEST_TYPE should be 0/1 (0 = Unittests / 1 = Performancetests)")
    exit()

os.mkdir(outFilePath)
loggingFolder = outFilePath
dstPath = outFilePath

# Validate DST_FOLDER
validate_and_remove_files(dstPath)

# Enable extglob
if os.path.exists("build"):
    shutil.rmtree("build")
os.makedirs("build")
os.chdir("build")

# Run cmake and make commands
subprocess.run(["cmake", ".."])
subprocess.run(["make", "-j16"])

print("\n\n\n\n\n")
print("##########################################################################################")
print("Running all layout Inputs...")
print("##########################################################################################")

if testType == 0:
    for case in caseList:
        if int(case) < 0 or int(case) > 86:
            print(f"Invalid case number {case}. Case number must be in the range of 0 to 86!")
            continue
        for layout in range(3):
            dstPathTemp, log_file_layout = process_layout(layout, qaMode, case, dstPath)

            if qaMode == 0:
                if not os.path.isdir(dstPathTemp):
                    os.mkdir(dstPathTemp)

            print("\n\n\n\n")
            print("--------------------------------")
            print("Running a New Functionality...")
            print("--------------------------------")

            for bitDepth in range(7):
                print("\n\n\nRunning New Bit Depth...\n-------------------------\n\n")

                for outputFormatToggle in range(2):
                    # There is no layout toggle for PLN1 case, so skip this case
                    if layout == 2 and outputFormatToggle == 1:
                        continue

                    if case == "8":
                        # Run all variants of noise type functions with additional argument of noiseType = gausssianNoise / shotNoise / saltandpepperNoise
                        for noiseType in range(3):
                            print(f"./Tensor_host {srcPath1} {srcPath2} {dstPathTemp} {bitDepth} {outputFormatToggle} {case} {noiseType} 0 ")
                            run_unit_test(srcPath1, srcPath2, dstPathTemp, bitDepth, outputFormatToggle, case, noiseType, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList)
                    elif case == "21" or case == "23" or case == "24":
                        # Run all variants of interpolation functions with additional argument of interpolationType = bicubic / bilinear / gaussian / nearestneigbor / lanczos / triangular
                        for interpolationType in range(6):
                            print(f"./Tensor_host {srcPath1} {srcPath2} {dstPathTemp} {bitDepth} {outputFormatToggle} {case} {interpolationType} 0")
                            run_unit_test(srcPath1, srcPath2, dstPathTemp, bitDepth, outputFormatToggle, case, interpolationType, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList)
                    else:
                        print(f"./Tensor_host {srcPath1} {srcPath2} {dstPathTemp} {bitDepth} {outputFormatToggle} {case} 0 {numRuns} {testType} {layout} 0")
                        run_unit_test(srcPath1, srcPath2, dstPathTemp, bitDepth, outputFormatToggle, case, "0", numRuns, testType, layout, qaMode, decoderType, batchSize, roiList)

                    print("------------------------------------------------------------------------------------------")
    layoutDict = {0:"PKD3", 1:"PLN3", 2:"PLN1"}
    if qaMode == 0:
        create_layout_directories(dstPath, layoutDict)
else:
    for case in caseList:
        if int(case) < 0 or int(case) > 86:
            print(f"Invalid case number {case}. Case number must be in the range of 0 to 86!")
            continue
        for layout in range(3):
            dstPathTemp, log_file_layout = process_layout(layout, qaMode, case, dstPath)

            print("\n\n\n\n")
            print("--------------------------------")
            print("Running a New Functionality...")
            print("--------------------------------")

            for bitDepth in range(7):
                print("\n\n\nRunning New Bit Depth...\n-------------------------\n\n")

                for outputFormatToggle in range(2):
                    # There is no layout toggle for PLN1 case, so skip this case
                    if layout == 2 and outputFormatToggle == 1:
                        continue
                    if case == "8":
                        # Run all variants of noise type functions with additional argument of noiseType = gausssianNoise / shotNoise / saltandpepperNoise
                        for noiseType in range(3):
                            run_performance_test(loggingFolder, log_file_layout, srcPath1, srcPath2, dstPath, bitDepth, outputFormatToggle, case, noiseType, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList)
                    elif case == "21" or case == "23" or case == "24":
                        # Run all variants of interpolation functions with additional argument of interpolationType = bicubic / bilinear / gaussian / nearestneigbor / lanczos / triangular
                        for interpolationType in range(6):
                            run_performance_test(loggingFolder, log_file_layout, srcPath1, srcPath2, dstPath, bitDepth, outputFormatToggle, case, interpolationType, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList)
                    else:
                        run_performance_test(loggingFolder, log_file_layout, srcPath1, srcPath2, dstPath, bitDepth, outputFormatToggle, case, "0", numRuns, testType, layout, qaMode, decoderType, batchSize, roiList)
                    print("------------------------------------------------------------------------------------------")

# print the results of qa tests
supportedCaseList = ['0', '1', '2', '4', '13', '20', '21', '23', '30', '31', '34', '36', '37', '38', '39', '70', '80', '81', '83', '84', '85', '86']
nonQACaseList = ['8', '24', '84']
supportedCases = 0
for num in caseList:
    if qaMode == 1 and num not in nonQACaseList and num in supportedCaseList:
        supportedCases += 1
    elif qaMode == 0 and num in supportedCaseList:
        supportedCases += 1
caseInfo = "Tests are run for " + str(supportedCases) + " supported cases out of the " + str(len(caseList)) + " cases requested"
if qaMode and testType == 0:
    qaFilePath = os.path.join(outFilePath, "QA_results.txt")
    checkFile = os.path.isfile(qaFilePath)
    if checkFile:
        f = open(qaFilePath, 'r+')
        print("---------------------------------- Results of QA Test ----------------------------------\n")
        for line in f:
            sys.stdout.write(line)
            sys.stdout.flush()
        f.write(caseInfo)
print("\n-------------- " + caseInfo + " --------------")

layoutDict = {0:"PKD3", 1:"PLN3", 2:"PLN1"}
# unit tests and QA mode disabled
if testType == 0 and qaMode == 0:
    create_layout_directories(dstPath, layoutDict)
# Performance tests
elif (testType == 1):
    log_file_list = get_log_file_list(preserveOutput)

    functionality_group_list = [
        "color_augmentations",
        "data_exchange_operations",
        "effects_augmentations",
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
        print("\n\nFunctionality\t\t\t\t\t\tFrames Count\tmax(ms/batch)\t\tmin(ms/batch)\t\tavg(ms/batch)\n")
        if len(functions) != 0:
            maxCharLength = len(max(functions, key = len))
            functions = [x + (' ' * (maxCharLength - len(x))) for x in functions]
            for i, func in enumerate(functions):
                print(func + "\t" + str(frames[i]) + "\t\t" + str(maxVals[i]) + "\t" + str(minVals[i]) + "\t" + str(avgVals[i]))
        else:
            print("No variants under this category")

        # Closing log file
        f.close()