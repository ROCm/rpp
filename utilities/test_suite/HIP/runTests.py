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

# Set the timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Set the value of an environment variable
os.environ["TIMESTAMP"] = timestamp

cwd = os.getcwd()
inFilePath1 = os.path.join(os.path.dirname(cwd), 'TEST_IMAGES', 'three_images_mixed_src1')
inFilePath2 = os.path.join(os.path.dirname(cwd), 'TEST_IMAGES', 'three_images_mixed_src2')

def case_file_check(CASE_FILE_PATH):
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

def validate_path(input_path):
    if not os.path.exists(input_path):
        raise ValueError("path " + input_path +" does not exist.")
    if not os.path.isdir(input_path):
        raise ValueError("path " + input_path + " is not a directory.")

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
        "../OUTPUT_PERFORMANCE_LOGS_HIP_" + timestamp + "/Tensor_hip_pkd3_raw_performance_log.txt",
        "../OUTPUT_PERFORMANCE_LOGS_HIP_" + timestamp + "/Tensor_hip_pln3_raw_performance_log.txt",
        "../OUTPUT_PERFORMANCE_LOGS_HIP_" + timestamp + "/Tensor_hip_pln1_raw_performance_log.txt"
    ]

# Functionality group finder
def func_group_finder(case_number):
    if case_number < 5 or case_number == 13 or case_number == 36:
        return "color_augmentations"
    elif case_number == 8 or case_number == 30 or case_number == 83 or case_number == 84:
        return "effects_augmentations"
    elif case_number < 40:
        return "geometric_augmentations"
    elif case_number < 42:
        return "morphological_operations"
    elif case_number == 49:
        return "filter_augmentations"
    elif case_number < 86:
        return "data_exchange_operations"
    else:
        return "miscellaneous"

def generate_performance_reports(d_counter, TYPE_LIST):
    import pandas as pd
    pd.options.display.max_rows = None
    # Generate performance report
    for TYPE in TYPE_LIST:
        print("\n\n\nKernels tested - ", d_counter[TYPE], "\n\n")
        df = pd.read_csv(RESULTS_DIR + "/consolidated_results_" + TYPE + ".stats.csv")
        df["AverageMs"] = df["AverageNs"] / 1000000
        dfPrint = df.drop(['Percentage'], axis=1)
        dfPrint["HIP Kernel Name"] = dfPrint.iloc[:,0].str.lstrip("Hip_")
        dfPrint_noIndices = dfPrint.astype(str)
        dfPrint_noIndices.replace(['0', '0.0'], '', inplace=True)
        dfPrint_noIndices = dfPrint_noIndices.to_string(index=False)
        print(dfPrint_noIndices)

def rpp_test_suite_parser_and_validator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path1", type = str, default = inFilePath1, help = "Path to the input folder 1")
    parser.add_argument("--input_path2", type = str, default = inFilePath2, help = "Path to the input folder 2")
    parser.add_argument("--case_start", type = int, default = 0, help="Testing range starting case # - (0:38)")
    parser.add_argument("--case_end", type = int, default = 38, help="Testing range ending case # - (0:38)")
    parser.add_argument('--test_type', type = int, default = 0, help="Type of Test - (0 = Unit tests / 1 = Performance tests)")
    parser.add_argument('--case_list', nargs = "+", help="List of case numbers to list", required=False)
    parser.add_argument('--profiling', type = str , default='NO', help='Run with profiler? - (YES/NO)', required=False)
    parser.add_argument('--qa_mode', type = int, default = 0, help = "Run with qa_mode? Output images from tests will be compared with golden outputs - (0 / 1)", required = False)
    parser.add_argument('--decoder_type', type = int, default = 0, help = "Type of Decoder to decode the input data - (0 = TurboJPEG / 1 = OpenCV)")
    parser.add_argument('--num_iterations', type = int, default = 0, help = "Specifies the number of iterations for running the performance tests")
    parser.add_argument('--preserve_output', type = int, default = 1, help = "preserves the output of the program - (0 = override output / 1 = preserve output )" )
    args = parser.parse_args()

    # check if the folder exists
    validate_path(args.input_path1)
    validate_path(args.input_path2)

    # validate the parameters passed by user
    if ((args.case_start < 0 or args.case_start > 38) or (args.case_end < 0 or args.case_end > 38)):
        print("Starting case# and Ending case# must be in the 0:38 range. Aborting!")
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
    elif args.case_list is not None and args.case_start > 0 and args.case_end < 38:
        print("Invalid input! Please provide only 1 option between case_list, case_start and case_end")
        exit(0)
    elif args.num_iterations < 0:
        print("Number of Iterations must be greater than 0. Aborting!")
        exit(0)
    elif args.preserve_output < 0 or args.preserve_output > 1:
        print("Preserve Output must be in the 0/1 (0 = override / 1 = preserve). Aborting")
        exit(0)

    if args.case_list is None:
        args.case_list = range(args.case_start, args.case_end + 1)
        args.case_list = [str(x) for x in args.case_list]
    else:
        for case in args.case_list:
            if int(case) < 0 or int(case) > 38:
                 print("The case# must be in the 0:38 range!")
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
profilingOption = args.profiling
qaMode = args.qa_mode
decoderType = args.decoder_type
numIterations = args.num_iterations
preserveOutput = args.preserve_output

if(testType == 0):
    if qaMode:
        outFilePath = os.path.join(os.path.dirname(cwd), 'QA_RESULTS_HIP_' + timestamp)
    else:
        outFilePath = os.path.join(os.path.dirname(cwd), 'OUTPUT_IMAGES_HIP_' + timestamp)
    numIterations = 1
elif(testType == 1):
    if numIterations == 0:
        numIterations = 100 #default numIterations for running performance tests
    outFilePath = os.path.join(os.path.dirname(cwd), 'OUTPUT_PERFORMANCE_LOGS_HIP_' + timestamp)
dstPath = outFilePath

if(testType == 0):
    subprocess.call(["./testAllScript.sh", srcPath1, srcPath2, str(testType), str(numIterations), "0", str(qaMode), str(decoderType), str(preserveOutput), " ".join(caseList)])  # nosec

    layoutDict ={0:"PKD3", 1:"PLN3", 2:"PLN1"}
    if qaMode == 0:
        create_layout_directories(dstPath, layoutDict)
else:
    log_file_list = get_log_file_list(preserveOutput)

    functionality_group_list = [
    "color_augmentations",
    "data_exchange_operations",
    "effects_augmentations",
    "filter_augmentations",
    "geometric_augmentations",
    "morphological_operations"
    ]

    if (testType == 1 and profilingOption == "NO"):
        subprocess.call(["./testAllScript.sh", srcPath1, srcPath2, str(testType), str(numIterations), "0", str(qaMode), str(decoderType), str(preserveOutput), " ".join(caseList)])  # nosec
        for log_file in log_file_list:
            # Opening log file
            try:
                f = open(log_file,"r")
                print("\n\n\nOpened log file -> " + log_file)
            except IOError:
                print("Skipping file -> " + log_file)
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
                    split_word_end = " "+ str(numIterations)
                    prevLine = prevLine.partition(split_word_start)[2].partition(split_word_end)[0]
                    if prevLine not in functions:
                        functions.append(prevLine)
                        frames.append(str(numIterations))
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
            print("Functionalities - " + str(funcCount))

            # Print summary of log
            print("\n\nFunctionality\t\t\t\t\t\tFrames Count\tmax(ms/batch)\t\tmin(ms/batch)\t\tavg(ms/batch)\n")
            if len(functions) != 0:
                maxCharLength = len(max(functions, key=len))
                functions = [x + (' ' * (maxCharLength - len(x))) for x in functions]
                for i, func in enumerate(functions):
                    print(func + "\t" + str(frames[i]) + "\t\t" + str(maxVals[i]) + "\t" + str(minVals[i]) + "\t" + str(avgVals[i]))
            else:
                print("No variants under this category")

            # Closing log file
            f.close()
    elif (testType == 1 and profilingOption == "YES"):
        subprocess.call(["./testAllScript.sh", srcPath1, srcPath2, str(testType), str(numIterations), "1", str(qaMode), str(decoderType), str(preserveOutput), " ".join(caseList)])  # nosec
        NEW_FUNC_GROUP_LIST = [0, 15, 20, 29, 36, 40, 42, 49, 56, 65, 69]

        RESULTS_DIR = ""
        RESULTS_DIR = "../OUTPUT_PERFORMANCE_LOGS_HIP_" + timestamp
        print("RESULTS_DIR = " + RESULTS_DIR)
        CONSOLIDATED_FILE_TENSOR_PKD3 = RESULTS_DIR + "/consolidated_results_Tensor_PKD3.stats.csv"
        CONSOLIDATED_FILE_TENSOR_PLN1 = RESULTS_DIR + "/consolidated_results_Tensor_PLN1.stats.csv"
        CONSOLIDATED_FILE_TENSOR_PLN3 = RESULTS_DIR + "/consolidated_results_Tensor_PLN3.stats.csv"

        TYPE_LIST = ["Tensor_PKD3", "Tensor_PLN1", "Tensor_PLN3"]
        TENSOR_TYPE_LIST = ["Tensor_PKD3", "Tensor_PLN1", "Tensor_PLN3"]
        CASE_NUM_LIST = caseList
        BIT_DEPTH_LIST = range(0, 7, 1)
        OFT_LIST = range(0, 2, 1)
        d_counter = {"Tensor_PKD3":0, "Tensor_PLN1":0, "Tensor_PLN3":0}

        for TYPE in TYPE_LIST:
            # Open csv file
            new_file = open(RESULTS_DIR + "/consolidated_results_" + TYPE + ".stats.csv",'w')
            new_file.write('"HIP Kernel Name","Calls","TotalDurationNs","AverageNs","Percentage"\n')

            prev=""

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
                        if (CASE_NUM == 40 or CASE_NUM == 41 or CASE_NUM == 49) and TYPE.startswith("Tensor"):
                            KSIZE_LIST = [3, 5, 7, 9]
                            # Loop through extra param kSize
                            for KSIZE in KSIZE_LIST:
                                # Write into csv file
                                CASE_FILE_PATH = CASE_RESULTS_DIR + "/output_case" + str(CASE_NUM) + "_bitDepth" + str(BIT_DEPTH) + "_oft" + str(OFT) + "_kSize" + str(KSIZE) + ".stats.csv"
                                print("CASE_FILE_PATH = " + CASE_FILE_PATH)
                                fileCheck = case_file_check(CASE_FILE_PATH)
                                if fileCheck == False:
                                    continue
                        elif (CASE_NUM == 24 or CASE_NUM == 21) and TYPE.startswith("Tensor"):
                            INTERPOLATIONTYPE_LIST = [0, 1, 2, 3, 4, 5]
                            # Loop through extra param interpolationType
                            for INTERPOLATIONTYPE in INTERPOLATIONTYPE_LIST:
                                # Write into csv file
                                CASE_FILE_PATH = CASE_RESULTS_DIR + "/output_case" + str(CASE_NUM) + "_bitDepth" + str(BIT_DEPTH) + "_oft" + str(OFT) + "_interpolationType" + str(INTERPOLATIONTYPE) + ".stats.csv"
                                print("CASE_FILE_PATH = " + CASE_FILE_PATH)
                                fileCheck = case_file_check(CASE_FILE_PATH)
                                if fileCheck == False:
                                    continue
                        elif (CASE_NUM == 8) and TYPE.startswith("Tensor"):
                            NOISETYPE_LIST = [0, 1, 2]
                            # Loop through extra param noiseType
                            for NOISETYPE in NOISETYPE_LIST:
                                # Write into csv file
                                CASE_FILE_PATH = CASE_RESULTS_DIR + "/output_case" + str(CASE_NUM) + "_bitDepth" + str(BIT_DEPTH) + "_oft" + str(OFT) + "_noiseType" + str(NOISETYPE) + ".stats.csv"
                                print("CASE_FILE_PATH = " + CASE_FILE_PATH)
                                fileCheck = case_file_check(CASE_FILE_PATH)
                                if fileCheck == False:
                                    continue
                        else:
                            # Write into csv file
                            CASE_FILE_PATH = CASE_RESULTS_DIR + "/output_case" + str(CASE_NUM) + "_bitDepth" + str(BIT_DEPTH) + "_oft" + str(OFT) + ".stats.csv"
                            print("CASE_FILE_PATH = " + CASE_FILE_PATH)
                            fileCheck = case_file_check(CASE_FILE_PATH)
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
supportedCaseList = ['0', '2', '4', '13', '31', '34', '36', '38']
supportedCases = 0
for num in caseList:
    if num in supportedCaseList:
        supportedCases += 1
caseInfo = "Tests are run for " + str(supportedCases) + " supported cases out of the " + str(len(caseList)) + " cases requested"
if qaMode and testType == 0:
    qaFilePath = os.path.join(outFilePath, "QA_results.txt")
    f = open(qaFilePath, 'r+')
    print("---------------------------------- Results of QA Test ----------------------------------\n")
    for line in f:
        sys.stdout.write(line)
        sys.stdout.flush()
    f.write(caseInfo)
print("\n-------------- " + caseInfo + " --------------")