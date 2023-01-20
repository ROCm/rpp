import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--case_start", type=str, default="0", help="Testing range starting case # - (0:86)")
parser.add_argument("--case_end", type=str, default="86", help="Testing range ending case # - (0:86)")
parser.add_argument('--test_type', type=str, default='0', help="Type of Test - (0 = Unittests / 1 = Performancetests)")
parser.add_argument('--case_list', nargs="+", help="List of case numbers to list", required=False)
args = parser.parse_args()

caseStart = args.case_start
caseEnd = args.case_end
testType = args.test_type
caseList = args.case_list
if (int(testType) == 0):
    numIterations = "1"
else:
    numIterations = "100"

if caseEnd < caseStart:
    print("Ending case# must be greater than starting case#. Aborting!")
    exit(0)

if caseStart < "0" or caseStart > "86":
    print("Starting case# must be in the 0:86 range. Aborting!")
    exit(0)

if caseEnd < "0" or caseEnd > "86":
    print("Ending case# must be in the 0:86 range. Aborting!")
    exit(0)

if testType < "0" or testType > "1":
    print("Test Type# must be in the 0:1 range. Aborting!")
    exit(0)

if caseList is not None and caseStart > 0 and caseEnd <86:
    print("Invalid input! Please provide only 1 option between case_list, case_start and case_end")
    exit(0)

if caseList is None:
    caseList = range(int(caseStart), int(caseEnd) + 1)
    caseList = [str(x) for x in caseList]
    subprocess.call(["./testAllScript.sh", testType, numIterations, " ".join(caseList)])
else:
    for case in caseList:
        if int(case) < 0 or int(case) > 86:
            print("The case# must be in the 0:86 range!")
            exit(0)
    subprocess.call(["./testAllScript.sh", testType, numIterations, " ".join(caseList)])

log_file_list = [
    "../OUTPUT_PERFORMANCE_LOGS_HOST_NEW/Tensor_host_pkd3_raw_performance_log.txt",
    "../OUTPUT_PERFORMANCE_LOGS_HOST_NEW/Tensor_host_pln3_raw_performance_log.txt",
    "../OUTPUT_PERFORMANCE_LOGS_HOST_NEW/Tensor_host_pln1_raw_performance_log.txt"
    ]

functionality_group_list = [
    "color_augmentations",
    "data_exchange_operations",
    "effects_augmentations",
    "filter_augmentations",
    "geometric_augmentations",
    "morphological_operations"
]

if(int(testType) == 1):
    for log_file in log_file_list:

        # Opening log file
        try:
            f = open(log_file,"r")
            print("\n\n\nOpened log file -> ", log_file)
        except IOError:
            print("Skipping file -> ", log_file)
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

            if "max,min,avg in ms" in line:
                split_word_start = "Running "
                split_word_end = " 100"
                prevLine = prevLine.partition(split_word_start)[2].partition(split_word_end)[0]
                if prevLine not in functions:
                    functions.append(prevLine)
                    frames.append(numIterations)
                    split_word_start = "max,min,avg in ms = "
                    split_word_end = "\n"
                    stats = line.partition(split_word_start)[2].partition(split_word_end)[0].split(",")
                    maxVals.append(stats[0])
                    minVals.append(stats[1])
                    avgVals.append(stats[2])
                    funcCount += 1

            if line != "\n":
                prevLine = line

        # Print log lengths
        print("Functionalities - ", funcCount)

        # Print summary of log
        print("\n\nFunctionality\t\t\t\t\t\tFrames Count\tmax(ms)\t\tmin(ms)\t\tavg(ms)\n")
        if len(functions) != 0:
            maxCharLength = len(max(functions, key=len))
            functions = [x + (' ' * (maxCharLength - len(x))) for x in functions]
            for i, func in enumerate(functions):
                print(func, "\t", frames[i], "\t\t", maxVals[i], "\t", minVals[i], "\t", avgVals[i])
        else:
            print("No variants under this category")

        # Closing log file
        f.close()