import os
import subprocess
import argparse

cwd = os.getcwd()
inFilePath1 = os.path.join(os.path.dirname(cwd), 'TEST_IMAGES', 'three_images_mixed_src1')
inFilePath2 = os.path.join(os.path.dirname(cwd), 'TEST_IMAGES', 'three_images_mixed_src2')

def validate_path(input_path):
    if not os.path.exists(input_path):
        raise ValueError("path " + input_path + " does not exist.")
    if not os.path.isdir(input_path):
        raise ValueError(" path " + input_path + " is not a directory.")

def rpp_test_suite_parser_and_validator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path1", type = str, default = inFilePath1, help = "Path to the input data")
    parser.add_argument("--input_path2", type = str, default = inFilePath2, help = "Path to the input data")
    parser.add_argument("--case_start", type = int, default = 0, help = "Testing range starting case # - (0:86)")
    parser.add_argument("--case_end", type = int, default = 86, help = "Testing range ending case # - (0:86)")
    parser.add_argument('--test_type', type = int, default = 0, help = "Type of Test - (0 = Unittests / 1 = Performancetests)")
    parser.add_argument('--case_list', nargs = "+", help = "List of case numbers to list", required = False)
    parser.add_argument('--QA_mode', type = int, default = 0, help = " Falg value to dump output buffer into csv files for debugging purposes")
    args = parser.parse_args()

    # check if the folder exists
    validate_path(args.input_path1)
    validate_path(args.input_path2)

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
    elif args.QA_mode < 0 or args.QA_mode > 1:
        print("Debug Flag# must be in the 0 / 1. Aborting!")
        exit(0)
    elif args.case_list is not None and args.case_start > 0 and args.case_end < 86:
        print("Invalid input! Please provide only 1 option between case_list, case_start and case_end")
        exit(0)

    if args.case_list is None:
        args.case_list = range(args.case_start, args.case_end + 1)
        args.case_list = [str(x) for x in args.case_list]
    else:
        for case in args.case_list:
            if int(case) < 0 or int(case) > 86:
                 print("The case# must be in the 0:86 range!")
                 exit(0)

    return args

args = rpp_test_suite_parser_and_validator()
srcPath1 = args.input_path1
srcPath2 = args.input_path2
caseStart = args.case_start
caseEnd = args.case_end
testType = args.test_type
caseList = args.case_list
qaFlag = args.QA_mode

# set the output folders and number of runs based on type of test (unit test / performance test)
if(testType == 0):
    outFilePath = os.path.join(os.path.dirname(cwd), 'OUTPUT_IMAGES_HOST_NEW')
    numIterations = 1
else:
    outFilePath = os.path.join(os.path.dirname(cwd), 'OUTPUT_PERFORMANCE_LOGS_HOST_NEW')
    numIterations = 100
dstPath = outFilePath

# run the shell script
subprocess.call(["./testAllScript.sh", srcPath1, args.input_path2, str(testType), str(numIterations), str(qaFlag), " ".join(caseList)])

layoutDict ={0:"PKD3", 1:"PLN3", 2:"PLN1"}
# unit tests
if testType == 0:
    for layout in range(3):
        currentLayout = layoutDict[layout]
        os.makedirs(dstPath + '/' + currentLayout, exist_ok=True)
        folderList = [f for f in os.listdir(dstPath) if currentLayout.lower() in f]
        for folder in folderList:
            os.rename(dstPath + '/' + folder, dstPath + '/' + currentLayout +  '/' + folder)
# Performance tests
elif (testType == 1):
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
                split_word_end = " " +str(numIterations)
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