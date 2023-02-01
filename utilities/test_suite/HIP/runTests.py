import os
import subprocess
import argparse

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
        raise ValueError(f" path {input_path} does not exist.")
    if not os.path.isdir(input_path):
        raise ValueError(f" path {input_path} is not a directory.")

def rpp_test_suite_parser_and_validator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path1", type = str, default = inFilePath1, help = "Path to the input data")
    parser.add_argument("--input_path2", type = str, default = inFilePath2, help = "Path to the input data")
    parser.add_argument("--case_start", type=str, default="0", help="Testing range starting case # - (0:86)")
    parser.add_argument("--case_end", type=str, default="86", help="Testing range ending case # - (0:86)")
    parser.add_argument('--test_type', type=str, default='0', help="Type of Test - (0 = Unittests / 1 = Performancetests)")
    parser.add_argument('--case_list', nargs="+", help="List of case numbers to list", required=False)
    parser.add_argument('--profiling', type=str, default='NO', help='Run with profiler? - (YES/NO)', required=False)

    args = parser.parse_args()

    if(args.test_type == '0'):
        #os.mkdir(f'{cwd}/OUTPUT_IMAGES_HOST_NEW')
        outFilePath = os.path.join(os.path.dirname(cwd), 'OUTPUT_IMAGES_HIP_NEW')
    else:
        #os.mkdir(f'{cwd}/OUTPUT_PERFORMANCE_LOGS_HOST_NEW')
        outFilePath = os.path.join(os.path.dirname(cwd), 'OUTPUT_PERFORMANCE_LOGS_HIP_NEW')
        
    if (int(args.test_type) == 0):
        numIterations = "1"
    else:
        numIterations = "100"
        
    validate_path(args.input_path1)
    validate_path(args.input_path2)
    if args.case_end < args.case_start:
        print("Ending case# must be greater than starting case#. Aborting!")
        exit(0)

    if args.case_start < "0" or args.case_start > "86":
        print("Starting case# must be in the 0:86 range. Aborting!")
        exit(0)

    if args.case_end < "0" or args.case_end > "86":
        print("Ending case# must be in the 0:86 range. Aborting!")
        exit(0)

    if args.test_type < "0" or args.test_type > "1":
        print("Test Type# must be in the 0:1 range. Aborting!")
        exit(0)

    if args.case_list is not None and int(args.case_start) > 0 and int(args.case_end) <86:
        print("Invalid input! Please provide only 1 option between case_list, case_start and case_end")
        exit(0)

    if args.case_list:
        for case in args.case_list:
            if int(case) < 0 or int(case) > 86:
                print("The case# must be in the 0:86 range!")
                exit(0)

    return parser.parse_args(), outFilePath, numIterations

args, outFilePath, numIterations = rpp_test_suite_parser_and_validator()

if args.case_list is None:
    args.case_list = range(int(args.case_start), int(args.case_end) + 1)
    args.case_list = [str(x) for x in args.case_list]

srcPath1 = args.input_path1
srcPath2 = args.input_path2
dstPath = outFilePath
caseStart = args.case_start
caseEnd = args.case_end
testType = args.test_type
caseList = args.case_list
profilingOption = args.profiling

log_file_list = [
    "../OUTPUT_PERFORMANCE_LOGS_HIP_NEW/Tensor_hip_pkd3_raw_performance_log.txt",
    "../OUTPUT_PERFORMANCE_LOGS_HIP_NEW/Tensor_hip_pln3_raw_performance_log.txt",
    "../OUTPUT_PERFORMANCE_LOGS_HIP_NEW/Tensor_hip_pln1_raw_performance_log.txt"
    ]

functionality_group_list = [
    "color_augmentations",
    "data_exchange_operations",
    "effects_augmentations",
    "filter_augmentations",
    "geometric_augmentations",
    "morphological_operations"
]

if(int(testType) == 0):
    subprocess.call(["./testAllScript.sh", srcPath1, srcPath2, "0", testType, numIterations, " ".join(caseList)])
elif (int(testType) == 1 and profilingOption == "NO"):
    subprocess.call(["./testAllScript.sh", srcPath1, srcPath2, "0", testType, numIterations, " ".join(caseList)])
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
                    frames.append("100")
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
elif (int(testType) == 1 and profilingOption == "YES"):
    subprocess.call(["./testAllScript.sh", "1", testType, numIterations, " ".join(caseList)])
    NEW_FUNC_GROUP_LIST = [0, 15, 20, 29, 36, 40, 42, 49, 56, 65, 69]

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

    RESULTS_DIR = "../OUTPUT_PERFORMANCE_LOGS_HIP_NEW"
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
        os.system('chown $USER:$USER ' + RESULTS_DIR + "/consolidated_results_" + TYPE + ".stats.csv")

    try:
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

    except ImportError:
        print("\nPandas not available! Results of GPU profiling experiment are available in the following files:\n" + \
                CONSOLIDATED_FILE_TENSOR_PKD3 + "\n" + \
                    CONSOLIDATED_FILE_TENSOR_PLN1 + "\n" + \
                        CONSOLIDATED_FILE_TENSOR_PLN3 + "\n")

    except IOError:
        print("Unable to open results in " + RESULTS_DIR + "/consolidated_results_" + TYPE + ".stats.csv")

DST_FOLDER = dstPath
if testType == '0':
    for layout in range(3):
        if layout == 0:
            os.makedirs(f'{DST_FOLDER}/PKD3',  exist_ok=True)
            PKD3_FOLDERS = [f for f in os.listdir(DST_FOLDER) if 'pkd3' in f]
            for TEMP_FOLDER in PKD3_FOLDERS:
                os.rename(f'{DST_FOLDER}/{TEMP_FOLDER}', f'{DST_FOLDER}/PKD3/{TEMP_FOLDER}')
        elif layout == 1:
            os.makedirs(f'{DST_FOLDER}/PLN3',  exist_ok=True)
            PLN3_FOLDERS = [f for f in os.listdir(DST_FOLDER) if 'pln3' in f]
            for TEMP_FOLDER in PLN3_FOLDERS:
                os.rename(f'{DST_FOLDER}/{TEMP_FOLDER}', f'{DST_FOLDER}/PLN3/{TEMP_FOLDER}')
        else:
            os.makedirs(f'{DST_FOLDER}/PLN1',  exist_ok=True)
            PLN1_FOLDERS = [f for f in os.listdir(DST_FOLDER) if 'pln1' in f]
            for TEMP_FOLDER in PLN1_FOLDERS:
                os.rename(f'{DST_FOLDER}/{TEMP_FOLDER}', f'{DST_FOLDER}/PLN1/{TEMP_FOLDER}')