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
        raise ValueError("path " + input_path +" does not exist.")
    if not os.path.isdir(input_path):
        raise ValueError("path " + input_path + " is not a directory.")

def rpp_test_suite_parser_and_validator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path1", type = str, default = inFilePath1, help = "Path to the input folder 1")
    parser.add_argument("--input_path2", type = str, default = inFilePath2, help = "Path to the input folder 2")
    parser.add_argument("--case_start", type = int, default = 0, help="Testing range starting case # - (0:38)")
    parser.add_argument("--case_end", type = int, default = 38, help="Testing range ending case # - (0:38)")
    parser.add_argument('--test_type', type = int, default = 0, help="Type of Test - (0 = Unittests / 1 = Performancetests)")
    parser.add_argument('--case_list', nargs = "+", help="List of case numbers to list", required=False)
    parser.add_argument('--profiling', type = str , default='NO', help='Run with profiler? - (YES/NO)', required=False)
    parser.add_argument('--qa_mode', type = int, default = 0, help = "Run with qa_mode? Outputs images from tests will be compared with golden outputs - (0 / 1)", required = False)
    parser.add_argument('--decoder_type', type = int, default = 0, help = "Type of Decoder to decode the input data - (0 = TurboJPEG / 1 = OpenCV)")
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

    if args.case_list is None:
        args.case_list = range(args.case_start, args.case_end + 1)
        args.case_list = [str(x) for x in args.case_list]
    else:
        for case in args.case_list:
            if int(case) < 0 or int(case) > 38:
                 print("The case# must be in the 0:38 range!")
                 exit(0)

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

if(testType == 0):
    outFilePath = os.path.join(os.path.dirname(cwd), 'OUTPUT_IMAGES_HIP_NEW')
    numIterations = 1
else:
    outFilePath = os.path.join(os.path.dirname(cwd), 'OUTPUT_PERFORMANCE_LOGS_HIP_NEW')
    numIterations = 100
dstPath = outFilePath

if(testType == 0):
    subprocess.call(["./testAllScript.sh", srcPath1, srcPath2, str(testType), str(numIterations), "0", str(qaMode), str(decoderType), " ".join(caseList)])

    # print the results of qa tests
    if qaMode:
        qaFilePath = os.path.join(outFilePath, "QA_results.txt")
        f = open(qaFilePath, 'r')
        print("---------------------------------- Results of QA Test ----------------------------------\n")
        for line in f:
            print(line)

    layoutDict ={0:"PKD3", 1:"PLN3", 2:"PLN1"}

    for layout in range(3):
        currentLayout = layoutDict[layout]
        try:
            os.makedirs(dstPath + '/' + currentLayout)
        except FileExistsError:
            pass
        folderList = [f for f in os.listdir(dstPath) if currentLayout.lower() in f]
        for folder in folderList:
            os.rename(dstPath + '/' + folder, dstPath + '/' + currentLayout +  '/' + folder)
else:
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

    if (testType == 1 and profilingOption == "NO"):
        subprocess.call(["./testAllScript.sh", srcPath1, srcPath2, str(testType), str(numIterations), "0", str(qaMode), str(decoderType), " ".join(caseList)])
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
                    split_word_end = " "+ str(numIterations)
                    prevLine = prevLine.partition(split_word_start)[2].partition(split_word_end)[0]
                    if prevLine not in functions:
                        functions.append(prevLine)
                        frames.append(str(numIterations))
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
    elif (testType == 1 and profilingOption == "YES"):
        subprocess.call(["./testAllScript.sh", srcPath1, srcPath2, str(testType), str(numIterations), "1", str(qaMode), str(decoderType), " ".join(caseList)])
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