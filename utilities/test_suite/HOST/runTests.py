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
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.join(os.path.dirname( __file__ ), '..' ))
from common import *

# Set the timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

scriptPath = os.path.dirname(os.path.realpath(__file__))
inFilePath1 = scriptPath + "/../TEST_IMAGES/three_images_mixed_src1"
inFilePath2 = scriptPath + "/../TEST_IMAGES/three_images_mixed_src2"
ricapInFilePath = scriptPath + "/../TEST_IMAGES/three_images_150x150_src1"
qaInputFile = scriptPath + "/../TEST_IMAGES/three_images_mixed_src1"
perfQaInputFile = scriptPath + "/../TEST_IMAGES/eight_images_mixed_src1"
outFolderPath = os.getcwd()
buildFolderPath = os.getcwd()
caseMin = 0
caseMax = 89

# Get a list of log files based on a flag for preserving output
def get_log_file_list(preserveOutput):
    return [
        outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HOST_" + timestamp + "/Tensor_host_pkd3_raw_performance_log.txt",
        outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HOST_" + timestamp + "/Tensor_host_pln3_raw_performance_log.txt",
        outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HOST_" + timestamp + "/Tensor_host_pln1_raw_performance_log.txt"
    ]

# Functionality group finder
def func_group_finder(case_number):
    if case_number < 5 or case_number == 13 or case_number == 36 or case_number == 31 or case_number == 45:
        return "color_augmentations"
    elif case_number == 8 or case_number == 30 or case_number == 82 or case_number == 83 or case_number == 84:
        return "effects_augmentations"
    elif case_number < 40 or case_number == 63:
        return "geometric_augmentations"
    elif case_number < 62:
        return "arithmetic_operations"
    elif case_number < 69:
        return "logical_operations"
    elif case_number < 87:
        return "data_exchange_operations"
    elif case_number < 88:
        return "statistical_operations"
    else:
        return "miscellaneous"

def run_unit_test(srcPath1, srcPath2, dstPathTemp, case, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList):
    print("\n\n\n\n")
    print("--------------------------------")
    print("Running a New Functionality...")
    print("--------------------------------")
    bitDepths = range(7)
    outputFormatToggles = [0, 1]
    if qaMode:
        bitDepths = [0]
        outputFormatToggles = [0]
    for bitDepth in bitDepths:
        print("\n\n\nRunning New Bit Depth...\n-------------------------\n\n")

        for outputFormatToggle in outputFormatToggles:
            # There is no layout toggle for PLN1 case, so skip this case
            if layout == 2 and outputFormatToggle == 1:
                continue

            if case == "8":
                # Run all variants of noise type functions with additional argument of noiseType = gausssianNoise / shotNoise / saltandpepperNoise
                for noiseType in range(3):
                    print(f"./Tensor_host {srcPath1} {srcPath2} {dstPathTemp} {bitDepth} {outputFormatToggle} {case} {noiseType} 0 ")
                    result = subprocess.run([buildFolderPath + "/build/Tensor_host", srcPath1, srcPath2, dstPathTemp, str(bitDepth), str(outputFormatToggle), str(case), str(noiseType), str(numRuns), str(testType), str(layout), "0", str(qaMode), str(decoderType), str(batchSize)] + roiList + [scriptPath], stdout=subprocess.PIPE)    # nosec
                    print(result.stdout.decode())
            elif case == "21" or case == "23" or case == "24":
                # Run all variants of interpolation functions with additional argument of interpolationType = bicubic / bilinear / gaussian / nearestneigbor / lanczos / triangular
                for interpolationType in range(6):
                    print(f"./Tensor_host {srcPath1} {srcPath2} {dstPathTemp} {bitDepth} {outputFormatToggle} {case} {interpolationType} 0")
                    result = subprocess.run([buildFolderPath + "/build/Tensor_host", srcPath1, srcPath2, dstPathTemp, str(bitDepth), str(outputFormatToggle), str(case), str(interpolationType), str(numRuns), str(testType), str(layout), "0", str(qaMode), str(decoderType), str(batchSize)] + roiList + [scriptPath], stdout=subprocess.PIPE)    # nosec
                    print(result.stdout.decode())
            else:
                print(f"./Tensor_host {srcPath1} {srcPath2} {dstPathTemp} {bitDepth} {outputFormatToggle} {case} 0 {numRuns} {testType} {layout} 0")
                result = subprocess.run([buildFolderPath + "/build/Tensor_host", srcPath1, srcPath2, dstPathTemp, str(bitDepth), str(outputFormatToggle), str(case), "0", str(numRuns), str(testType), str(layout), "0", str(qaMode), str(decoderType), str(batchSize)] + roiList + [scriptPath], stdout=subprocess.PIPE)    # nosec
                print(result.stdout.decode())

            print("------------------------------------------------------------------------------------------")

def run_performance_test_cmd(loggingFolder, log_file_layout, srcPath1, srcPath2, dstPath, bitDepth, outputFormatToggle, case, additionalParam, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList):
    if qaMode == 1:
        with open("{}/BatchPD_host_{}_raw_performance_log.txt".format(loggingFolder, log_file_layout), "a") as log_file:
            process = subprocess.Popen([buildFolderPath + "/build/BatchPD_host_" + log_file_layout, srcPath1, srcPath2, str(bitDepth), str(outputFormatToggle), str(case), str(additionalParam), "0"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)    # nosec
            while True:
                output = process.stdout.readline()
                if not output and process.poll() is not None:
                    break
                print(output.strip())
                log_file.write(output)

    with open("{}/Tensor_host_{}_raw_performance_log.txt".format(loggingFolder, log_file_layout), "a") as log_file:
        print(f"./Tensor_host {srcPath1} {srcPath2} {dstPath} {bitDepth} {outputFormatToggle} {case} {additionalParam} 0 ")
        process = subprocess.Popen([buildFolderPath + "/build/Tensor_host", srcPath1, srcPath2, dstPath, str(bitDepth), str(outputFormatToggle), str(case), str(additionalParam), str(numRuns), str(testType), str(layout), "0", str(qaMode), str(decoderType), str(batchSize)] + roiList + [scriptPath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)    # nosec
        while True:
            output = process.stdout.readline()
            if not output and process.poll() is not None:
                break
            print(output.strip())
            log_file.write(output)

def run_performance_test(loggingFolder, log_file_layout, srcPath1, srcPath2, dstPath, case, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList):
    print("\n\n\n\n")
    print("--------------------------------")
    print("Running a New Functionality...")
    print("--------------------------------")
    bitDepths = range(7)
    if qaMode:
        bitDepths = [0]
    for bitDepth in bitDepths:
        print("\n\n\nRunning New Bit Depth...\n-------------------------\n\n")

        for outputFormatToggle in range(2):
            # There is no layout toggle for PLN1 case, so skip this case
            if layout == 2 and outputFormatToggle == 1:
                continue
            if case == "8":
                # Run all variants of noise type functions with additional argument of noiseType = gausssianNoise / shotNoise / saltandpepperNoise
                for noiseType in range(3):
                    run_performance_test_cmd(loggingFolder, log_file_layout, srcPath1, srcPath2, dstPath, bitDepth, outputFormatToggle, case, noiseType, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList)
            elif case == "21" or case == "23" or case == "24":
                # Run all variants of interpolation functions with additional argument of interpolationType = bicubic / bilinear / gaussian / nearestneigbor / lanczos / triangular
                for interpolationType in range(6):
                    run_performance_test_cmd(loggingFolder, log_file_layout, srcPath1, srcPath2, dstPath, bitDepth, outputFormatToggle, case, interpolationType, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList)
            else:
                run_performance_test_cmd(loggingFolder, log_file_layout, srcPath1, srcPath2, dstPath, bitDepth, outputFormatToggle, case, "0", numRuns, testType, layout, qaMode, decoderType, batchSize, roiList)
            print("------------------------------------------------------------------------------------------")

# Parse and validate command-line arguments for the RPP test suite
def rpp_test_suite_parser_and_validator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path1", type = str, default = inFilePath1, help = "Path to the input folder 1")
    parser.add_argument("--input_path2", type = str, default = inFilePath2, help = "Path to the input folder 2")
    parser.add_argument("--case_start", type = int, default = caseMin, help = "Testing start case # - Range must be in [" + str(caseMin) + ":" + str(caseMax) + "]")
    parser.add_argument("--case_end", type = int, default = caseMax, help = "Testing end case # - Range must be in [" + str(caseMin) + ":" + str(caseMax) + "]")
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
    validate_path(perfQaInputFile)

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
    elif args.qa_mode < 0 or args.qa_mode > 1:
        print("QA mode must be in the 0 / 1. Aborting!")
        exit(0)
    elif args.decoder_type < 0 or args.decoder_type > 1:
        print("Decoder Type must be in the 0/1 (0 = OpenCV / 1 = TurboJPEG). Aborting")
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
            if int(case) < caseMin or int(case) > caseMax:
                print(f"Invalid case number {case}! Case number must be in the {caseMin}:{caseMax} range. Aborting!")
                exit(0)

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

if qaMode and testType == 0 and batchSize != 3:
    print("QA mode can only run with a batch size of 3.")
    exit(0)

if qaMode and testType == 1 and batchSize != 8:
    print("Performance QA mode can only run with a batch size of 8.")
    exit(0)

# set the output folders and number of runs based on type of test (unit test / performance test)
if(testType == 0):
    if qaMode:
        outFilePath = outFolderPath + "/QA_RESULTS_HOST_" + timestamp
    else:
        outFilePath = outFolderPath + "/OUTPUT_IMAGES_HOST_" + timestamp
    numRuns = 1
elif(testType == 1):
    if "--num_runs" not in sys.argv:
        numRuns = 100 #default numRuns for running performance tests
    outFilePath = outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HOST_" + timestamp
else:
    print("Invalid TEST_TYPE specified. TEST_TYPE should be 0/1 (0 = Unittests / 1 = Performancetests)")
    exit()

if preserveOutput == 0:
    validate_and_remove_folders(outFolderPath, "OUTPUT_IMAGES_HOST")
    validate_and_remove_folders(outFolderPath, "QA_RESULTS_HOST")
    validate_and_remove_folders(outFolderPath, "OUTPUT_PERFORMANCE_LOGS_HOST")

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

# List of cases supported
supportedCaseList = ['0', '1', '2', '4', '8', '13', '20', '21', '23', '29', '30', '31', '33', '34', '36', '37', '38', '39', '45', '46', '54', '61', '63', '65', '68', '70', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89']

print("\n\n\n\n\n")
print("##########################################################################################")
print("Running all layout Inputs...")
print("##########################################################################################")

if testType == 0:
    for case in caseList:
        if case not in supportedCaseList:
            continue
        if case == "82" and (("--input_path1" not in sys.argv and "--input_path2" not in sys.argv) or qaMode == 1):
            srcPath1 = ricapInFilePath
            srcPath2 = ricapInFilePath
        # if QA mode is enabled overwrite the input folders with the folders used for generating golden outputs
        if qaMode == 1 and case != "82":
            srcPath1 = inFilePath1
            srcPath2 = inFilePath2
        for layout in range(3):
            dstPathTemp, log_file_layout = process_layout(layout, qaMode, case, dstPath, "host", func_group_finder)

            if qaMode == 0:
                if not os.path.isdir(dstPathTemp):
                    os.mkdir(dstPathTemp)

            run_unit_test(srcPath1, srcPath2, dstPathTemp, case, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList)
    layoutDict = {0:"PKD3", 1:"PLN3", 2:"PLN1"}
    if qaMode == 0:
        create_layout_directories(dstPath, layoutDict)
else:
    for case in caseList:
        if case not in supportedCaseList:
            continue
        # if QA mode is enabled overwrite the input folders with the folders used for generating golden outputs
        if qaMode == 1 and case != "82":
            srcPath1 = inFilePath1
            srcPath2 = inFilePath2
        if case == "82" and "--input_path1" not in sys.argv and "--input_path2" not in sys.argv:
            srcPath1 = ricapInFilePath
            srcPath2 = ricapInFilePath
        for layout in range(3):
            dstPathTemp, log_file_layout = process_layout(layout, qaMode, case, dstPath, "host", func_group_finder)
            run_performance_test(loggingFolder, log_file_layout, srcPath1, srcPath2, dstPath, case, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList)

# print the results of qa tests
nonQACaseList = ['8', '24', '54', '84'] # Add cases present in supportedCaseList, but without QA support

if qaMode and testType == 0:
    qaFilePath = os.path.join(outFilePath, "QA_results.txt")
    checkFile = os.path.isfile(qaFilePath)
    if checkFile:
        print("---------------------------------- Results of QA Test - Tensor_host ----------------------------------\n")
        print_qa_tests_summary(qaFilePath, supportedCaseList, nonQACaseList)

layoutDict = {0:"PKD3", 1:"PLN3", 2:"PLN1"}
# unit tests and QA mode disabled
if testType == 0 and qaMode == 0:
    create_layout_directories(dstPath, layoutDict)
# Performance tests
elif (testType == 1 and qaMode == 1):
    columns = ['BatchPD_Augmentation_Type', 'Tensor_Augmentation_Type', 'Performance Speedup (%)', 'Test_Result']
    tensorAugVariations = []
    batchPDAugVariations = []
    achievedPerf = []
    status = []
    df = pd.DataFrame(columns=columns)
    tensorLogFileList = get_log_file_list(preserveOutput)
    batchpdLogFileList = [sub.replace("Tensor_host", "BatchPD_host") for sub in tensorLogFileList] # will be needed only in qa mode

    stats = []
    tensorVal = []
    batchpdVal = []
    functions = []
    functionsBatchPD = []
    funcCount = 0
    performanceNoise = 10
    perfQASupportCaseList = ["resize", "color_twist", "phase"]
    for i in range(3):
        tensorLogFile = tensorLogFileList[i]
        batchpdLogFile = batchpdLogFileList[i]
        # Opening log file
        try:
            tensorFile = open(tensorLogFile,"r")
        except IOError:
            print("Skipping file -> "+ tensorLogFile)
            continue

        # Opening log file
        try:
            batchpdFile = open(batchpdLogFile,"r")
        except IOError:
            print("Skipping file -> "+ batchpdLogFile)
            continue

        prevLine = ""
        # Loop over each line
        for line in tensorFile:
            if "max,min,avg wall times in ms/batch" in line and "u8_Tensor" in prevLine:
                layoutCheck = "PKD3_toPKD3" in prevLine or "PLN3_toPLN3" in prevLine or "PLN1_toPLN1" in prevLine
                interpolationCheck = "interpolationType" not in prevLine or "interpolationTypeBilinear" in prevLine
                if layoutCheck and interpolationCheck:
                    splitWordStart = "Running "
                    splitWordEnd = " " + str(numRuns)
                    prevLine = prevLine.partition(splitWordStart)[2].partition(splitWordEnd)[0]
                    splitWordStart = "max,min,avg wall times in ms/batch = "
                    splitWordEnd = "\n"
                    if prevLine not in functions:
                        functions.append(prevLine)
                        stats = line.partition(splitWordStart)[2].partition(splitWordEnd)[0].split(",")
                        tensorVal.append(float(stats[2]))
                        funcCount += 1

            if line != "\n":
                prevLine = line

        # Closing log file
        tensorFile.close()

        stats = []
        prevLine = ""
        for line in batchpdFile:
            if "max,min,avg" in line and "u8_BatchPD" in prevLine:
                if "PKD3_toPKD3" in prevLine or "PLN3_toPLN3" in prevLine or "PLN1_toPLN1" in prevLine:
                    splitWordStart = "Running "
                    splitWordEnd = " " + str(numRuns)
                    prevLine = prevLine.partition(splitWordStart)[2].partition(splitWordEnd)[0]
                    splitWordStart = "max,min,avg"
                    splitWordEnd = "\n"
                    if prevLine not in functionsBatchPD:
                        functionsBatchPD.append(prevLine)
                        stats = line.partition(splitWordStart)[2].partition(splitWordEnd)[0].split(",")
                        batchpdVal.append(float(stats[2]) * float(1000.0))

            if line != "\n":
                prevLine = line

        # Closing log file
        batchpdFile.close()

    print("---------------------------------- Results of QA Test - Tensor_host ----------------------------------\n")
    qaFilePath = os.path.join(outFilePath, "QA_results.txt")
    excelFilePath = os.path.join(outFilePath, "performance_qa_results.xlsx")
    f = open(qaFilePath, 'w')
    numLines = 0
    numPassed = 0
    removalList = ["_HOST", "_toPKD3", "_toPLN3", "_toPLN1"]
    for i in range(len(functions)):
        perfImprovement = int(((batchpdVal[i] - tensorVal[i]) / batchpdVal[i]) * 100)
        numLines += 1
        funcName = functions[i]
        caseName = funcName.split("_u8_")[0]
        for string in removalList:
            funcName = funcName.replace(string, "")
        if caseName not in perfQASupportCaseList:
            print("Error! QA mode is not yet available for variant: " + funcName)
            continue
        achievedPerf.append(perfImprovement)
        tensorAugVariations.append(funcName)
        if perfImprovement > -performanceNoise:
            numPassed += 1
            status.append("PASSED")
            print(funcName + ": PASSED")
        else:
            status.append("FAILED")
            print(funcName + ": FAILED")

    resultsInfo = "\n\nFinal Results of Tests:"
    resultsInfo += "\n    - Total test cases including all subvariants REQUESTED = " + str(numLines)
    resultsInfo += "\n    - Total test cases including all subvariants PASSED = " + str(numPassed)
    f.write(resultsInfo)
    batchPDAugVariations = [s.replace('Tensor', 'BatchPD') for s in tensorAugVariations]
    df['Tensor_Augmentation_Type'] = tensorAugVariations
    df['BatchPD_Augmentation_Type'] = batchPDAugVariations
    df['Performance Speedup (%)'] = achievedPerf
    df['Test_Result'] = status
    # Calculate the number of cases passed and failed
    passedCases = df['Test_Result'].eq('PASSED').sum()
    failedCases = df['Test_Result'].eq('FAILED').sum()

    summary_row = {'BatchPD_Augmentation_Type': pd.NA,
                   'Tensor_Augmentation_Type': pd.NA,
                   'Performance Speedup (%)': pd.NA,
                   'Test_Result': f'Final Results of Tests: Passed: {passedCases}, Failed: {failedCases}'}

    print("\n", df.to_markdown())

    # Append the summary row to the DataFrame
    # Convert the dictionary to a DataFrame
    summary_row = pd.DataFrame([summary_row])
    df = pd.concat([df, summary_row], ignore_index=True)

    df.to_excel(excelFilePath, index=False)
    print("\n-------------------------------------------------------------------" + resultsInfo + "\n\n-------------------------------------------------------------------")
    print("\nIMPORTANT NOTE:")
    print("- The following performance comparison shows Performance Speedup percentages between times measured on previous generation RPP-BatchPD APIs against current generation RPP-Tensor APIs.")
    print(f"- All APIs have been improved for performance ranging from {0}% (almost same) to {100}% faster.")
    print("- Random observations of negative speedups might always occur due to current test machine temperature/load variances or other CPU/GPU state-dependent conditions.")
    print("\n-------------------------------------------------------------------\n")
elif (testType == 1 and qaMode == 0):
    log_file_list = get_log_file_list(preserveOutput)

    functionality_group_list = [
        "color_augmentations",
        "data_exchange_operations",
        "effects_augmentations",
        "geometric_augmentations",
        "arithmetic_operations",
        "statistical_operations",
    ]

    for log_file in log_file_list:
        print_performance_tests_summary(log_file, functionality_group_list, numRuns)
