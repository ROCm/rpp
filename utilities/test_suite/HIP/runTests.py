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
outFolderPath = os.getcwd()
buildFolderPath = os.getcwd()
caseMin = 0
caseMax = 90

# Get a list of log files based on a flag for preserving output
def get_log_file_list(preserveOutput):
    return [
        outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HIP_" + timestamp + "/Tensor_hip_pkd3_raw_performance_log.txt",
        outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HIP_" + timestamp + "/Tensor_hip_pln3_raw_performance_log.txt",
        outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HIP_" + timestamp + "/Tensor_hip_pln1_raw_performance_log.txt"
    ]

# Functionality group finder
def func_group_finder(case_number):
    if case_number < 5 or case_number == 13 or case_number == 36 or case_number == 45:
        return "color_augmentations"
    elif case_number == 8 or case_number == 30 or case_number == 82 or case_number == 83 or case_number == 84:
        return "effects_augmentations"
    elif case_number < 40 or case_number == 63:
        return "geometric_augmentations"
    elif case_number < 42:
        return "morphological_operations"
    elif case_number == 49 or case_number == 54:
        return "filter_augmentations"
    elif case_number < 40:
        return "geometric_augmentations"
    elif case_number == 61:
        return "arithmetic_operations"
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

            if case == "40" or case == "41" or case == "49" or case == "54":
                for kernelSize in range(3, 10, 2):
                    print(f"./Tensor_hip {srcPath1} {srcPath2} {dstPath} {bitDepth} {outputFormatToggle} {case} {kernelSize}")
                    result = subprocess.run([buildFolderPath + "/build/Tensor_hip", srcPath1, srcPath2, dstPathTemp, str(bitDepth), str(outputFormatToggle), str(case), str(kernelSize), str(numRuns), str(testType), str(layout), "0", str(qaMode), str(decoderType), str(batchSize)] + roiList + [scriptPath], stdout=subprocess.PIPE)    # nosec
                    print(result.stdout.decode())
            elif case == "8":
                # Run all variants of noise type functions with additional argument of noiseType = gausssianNoise / shotNoise / saltandpepperNoise
                for noiseType in range(3):
                    print(f"./Tensor_hip {srcPath1} {srcPath2} {dstPathTemp} {bitDepth} {outputFormatToggle} {case} {noiseType} ")
                    result = subprocess.run([buildFolderPath + "/build/Tensor_hip", srcPath1, srcPath2, dstPathTemp, str(bitDepth), str(outputFormatToggle), str(case), str(noiseType), str(numRuns), str(testType), str(layout), "0", str(qaMode), str(decoderType), str(batchSize)] + roiList + [scriptPath], stdout=subprocess.PIPE)    # nosec
                    print(result.stdout.decode())
            elif case == "21" or case == "23" or case == "24":
                # Run all variants of interpolation functions with additional argument of interpolationType = bicubic / bilinear / gaussian / nearestneigbor / lanczos / triangular
                for interpolationType in range(6):
                    print(f"./Tensor_hip {srcPath1} {srcPath2} {dstPathTemp} {bitDepth} {outputFormatToggle} {case} {interpolationType}")
                    result = subprocess.run([buildFolderPath + "/build/Tensor_hip", srcPath1, srcPath2, dstPathTemp, str(bitDepth), str(outputFormatToggle), str(case), str(interpolationType), str(numRuns), str(testType), str(layout), "0", str(qaMode), str(decoderType), str(batchSize)] + roiList + [scriptPath], stdout=subprocess.PIPE)    # nosec
                    print(result.stdout.decode())
            else:
                print(f"./Tensor_hip {srcPath1} {srcPath2} {dstPathTemp} {bitDepth} {outputFormatToggle} {case} 0 {numRuns} {testType} {layout}")
                result = subprocess.run([buildFolderPath + "/build/Tensor_hip", srcPath1, srcPath2, dstPathTemp, str(bitDepth), str(outputFormatToggle), str(case), "0", str(numRuns), str(testType), str(layout), "0", str(qaMode), str(decoderType), str(batchSize)] + roiList + [scriptPath], stdout=subprocess.PIPE)    # nosec
                print(result.stdout.decode())

            print("------------------------------------------------------------------------------------------")

def run_performance_test_cmd(loggingFolder, log_file_layout, srcPath1, srcPath2, dstPath, bitDepth, outputFormatToggle, case, additionalParam, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList):
    with open("{}/Tensor_hip_{}_raw_performance_log.txt".format(loggingFolder, log_file_layout), "a") as log_file:
        print(f"./Tensor_hip {srcPath1} {srcPath2} {dstPath} {bitDepth} {outputFormatToggle} {case} {additionalParam} 0 ")
        process = subprocess.Popen([buildFolderPath + "/build/Tensor_hip", srcPath1, srcPath2, dstPath, str(bitDepth), str(outputFormatToggle), str(case), str(additionalParam), str(numRuns), str(testType), str(layout), "0", str(qaMode), str(decoderType), str(batchSize)] + roiList + [scriptPath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)   # nosec
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

    for bitDepth in range(7):
        print("\n\n\nRunning New Bit Depth...\n-------------------------\n\n")

        for outputFormatToggle in range(2):
            # There is no layout toggle for PLN1 case, so skip this case
            if layout == 2 and outputFormatToggle == 1:
                continue

            if case == "40" or case == "41" or case == "49" or case == "54":
                for kernelSize in range(3, 10, 2):
                    run_performance_test_cmd(loggingFolder, log_file_layout, srcPath1, srcPath2, dstPath, bitDepth, outputFormatToggle, case, kernelSize, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList)
            elif case == "8":
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

def run_performance_test_with_profiler(loggingFolder, log_file_layout, srcPath1, srcPath2, dstPath, bitDepth, outputFormatToggle, case, additionalParam, additionalParamType, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList):
    addtionalParamString = additionalParamType + str(additionalParam)
    if layout == 0:
        if not os.path.isdir(f"{dstPath}/Tensor_PKD3/case_{case}"):
            os.mkdir(f"{dstPath}/Tensor_PKD3/case_{case}")
        with open(f"{loggingFolder}/Tensor_hip_{log_file_layout}_raw_performance_log.txt", "a") as log_file:
            print(f'rocprof --basenames on --timestamp on --stats -o {dstPath}/Tensor_PKD3/case_{case}/output_case{case}_bitDepth{bitDepth}_oft{outputFormatToggle}{addtionalParamString}.csv ./Tensor_hip {srcPath1} {srcPath2} {bitDepth} {outputFormatToggle} {case} {additionalParam} 0')
            process = subprocess.Popen(['rocprof', '--basenames', 'on', '--timestamp', 'on', '--stats', '-o', f'{dstPath}/Tensor_PKD3/case_{case}/output_case{case}_bitDepth{bitDepth}_oft{outputFormatToggle}{addtionalParamString}.csv', buildFolderPath + "/build/Tensor_hip", srcPath1, srcPath2, dstPath, str(bitDepth), str(outputFormatToggle), str(case), str(additionalParam), str(numRuns), str(testType), str(layout), '0', str(qaMode), str(decoderType), str(batchSize)] + roiList + [scriptPath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)   # nosec
            while True:
                output = process.stdout.readline()
                if not output and process.poll() is not None:
                    break
                print(output.strip())
                output_str = output.decode('utf-8')
                log_file.write(output_str)
    elif layout == 1:
        if not os.path.isdir(f"{dstPath}/Tensor_PLN3/case_{case}"):
            os.mkdir(f"{dstPath}/Tensor_PLN3/case_{case}")
        with open(f"{loggingFolder}/Tensor_hip_{log_file_layout}_raw_performance_log.txt", "a") as log_file:
            print(f'rocprof --basenames on --timestamp on --stats -o {dstPath}/Tensor_PLN3/case_{case}/output_case{case}_bitDepth{bitDepth}_oft{outputFormatToggle}{addtionalParamString}.csv ./Tensor_hip {srcPath1} {srcPath2} {bitDepth} {outputFormatToggle} {case} {additionalParam} 0')
            process = subprocess.Popen(['rocprof', '--basenames', 'on', '--timestamp', 'on', '--stats', '-o', f'{dstPath}/Tensor_PLN3/case_{case}/output_case{case}_bitDepth{bitDepth}_oft{outputFormatToggle}{addtionalParamString}.csv', buildFolderPath + "/build/Tensor_hip", srcPath1, srcPath2, dstPath, str(bitDepth), str(outputFormatToggle), str(case), str(additionalParam), str(numRuns), str(testType), str(layout), '0', str(qaMode), str(decoderType), str(batchSize)] + roiList + [scriptPath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)    # nosec
            while True:
                output = process.stdout.readline()
                if not output and process.poll() is not None:
                    break
                print(output.strip())
                output_str = output.decode('utf-8')
                log_file.write(output_str)
    elif layout == 2:
        if not os.path.isdir(f"{dstPath}/Tensor_PLN1/case_{case}"):
            os.mkdir(f"{dstPath}/Tensor_PLN1/case_{case}")
        with open(f"{loggingFolder}/Tensor_hip_{log_file_layout}_raw_performance_log.txt", "a") as log_file:
            print(f'rocprof --basenames on --timestamp on --stats -o "{dstPath}/Tensor_PLN1/case_{case}/output_case{case}_bitDepth{bitDepth}_oft{outputFormatToggle}{addtionalParamString}.csv" "./Tensor_hip {srcPath1} {srcPath2} {bitDepth} {outputFormatToggle} {case} {additionalParam} 0"')
            process = subprocess.Popen(['rocprof', '--basenames', 'on', '--timestamp', 'on', '--stats', '-o', f'{dstPath}/Tensor_PLN1/case_{case}/output_case{case}_bitDepth{bitDepth}_oft{outputFormatToggle}{addtionalParamString}.csv', buildFolderPath + "/build/Tensor_hip", srcPath1, srcPath2, dstPath, str(bitDepth), str(outputFormatToggle), str(case), str(additionalParam), str(numRuns), str(testType), str(layout), '0', str(qaMode), str(decoderType), str(batchSize)] + roiList + [scriptPath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)   # nosec
            while True:
                output = process.stdout.readline()
                if not output and process.poll() is not None:
                    break
                print(output.strip())
                output_str = output.decode('utf-8')
                log_file.write(output_str)

# Parse and validate command-line arguments for the RPP test suite
def rpp_test_suite_parser_and_validator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path1", type = str, default = inFilePath1, help = "Path to the input folder 1")
    parser.add_argument("--input_path2", type = str, default = inFilePath2, help = "Path to the input folder 2")
    parser.add_argument("--case_start", type = int, default = caseMin, help = "Testing start case # - Range must be in [" + str(caseMin) + ":" + str(caseMax) + "]")
    parser.add_argument("--case_end", type = int, default = caseMax, help = "Testing end case # - Range must be in [" + str(caseMin) + ":" + str(caseMax) + "]")
    parser.add_argument('--test_type', type = int, default = 0, help = "Type of Test - (0 = Unit tests / 1 = Performance tests)")
    parser.add_argument('--case_list', nargs = "+", help = "List of case numbers to list", required = False)
    parser.add_argument('--profiling', type = str , default = 'NO', help = 'Run with profiler? - (YES/NO)', required = False)
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
    elif args.preserve_output < 0 or args.preserve_output > 1:
        print("Preserve Output must be in the 0/1 (0 = override / 1 = preserve). Aborting")
        exit(0)
    elif args.batch_size <= 0:
        print("Batch size must be greater than 0. Aborting!")
        exit(0)
    elif args.profiling != 'YES' and args.profiling != 'NO':
        print("Profiling option value must be either 'YES' or 'NO'.")
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
profilingOption = args.profiling
qaMode = args.qa_mode
decoderType = args.decoder_type
numRuns = args.num_runs
preserveOutput = args.preserve_output
batchSize = args.batch_size
roiList = ['0', '0', '0', '0'] if args.roi is None else args.roi

if qaMode and batchSize != 3:
    print("QA mode can only run with a batch size of 3.")
    exit(0)

# set the output folders and number of runs based on type of test (unit test / performance test)
if(testType == 0):
    if qaMode:
        outFilePath = outFolderPath + "/QA_RESULTS_HIP_" + timestamp
    else:
        outFilePath = outFolderPath + "/OUTPUT_IMAGES_HIP_" + timestamp
    numRuns = 1
elif(testType == 1):
    if "--num_runs" not in sys.argv:
        numRuns = 100 #default numRuns for running performance tests
    outFilePath = outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HIP_" + timestamp
else:
    print("Invalid TEST_TYPE specified. TEST_TYPE should be 0/1 (0 = Unittests / 1 = Performancetests)")
    exit()

if preserveOutput == 0:
    validate_and_remove_folders(outFolderPath, "OUTPUT_IMAGES_HIP")
    validate_and_remove_folders(outFolderPath, "QA_RESULTS_HIP")
    validate_and_remove_folders(outFolderPath, "OUTPUT_PERFORMANCE_LOGS_HIP")

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
supportedCaseList = ['0', '1', '2', '4', '8', '13', '20', '21', '23', '29', '30', '31', '34', '36', '37', '38', '39', '45', '46', '54', '61', '63', '70', '80', '82', '83', '84', '85', '86', '87', '88', '89', '90']

# Create folders based on testType and profilingOption
if testType == 1 and profilingOption == "YES":
    os.makedirs(f"{dstPath}/Tensor_PKD3")
    os.makedirs(f"{dstPath}/Tensor_PLN1")
    os.makedirs(f"{dstPath}/Tensor_PLN3")

print("\n\n\n\n\n")
print("##########################################################################################")
print("Running all layout Inputs...")
print("##########################################################################################")

if(testType == 0):
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
            dstPathTemp, log_file_layout = process_layout(layout, qaMode, case, dstPath, "hip", func_group_finder)

            if qaMode == 0:
                if not os.path.isdir(dstPathTemp):
                    os.mkdir(dstPathTemp)

            run_unit_test(srcPath1, srcPath2, dstPathTemp, case, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList)

    layoutDict = {0:"PKD3", 1:"PLN3", 2:"PLN1"}
    if qaMode == 0:
        create_layout_directories(dstPath, layoutDict)
else:
    if (testType == 1 and profilingOption == "NO"):
        for case in caseList:
            if case not in supportedCaseList:
                continue
            if case == "82" and "--input_path1" not in sys.argv and "--input_path2" not in sys.argv:
                srcPath1 = ricapInFilePath
                srcPath2 = ricapInFilePath
            for layout in range(3):
                dstPathTemp, log_file_layout = process_layout(layout, qaMode, case, dstPath, "hip", func_group_finder)

                run_performance_test(loggingFolder, log_file_layout, srcPath1, srcPath2, dstPath, case, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList)

    elif (testType == 1 and profilingOption == "YES"):
        NEW_FUNC_GROUP_LIST = [0, 15, 20, 29, 36, 40, 42, 49, 56, 65, 69]

        for case in caseList:
            if case not in supportedCaseList:
                continue
            if case == "82" and "--input_path1" not in sys.argv and "--input_path2" not in sys.argv:
                srcPath1 = ricapInFilePath
                srcPath2 = ricapInFilePath
            for layout in range(3):
                dstPathTemp, log_file_layout = process_layout(layout, qaMode, case, dstPath, "hip", func_group_finder)

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

                        if case == "40" or case == "41" or case == "49" or case == "54":
                            for kernelSize in range(3, 10, 2):
                                run_performance_test_with_profiler(loggingFolder, log_file_layout, srcPath1, srcPath2, dstPath, bitDepth, outputFormatToggle, case, kernelSize, "_kernelSize", numRuns, testType, layout, qaMode, decoderType, batchSize, roiList)
                        elif case == "8":
                            # Run all variants of noise type functions with additional argument of noiseType = gausssianNoise / shotNoise / saltandpepperNoise
                            for noiseType in range(3):
                                run_performance_test_with_profiler(loggingFolder, log_file_layout, srcPath1, srcPath2, dstPath, bitDepth, outputFormatToggle, case, noiseType, "_noiseType", numRuns, testType, layout, qaMode, decoderType, batchSize, roiList)
                        elif case == "21" or case == "23" or case == "24":
                            # Run all variants of interpolation functions with additional argument of interpolationType = bicubic / bilinear / gaussian / nearestneigbor / lanczos / triangular
                            for interpolationType in range(6):
                                run_performance_test_with_profiler(loggingFolder, log_file_layout, srcPath1, srcPath2, dstPath, bitDepth, outputFormatToggle, case, interpolationType, "_interpolationType", numRuns, testType, layout, qaMode, decoderType, batchSize, roiList)
                        else:
                            run_performance_test_with_profiler(loggingFolder, log_file_layout, srcPath1, srcPath2, dstPath, bitDepth, outputFormatToggle, case, "", "", numRuns, testType, layout, qaMode, decoderType, batchSize, roiList)

                        print("------------------------------------------------------------------------------------------")

        RESULTS_DIR = ""
        RESULTS_DIR = outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HIP_" + timestamp
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
                        if (CASE_NUM == "40" or CASE_NUM == "41" or CASE_NUM == "49") and TYPE.startswith("Tensor"):
                            KSIZE_LIST = [3, 5, 7, 9]
                            # Loop through extra param kSize
                            for KSIZE in KSIZE_LIST:
                                # Write into csv file
                                CASE_FILE_PATH = CASE_RESULTS_DIR + "/output_case" + str(CASE_NUM) + "_bitDepth" + str(BIT_DEPTH) + "_oft" + str(OFT) + "_kernelSize" + str(KSIZE) + ".stats.csv"
                                print("CASE_FILE_PATH = " + CASE_FILE_PATH)
                                fileCheck = case_file_check(CASE_FILE_PATH, TYPE, TENSOR_TYPE_LIST, new_file, d_counter)
                                if fileCheck == False:
                                    continue
                        elif (CASE_NUM == "24" or CASE_NUM == "21" or CASE_NUM == "23") and TYPE.startswith("Tensor"):
                            INTERPOLATIONTYPE_LIST = [0, 1, 2, 3, 4, 5]
                            # Loop through extra param interpolationType
                            for INTERPOLATIONTYPE in INTERPOLATIONTYPE_LIST:
                                # Write into csv file
                                CASE_FILE_PATH = CASE_RESULTS_DIR + "/output_case" + str(CASE_NUM) + "_bitDepth" + str(BIT_DEPTH) + "_oft" + str(OFT) + "_interpolationType" + str(INTERPOLATIONTYPE) + ".stats.csv"
                                print("CASE_FILE_PATH = " + CASE_FILE_PATH)
                                fileCheck = case_file_check(CASE_FILE_PATH, TYPE, TENSOR_TYPE_LIST, new_file, d_counter)
                                if fileCheck == False:
                                    continue
                        elif (CASE_NUM == "8") and TYPE.startswith("Tensor"):
                            NOISETYPE_LIST = [0, 1, 2]
                            # Loop through extra param noiseType
                            for NOISETYPE in NOISETYPE_LIST:
                                # Write into csv file
                                CASE_FILE_PATH = CASE_RESULTS_DIR + "/output_case" + str(CASE_NUM) + "_bitDepth" + str(BIT_DEPTH) + "_oft" + str(OFT) + "_noiseType" + str(NOISETYPE) + ".stats.csv"
                                print("CASE_FILE_PATH = " + CASE_FILE_PATH)
                                fileCheck = case_file_check(CASE_FILE_PATH, TYPE, TENSOR_TYPE_LIST, new_file, d_counter)
                                if fileCheck == False:
                                    continue
                        else:
                            # Write into csv file
                            CASE_FILE_PATH = CASE_RESULTS_DIR + "/output_case" + str(CASE_NUM) + "_bitDepth" + str(BIT_DEPTH) + "_oft" + str(OFT) + ".stats.csv"
                            print("CASE_FILE_PATH = " + CASE_FILE_PATH)
                            fileCheck = case_file_check(CASE_FILE_PATH, TYPE, TENSOR_TYPE_LIST, new_file, d_counter)
                            if fileCheck == False:
                                continue

            new_file.close()
            subprocess.call(['chown', '{}:{}'.format(os.getuid(), os.getgid()), RESULTS_DIR + "/consolidated_results_" + TYPE + ".stats.csv"])  # nosec
        try:
            generate_performance_reports(d_counter, TYPE_LIST, RESULTS_DIR)

        except ImportError:
            print("\nPandas not available! Results of GPU profiling experiment are available in the following files:\n" + \
                    CONSOLIDATED_FILE_TENSOR_PKD3 + "\n" + \
                        CONSOLIDATED_FILE_TENSOR_PLN1 + "\n" + \
                            CONSOLIDATED_FILE_TENSOR_PLN3 + "\n")

        except IOError:
            print("Unable to open results in " + RESULTS_DIR + "/consolidated_results_" + TYPE + ".stats.csv")

if (testType == 1 and profilingOption == "NO"):
    log_file_list = get_log_file_list(preserveOutput)

    functionality_group_list = [
    "color_augmentations",
    "data_exchange_operations",
    "effects_augmentations",
    "filter_augmentations",
    "geometric_augmentations",
    "morphological_operations",
    "arithmetic_operations",
    "statistical_operations"
    ]
    for log_file in log_file_list:
        print_performance_tests_summary(log_file, functionality_group_list, numRuns)

# print the results of qa tests
nonQACaseList = ['8', '24', '54', '84'] # Add cases present in supportedCaseList, but without QA support

if qaMode and testType == 0:
    qaFilePath = os.path.join(outFilePath, "QA_results.txt")
    checkFile = os.path.isfile(qaFilePath)
    if checkFile:
        print("---------------------------------- Results of QA Test - Tensor_hip ----------------------------------\n")
        print_qa_tests_summary(qaFilePath, supportedCaseList, nonQACaseList)
