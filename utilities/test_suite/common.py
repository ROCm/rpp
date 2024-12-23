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
import pandas as pd

try:
    from errno import FileExistsError
except ImportError:
    # Python 2 compatibility
    FileExistsError = OSError

imageAugmentationMap = {
    0: ["brightness", "HOST", "HIP"],
    1: ["gamma_correction", "HOST", "HIP"],
    2: ["blend", "HOST", "HIP"],
    4: ["contrast", "HOST", "HIP"],
    5: ["pixelate", "HOST", "HIP"],
    6: ["jitter", "HOST", "HIP"],
    8: ["noise", "HOST", "HIP"],
    13: ["exposure", "HOST", "HIP"],
    20: ["flip", "HOST", "HIP"],
    21: ["resize", "HOST", "HIP"],
    23: ["rotate", "HOST", "HIP"],
    24: ["warp_affine", "HOST", "HIP"],
    26: ["lens_correction", "HOST", "HIP"],
    29: ["water", "HOST", "HIP"],
    30: ["non_linear_blend", "HOST", "HIP"],
    31: ["color_cast", "HOST", "HIP"],
    32: ["erase", "HOST", "HIP"],
    33: ["crop_and_patch", "HOST", "HIP"],
    34: ["lut", "HOST", "HIP"],
    35: ["glitch", "HOST", "HIP"],
    36: ["color_twist", "HOST", "HIP"],
    37: ["crop", "HOST", "HIP"],
    38: ["crop_mirror_normalize", "HOST", "HIP"],
    39: ["resize_crop_mirror", "HOST", "HIP"],
    45: ["color_temperature", "HOST", "HIP"],
    46: ["vignette", "HOST", "HIP"],
    49: ["box_filter", "HIP", "HOST"],
    54: ["gaussian_filter", "HIP"],
    61: ["magnitude", "HOST", "HIP"],
    63: ["phase", "HOST", "HIP"],
    65: ["bitwise_and", "HOST", "HIP"],
    68: ["bitwise_or", "HOST", "HIP"],
    70: ["copy", "HOST", "HIP"],
    79: ["remap", "HOST", "HIP"],
    80: ["resize_mirror_normalize", "HOST", "HIP"],
    81: ["color_jitter", "HOST"],
    82: ["ricap", "HOST", "HIP"],
    83: ["gridmask", "HOST", "HIP"],
    84: ["spatter", "HOST", "HIP"],
    85: ["swap_channels", "HOST", "HIP"],
    86: ["color_to_greyscale", "HOST", "HIP"],
    87: ["tensor_sum", "HOST", "HIP"],
    88: ["tensor_min", "HOST", "HIP"],
    89: ["tensor_max", "HOST", "HIP"],
    90: ["tensor_mean", "HOST", "HIP"],
    91: ["tensor_stddev", "HOST", "HIP"],
    92: ["slice", "HOST", "HIP"]
}

audioAugmentationMap = {
    0: ["non_silent_region_detection", "HOST", "HIP"],
    1: ["to_decibels", "HOST", "HIP"],
    2: ["pre_emphasis_filter", "HOST", "HIP"],
    3: ["down_mixing", "HOST", "HIP"],
    4: ["spectrogram", "HOST", "HIP"],
    5: ["slice", "HOST", "HIP"],
    6: ["resample", "HOST", "HIP"],
    7: ["mel_filter_bank", "HOST", "HIP"]
}

voxelAugmentationMap = {
    0: ["fused_multiply_add_scalar", "HOST", "HIP"],
    1: ["slice", "HOST", "HIP"],
    2: ["add_scalar", "HOST", "HIP"],
    3: ["subtract_scalar", "HOST", "HIP"],
    4: ["flip_voxel", "HOST", "HIP"],
    5: ["multiply_scalar", "HOST", "HIP"],
    6: ["gaussian_noise_voxel", "HOST", "HIP"]
}

miscAugmentationMap  = {
    0: ["transpose","HOST", "HIP"],
    1: ["normalize", "HOST", "HIP"],
    2: ["log", "HOST", "HIP"],
    3: ["log1p","HOST","HIP"]
}

ImageAugmentationGroupMap = {
    "color_augmentations" : [0, 1, 2, 3, 4, 13, 31, 34, 36, 45, 81],
    "effects_augmentations" : [5, 6, 8, 29, 30, 32, 35, 46, 82, 83, 84],
    "geometric_augmentations" : [20, 21, 23, 24, 26, 33, 37, 38, 39, 63, 79, 80, 92],
    "filter_augmentations" : [49, 54],
    "arithmetic_operations" : [61],
    "logical_operations" : [65, 68],
    "data_exchange_operations" : [70, 85, 86],
    "statistical_operations" : [87, 88, 89, 90, 91]
}

# Checks if the folder path is empty, or is it a root folder, or if it exists, and remove its contents
def validate_and_remove_files(path):
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

# Check if a case file exists and filter its contents based on certain conditions
def case_file_check(CASE_FILE_PATH, TYPE, TENSOR_TYPE_LIST, new_file, d_counter):
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
def directory_name_generator(qaMode, affinity, layoutType, case, path, func_group_finder):
    if qaMode == 0:
        functionality_group = func_group_finder(int(case))
        dst_folder_temp = path + "/rpp_" + affinity + "_" + layoutType + "_" + functionality_group
    else:
        dst_folder_temp = path

    return dst_folder_temp

# Process the layout based on the given parameters and generate the directory name and log file layout.
def process_layout(layout, qaMode, case, dstPath, backend, func_group_finder):
    if layout == 0:
        dstPathTemp = directory_name_generator(qaMode, backend, "pkd3", case, dstPath, func_group_finder)
        log_file_layout = "pkd3"
    elif layout == 1:
        dstPathTemp = directory_name_generator(qaMode, backend, "pln3", case, dstPath, func_group_finder)
        log_file_layout = "pln3"
    elif layout == 2:
        dstPathTemp = directory_name_generator(qaMode, backend, "pln1", case, dstPath, func_group_finder)
        log_file_layout = "pln1"

    return dstPathTemp, log_file_layout

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

# Read data from the logs generated from rocprof, process the data
# and generate performance reports based on counters and a list of types
def generate_performance_reports(d_counter, TYPE_LIST, RESULTS_DIR):
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

# Read the data from QA logs, process the data and print the results as a summary
def print_qa_tests_summary(qaFilePath, supportedCaseList, nonQACaseList, fileName):
    f = open(qaFilePath, 'r+')
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
    print("\n---------------------------------- Summary of QA Test - " + fileName + " ----------------------------------" + resultsInfo + "\n\n-------------------------------------------------------------------")

# Read the data from performance logs, process the data and print the results as a summary
def print_performance_tests_summary(logFile, functionalityGroupList, numRuns):
    try:
        f = open(logFile, "r")
        print("\nOpened log file -> " + logFile)
    except IOError:
        print("Skipping file -> " + logFile)
        return

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
        for functionalityGroup in functionalityGroupList:
            if functionalityGroup in line:
                functions.extend([" ", functionalityGroup, " "])
                frames.extend([" ", " ", " "])
                maxVals.extend([" ", " ", " "])
                minVals.extend([" ", " ", " "])
                avgVals.extend([" ", " ", " "])

        if "max,min,avg wall times in ms/batch" in line:
            splitWordStart = "Running "
            splitWordEnd = " " + str(numRuns)
            prevLine = prevLine.partition(splitWordStart)[2].partition(splitWordEnd)[0]
            if prevLine not in functions:
                functions.append(prevLine)
                frames.append(numRuns)
                splitWordStart = "max,min,avg wall times in ms/batch = "
                splitWordEnd = "\n"
                stats = line.partition(splitWordStart)[2].partition(splitWordEnd)[0].split(",")
                maxVals.append(stats[0])
                minVals.append(stats[1])
                avgVals.append(stats[2])
                funcCount += 1

        if line != "\n":
            prevLine = line

    # Print log lengths
    print("Functionalities - " + str(funcCount))

    # Print summary of log
    headerFormat = "{:<70} {:<15} {:<15} {:<15} {:<15}"
    rowFormat = "{:<70} {:<15} {:<15} {:<15} {:<15}"
    print("\n" + headerFormat.format("Functionality", "Frames Count", "max(ms/batch)", "min(ms/batch)", "avg(ms/batch)") + "\n")
    if len(functions) != 0:
        for i, func in enumerate(functions):
            print(rowFormat.format(func, str(frames[i]), str(maxVals[i]), str(minVals[i]), str(avgVals[i])))
    else:
        print("No variants under this category")

    # Closing log file
    f.close()

# Read the standard output from subprocess and writes to log file
def read_from_subprocess_and_write_to_log(process, logFile):
    while True:
        output = process.stdout.readline()
        if not output and process.poll() is not None:
            break
        output = output.decode().strip()  # Decode bytes to string and strip extra whitespace
        if output:
            print(output)
            logFile.write(output + '\n')

# Returns the layout name based on layout value
def get_layout_name(layout):
    if layout == 0:
        return "PKD3"
    elif  layout == 1:
        return "PLN3"
    elif layout == 2:
        return "PLN1"

# Prints entire case list if user asks for help
def print_case_list(imageAugmentationMap, backendType, parser):
    if '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
        print("\n" + "="*30)
        print("Functionality Reference List")
        print("="*30 + "\n")
        headerFormat = "{:<12} {:<15}"
        print(headerFormat.format("CaseNumber", "Functionality"))
        print("-" * 27)
        rowFormat = "{:<12} {:<15}"
        for key, value_list in imageAugmentationMap.items():
            if backendType in value_list:
                print(rowFormat.format(key, value_list[0]))

        sys.exit(0)

# Functionality group finder
def func_group_finder(case_number):
    for key, value in ImageAugmentationGroupMap.items():
        if case_number in value:
            return key
    return "miscellaneous"

def dataframe_to_markdown(df):
    # Calculate the maximum width of each column
    column_widths = {}
    for col in df.columns:
        max_length = len(col)
        for value in df[col]:
            max_length = max(max_length, len(str(value)))
        column_widths[col] = max_length

    # Create the header row
    md = '| ' + ' | '.join([col.ljust(column_widths[col]) for col in df.columns]) + ' |\n'
    md += '| ' + ' | '.join(['-' * column_widths[col] for col in df.columns]) + ' |\n'

    # Create the data rows
    for i, row in df.iterrows():
        md += '| ' + ' | '.join([str(value).ljust(column_widths[df.columns[j]]) for j, value in enumerate(row.values)]) + ' |\n'

    return md
