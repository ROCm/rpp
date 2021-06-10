import subprocess

subprocess.call("./rawLogsGenScript.sh", shell=True)

log_file_list = [
    "../OUTPUT_PERFORMANCE_LOGS_OCL_NEW/BatchPD_ocl_pkd3_ocl_raw_performance_log.txt",
    "../OUTPUT_PERFORMANCE_LOGS_OCL_NEW/BatchPD_ocl_pln3_ocl_raw_performance_log.txt",
    "../OUTPUT_PERFORMANCE_LOGS_OCL_NEW/BatchPD_ocl_pln1_ocl_raw_performance_log.txt"
    ]

for log_file in log_file_list:

    # Opening log file
    f = open(log_file,"r")
    print("\n\n\nOpened log file -> ", log_file)

    stats = []
    maxVals = []
    minVals = []
    avgVals = []
    functions = []
    prevLine = ""

    # Loop over each line
    for line in f:
        if "max,min,avg" in line:
            split_word_start = "Running "
            split_word_end = " 100"
            prevLine = prevLine.partition(split_word_start)[2].partition(split_word_end)[0]
            if prevLine not in functions:
                functions.append(prevLine)
                split_word_start = "max,min,avg = "
                split_word_end = "\n"
                stats = line.partition(split_word_start)[2].partition(split_word_end)[0].split(",")
                maxVals.append(stats[0])
                minVals.append(stats[1])
                avgVals.append(stats[2])

        prevLine = line

    # Print log lengths
    print("Functionalities - ", len(functions))

    # Print summary of log
    print("\n\nFunctionality\t\t\t\t\t\tFrames Count\tmax(s)\t\tmin(s)\t\tavg(s)\n")
    maxCharLength = len(max(functions, key=len))
    functions = [x + (' ' * (maxCharLength - len(x))) for x in functions]
    for i, func in enumerate(functions):
        print(func, "\t100\t\t", maxVals[i], "\t", minVals[i], "\t", avgVals[i])

    # Closing log file
    f.close()