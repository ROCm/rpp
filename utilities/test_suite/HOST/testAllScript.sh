#!/bin/bash

cwd=$(pwd)

# <<<<<<<<<<<<<< VALIDATION CHECK FOR FOLDER PATHS >>>>>>>>>>>>>>>>>>>>>>>>>>>>
function VALIDATE_PATH {
    if [ -z "$1" ]; then  #check if a string is empty
        echo "$1 Folder path is empty."
        exit
    fi
    if [ "$1" = "/*" ]; then  # check if the root directory is passed to the function
        echo "$1 is root folder, cannot delete root folder."
        exit
    fi
    if [ -e "$1" ]; then  # check if a Folder exists
        rm -rvf "$1"/*  # Delete the directory if it exists
    else
        echo "$1 path is invalid or does not exist."
        exit
    fi
}

function VALIDATE_FOLDERS {
    if [ "$1" = "/*" ]; then    # check if the root directory is passed to the function
        echo "$1 is root folder, cannot delete root folder."
        exit
    fi
    if [ -n "$1" ] && [ -d "$1/.." ]; then  #checks if directory string is not empty and it exists
        output_folders=("$1/../$2"*)  # Get a list of all directories starting with given input string in the parent directory

        # Loop through each directory and delete it only if it exists
        for folder in "${output_folders[@]}"; do
            if [ -d "$folder" ]; then
                rm -rf "$folder"  # Delete the directory if it exists
                echo "Deleted directory: $folder"
            else
                echo "Directory not found: $folder"
            fi
        done
    fi
}

#Input Images - Three images (mixed size)
DEFAULT_SRC_FOLDER_1="$cwd/../TEST_IMAGES/three_images_mixed_src1"
DEFAULT_SRC_FOLDER_2="$cwd/../TEST_IMAGES/three_images_mixed_src2"

# <<<<<<<<<<<<<< PROCESSING OF INPUT ARGUMENTS (NEED NOT CHANGE) >>>>>>>>>>>>>>

CASE_MIN=0
CASE_MAX=87
if (( "$#" < 3 )); then
    SRC_FOLDER_1="$DEFAULT_SRC_FOLDER_1"
    SRC_FOLDER_2="$DEFAULT_SRC_FOLDER_2"
    TEST_TYPE="0"
    QA_MODE="0"
    DECODER_TYPE="0"
    NUM_RUNS="1"
    PRESERVE_OUTPUT="1"
    BATCH_SIZE="1"
    CASE_LIST=()
    for ((case="$CASE_MIN";case<="$CASE_MAX";case++))
    do
        CASE_LIST+=("$case")
    done
else
    SRC_FOLDER_1="$1"
    SRC_FOLDER_2="$2"
    TEST_TYPE="$3"
    NUM_RUNS="$4"
    QA_MODE="$5"
    DECODER_TYPE="$6"
    PRESERVE_OUTPUT="$7"
    BATCH_SIZE="$8"
    CASE_LIST="${@:9}"
fi

# <<<<<<<<<<<<<< VALIDATION CHECKS FOR ALL INPUT PARAMETERS >>>>>>>>>>>>>>>>>>>>>>>>>>>>

if [[ "$TEST_TYPE" -ne 0 ]] && [[ "$TEST_TYPE" -ne 1 ]]; then
    echo "Invalid TEST_TYPE specified. TEST_TYPE should be 0/1 (0 = Unittests / 1 = Performancetests)"
    exit
elif [[ "$QA_MODE" -ne 0 ]] && [[ "$QA_MODE" -ne 1 ]]; then
    echo "QA mode must be in the 0 / 1. Aborting!"
    exit 0
elif [[ "$DECODER_TYPE" -ne 0 ]] && [[ "$DECODER_TYPE" -ne 1 ]]; then
    echo "Decoder Type must be in the 0/1 (0 = OpenCV / 1 = TurboJPEG). Aborting!"
    exit 0
elif [[ "$NUM_RUNS" < 1 ]]; then
    echo "Number of Iterations must be greater than or equal to 1. Aborting!"
    exit 0
elif [[ "$BATCH_SIZE" < 1 ]]; then
    echo "Batch size must be greater than or equal to 1. Aborting!"
    exit 0
elif [[ "$PRESERVE_OUTPUT" -ne 0 ]] && [[ "$PRESERVE_OUTPUT" -ne 1 ]]; then
    echo "Preserve Output must be in the 0/1 (0 = override / 1 = preserve). Aborting"
    exit 0
fi

for case in $CASE_LIST; do
    if [[ $case -lt 0 || $case -gt 87 ]]; then
        echo "The case# must be in the 0:87 range!"
    fi
done

if [[ $test_type -eq 0 && $numIterations -gt 1 ]]; then
    echo "Number of iterations should be 1 in case of unittests"
    exit 0
fi

if [[ "$TEST_TYPE" -eq 0 ]]; then
    NUM_RUNS="1"
fi

# <<<<<<<<<<<<<< REMOVE FOLDERS FROM PREVIOUS RUN BASED ON PRESERVE_OUTPUT >>>>>>>>>>>>>>>>>>>>>>>>>>>>

if [ "$PRESERVE_OUTPUT" -eq 0 ]; then
    VALIDATE_FOLDERS "$cwd" "OUTPUT_IMAGES_HOST"
    VALIDATE_FOLDERS "$cwd" "QA_RESULTS_HOST"
    VALIDATE_FOLDERS "$cwd" "OUTPUT_PERFORMANCE_LOGS_HOST"
fi

# Checking Time stamp string null or not
if [ -z "$TIMESTAMP" ]; then
  TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
fi

# <<<<<<<<<<<<<< CREATE OUTPUT FOLDERS BASED ON TEST TYPE>>>>>>>>>>>>>>>>>>>>>>>>>>>>

if [ "$TEST_TYPE" -eq 0 ]; then
    if [ "$QA_MODE" -eq 0 ]; then
        printf "\nRunning Unittests...\n"
        mkdir "$cwd/../OUTPUT_IMAGES_HOST_$TIMESTAMP"
        DEFAULT_DST_FOLDER="$cwd/../OUTPUT_IMAGES_HOST_$TIMESTAMP"
    else
        printf "\nRunning Unittests with QA mode...\n"
        mkdir "$cwd/../QA_RESULTS_HOST_$TIMESTAMP"
        DEFAULT_DST_FOLDER="$cwd/../QA_RESULTS_HOST_$TIMESTAMP"
    fi
elif [ "$TEST_TYPE" -eq 1 ]; then
    printf "\nRunning Performance tests...\n"
    mkdir "$cwd/../OUTPUT_PERFORMANCE_LOGS_HOST_$TIMESTAMP"
    DEFAULT_DST_FOLDER="$cwd/../OUTPUT_PERFORMANCE_LOGS_HOST_$TIMESTAMP"
    LOGGING_FOLDER="$cwd/../OUTPUT_PERFORMANCE_LOGS_HOST_$TIMESTAMP"
else
    echo "Invalid TEST_TYPE specified. TEST_TYPE should be 0/1 (0 = Unittests / 1 = Performancetests)"
    exit
fi
DST_FOLDER="$DEFAULT_DST_FOLDER"

# <<<<<<<<<<<<<< EXECUTION OF ALL FUNCTIONALITIES (NEED NOT CHANGE) >>>>>>>>>>>>>>

directory_name_generator() {

    if [ "$QA_MODE" -eq 0 ]; then
        AFFINITY=$1
        TYPE=$2

        if [ "$case" -lt 5 ] || [ "$case" -eq 13 ] || [ "$case" -eq 31 ] || [ "$case" -eq 34 ] || [ "$case" -eq 36 ]
        then
            FUNCTIONALITY_GROUP="color_augmentations"
        elif [ "$case" -eq 8 ] || [ "$case" -eq 30 ] || [ "$case" -eq 83 ] || [ "$case" -eq 84 ]
        then
            FUNCTIONALITY_GROUP="effects_augmentations"
        elif [ "$case" -lt 40 ]
        then
            FUNCTIONALITY_GROUP="geometric_augmentations"
        elif [ "$case" -lt 42 ]
        then
            FUNCTIONALITY_GROUP="morphological_operations"
        elif [ "$case" -eq 49 ]
        then
            FUNCTIONALITY_GROUP="filter_augmentations"
        elif [ "$case" -lt 86 ]
        then
            FUNCTIONALITY_GROUP="data_exchange_operations"
        elif [ "$case" -lt 88 ]
        then
            FUNCTIONALITY_GROUP="statistical_operations"
        else
            FUNCTIONALITY_GROUP="miscellaneous"
        fi

        DST_FOLDER_TEMP="$DST_FOLDER""/rpp_""$AFFINITY""_""$TYPE""_""$FUNCTIONALITY_GROUP"
    else
        DST_FOLDER_TEMP="$DST_FOLDER"
    fi
}

VALIDATE_PATH "$DST_FOLDER"

shopt -s extglob
mkdir build
rm -rvf build/*
cd build
cmake ..
make -j16

printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "Running all layout Inputs..."
echo "##########################################################################################"

if [ "$TEST_TYPE" -eq 0 ]; then
    for case in ${CASE_LIST[@]};
    do
        if [ "$QA_MODE" -eq 1 ]; then
            if [ "$case" -eq "54" ] || [ "$case" -eq " 84" ]; then
                echo "QA tests are not supported for case number $case, since it generates random output"
                continue
            fi
        fi
        if [ "$case" -lt "0" ] || [ "$case" -gt " 87" ]; then
            echo "Invalid case number $case. case number must be in the 0:87 range!"
            continue
        fi
        for ((layout=0;layout<3;layout++))
        do
            if [ $layout -eq 0 ]; then
                directory_name_generator "host" "pkd3" "$case"
                log_file_layout="pkd3"
            fi
            if [ $layout -eq 1 ]; then
                directory_name_generator "host" "pln3" "$case"
                log_file_layout="pln3"
            fi
            if [ $layout -eq 2 ]; then
                directory_name_generator "host" "pln1" "$case"
                log_file_layout="pln1"
            fi
            if [ "$QA_MODE" -eq 0 ]; then
                if [ ! -d "$DST_FOLDER_TEMP" ]; then
                    mkdir "$DST_FOLDER_TEMP"
                fi
            fi

            printf "\n\n\n\n"
            echo "--------------------------------"
            printf "Running a New Functionality...\n"
            echo "--------------------------------"
            for ((bitDepth=0;bitDepth<7;bitDepth++))
            do
                printf "\n\n\nRunning New Bit Depth...\n-------------------------\n\n"
                for ((outputFormatToggle=0;outputFormatToggle<2;outputFormatToggle++))
                do
                    SRC_FOLDER_1_TEMP="$SRC_FOLDER_1"
                    SRC_FOLDER_2_TEMP="$SRC_FOLDER_2"

                    # There is no layout toggle for PLN1 case, so skip this case
                    if [[ $layout -eq 2 ]] && [[ $outputFormatToggle -eq 1 ]]; then
                        continue
                    fi

                    if [ "$case" -eq 8 ]
                    then
                        for ((noiseType=0;noiseType<3;noiseType++))
                        do
                            printf "\n./Tensor_host $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case $noiseType 0"
                            ./Tensor_host "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$noiseType" "$NUM_RUNS" "$TEST_TYPE" "$layout" "0" "$QA_MODE" "$DECODER_TYPE" "$BATCH_SIZE"
                        done
                    elif [ "$case" -eq 21 ] || [ "$case" -eq 23 ] || [ "$case" -eq 24 ]
                    then
                        for ((interpolationType=0;interpolationType<6;interpolationType++))
                        do
                            printf "\n./Tensor_host $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case $interpolationType 0"
                            ./Tensor_host "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$interpolationType" "$NUM_RUNS" "$TEST_TYPE" "$layout" "0" "$QA_MODE" "$DECODER_TYPE" "$BATCH_SIZE"
                        done
                    else
                        printf "\n./Tensor_host $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case ${NUM_RUNS} ${TEST_TYPE} ${layout} 0 ${QA_MODE}" "$DECODER_TYPE" "$BATCH_SIZE"
                        ./Tensor_host "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0" "$NUM_RUNS" "$TEST_TYPE" "$layout" "0" "$QA_MODE" "$DECODER_TYPE" "$BATCH_SIZE"
                    fi

                    echo "------------------------------------------------------------------------------------------"
                done
            done
        done
    done
else
    for case in ${CASE_LIST[@]};
    do
        if [ "$case" -lt "0" ] || [ "$case" -gt " 87" ]; then
            echo "Invalid case number $case. case number must be in the 0:87 range!"
            continue
        fi
        for ((layout=0;layout<3;layout++))
        do
            if [ $layout -eq 0 ]; then
                directory_name_generator "host" "pkd3" "$case"
                log_file_layout="pkd3"
            fi
            if [ $layout -eq 1 ]; then
                directory_name_generator "host" "pln3" "$case"
                log_file_layout="pln3"
            fi
            if [ $layout -eq 2 ]; then
                directory_name_generator "host" "pln1" "$case"
                log_file_layout="pln1"
            fi

            printf "\n\n\n\n"
            echo "--------------------------------"
            printf "Running a New Functionality...\n"
            echo "--------------------------------"
            for ((bitDepth=0;bitDepth<7;bitDepth++))
            do
                printf "\n\n\nRunning New Bit Depth...\n-------------------------\n\n"
                for ((outputFormatToggle=0;outputFormatToggle<2;outputFormatToggle++))
                do
                    SRC_FOLDER_1_TEMP="$SRC_FOLDER_1"
                    SRC_FOLDER_2_TEMP="$SRC_FOLDER_2"

                    # There is no layout toggle for PLN1 case, so skip this case
                    if [[ $layout -eq 2 ]] && [[ $outputFormatToggle -eq 1 ]]; then
                        continue
                    fi

                    if [ "$case" -eq 8 ]
                    then
                        for ((noiseType=0;noiseType<3;noiseType++))
                        do
                            printf "\n./Tensor_host $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case $noiseType 0"
                            ./Tensor_host "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$noiseType" "$NUM_RUNS" "$TEST_TYPE" "$layout" "0" "$QA_MODE" "$DECODER_TYPE" "$BATCH_SIZE"| tee -a "$LOGGING_FOLDER/Tensor_host_${log_file_layout}_raw_performance_log.txt"
                        done
                    elif [ "$case" -eq 21 ] || [ "$case" -eq 23 ] || [ "$case" -eq 24 ]
                    then
                        for ((interpolationType=0;interpolationType<6;interpolationType++))
                        do
                            printf "\n./Tensor_host $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case $interpolationType 0"
                            ./Tensor_host "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$interpolationType" "$NUM_RUNS" "$TEST_TYPE" "$layout" "0" "$QA_MODE" "$DECODER_TYPE" "$BATCH_SIZE"| tee -a "$LOGGING_FOLDER/Tensor_host_${log_file_layout}_raw_performance_log.txt"
                        done
                    else
                        printf "\n./Tensor_host $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case ${NUM_RUNS} ${TEST_TYPE} ${layout} 0 ${QA_MODE}" "$DECODER_TYPE" "$BATCH_SIZE"
                        ./Tensor_host "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0" "$NUM_RUNS" "$TEST_TYPE" "$layout" "0" "$QA_MODE" "$DECODER_TYPE" "$BATCH_SIZE"| tee -a "$LOGGING_FOLDER/Tensor_host_${log_file_layout}_raw_performance_log.txt"
                    fi

                    echo "------------------------------------------------------------------------------------------"
                done
            done
        done
    done
fi
