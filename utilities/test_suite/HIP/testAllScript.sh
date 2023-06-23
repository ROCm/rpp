#!/bin/bash

# <<<<<<<<<<<<<< DEFAULT SOURCE AND DESTINATION FOLDERS (NEED NOT CHANGE) >>>>>>>>>>>>>>

cwd=$(pwd)

#Input Images - Three images (mixed size)
DEFAULT_SRC_FOLDER_1="$cwd/../TEST_IMAGES/three_images_mixed_src1"
DEFAULT_SRC_FOLDER_2="$cwd/../TEST_IMAGES/three_images_mixed_src2"

# <<<<<<<<<<<<<< PROCESSING OF INPUT ARGUMENTS (NEED NOT CHANGE) >>>>>>>>>>>>>>

CASE_MIN=0
CASE_MAX=88
if (( "$#" < 4 )); then
    SRC_FOLDER_1="$DEFAULT_SRC_FOLDER_1"
    SRC_FOLDER_2="$DEFAULT_SRC_FOLDER_2"
    PROFILING_OPTION="0"
    TEST_TYPE="0"
    QA_MODE="0"
    DECODER_TYPE="0"
    NUM_ITERATIONS="1"
    PRESERVE_OUTPUT="1"
    CASE_LIST=()
    for ((case="$CASE_MIN";case<="$CASE_MAX";case++))
    do
        CASE_LIST+=("$case")
    done
else
    SRC_FOLDER_1="$1"
    SRC_FOLDER_2="$2"
    TEST_TYPE="$3"
    NUM_ITERATIONS="$4"
    PROFILING_OPTION="$5"
    QA_MODE="$6"
    DECODER_TYPE="$7"
    PRESERVE_OUTPUT="$8"
    CASE_LIST="${@:9}"
fi

# <<<<<<<<<<<<<< VALIDATION CHECK FOR TEST_TYPE AND CASE NUMBERS >>>>>>>>>>>>>>>>>>>>>>>>>>>>

if [[ $TEST_TYPE -ne 0 ]] && [[ $TEST_TYPE -ne 1 ]]; then
    echo "Invalid TEST_TYPE specified. TEST_TYPE should be 0/1 (0 = Unittests / 1 = Performancetests)"
    exit
fi

for case in $CASE_LIST; do
    if [[ $case -lt 0 || $case -gt 88 ]]; then
        echo "The case# must be in the 0:88 range!"
    fi
done

# <<<<<<<<<<<<<< REMOVE FOLDERS FROM PREVIOUS RUN BASED ON PRESERVE_OUTPUT >>>>>>>>>>>>>>>>>>>>>>>>>>>>

if [ "$PRESERVE_OUTPUT" -ne 1 ]; then
    rm -rvf "$cwd/.."/OUTPUT_IMAGES_HIP*
    rm -rvf "$cwd/.."/QA_RESULTS_HIP*
    rm -rvf "$cwd/.."/OUTPUT_PERFORMANCE_LOGS_HIP*
fi

# <<<<<<<<<<<<<< CREATE OUTPUT FOLDERS BASED ON TEST TYPE>>>>>>>>>>>>>>>>>>>>>>>>>>>>

if [ "$TEST_TYPE" -eq 0 ]; then
    if [ "$QA_MODE" -eq 0 ]; then
        printf "\nRunning Unittests...\n"
        mkdir "$cwd/../OUTPUT_IMAGES_HIP_$TIMESTAMP"
        DEFAULT_DST_FOLDER="$cwd/../OUTPUT_IMAGES_HIP_$TIMESTAMP"
    else
        printf "\nRunning Unittests with QA mode...\n"
        mkdir "$cwd/../QA_RESULTS_HIP_$TIMESTAMP"
        DEFAULT_DST_FOLDER="$cwd/../QA_RESULTS_HIP_$TIMESTAMP"
    fi
elif [ "$TEST_TYPE" -eq 1 ]; then
    printf "\nRunning Performance tests...\n"
    mkdir "$cwd/../OUTPUT_PERFORMANCE_LOGS_HIP_$TIMESTAMP"
    DEFAULT_DST_FOLDER="$cwd/../OUTPUT_PERFORMANCE_LOGS_HIP_$TIMESTAMP"
    LOGGING_FOLDER="$cwd/../OUTPUT_PERFORMANCE_LOGS_HIP_$TIMESTAMP"
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
        else
            FUNCTIONALITY_GROUP="miscellaneous"
        fi

        DST_FOLDER_TEMP="$DST_FOLDER""/rpp_""$AFFINITY""_""$TYPE""_""$FUNCTIONALITY_GROUP"
    else
        DST_FOLDER_TEMP="$DST_FOLDER"
    fi
}

rm -rvf "$DST_FOLDER"/*
shopt -s extglob
mkdir build
cd build
rm -rvf ./*
cmake ..
make -j16

if [[ $TEST_TYPE -eq 1 ]] && [[ $PROFILING_OPTION -eq 1 ]]; then
    mkdir "$DST_FOLDER/Tensor_PKD3"
    mkdir "$DST_FOLDER/Tensor_PLN1"
    mkdir "$DST_FOLDER/Tensor_PLN3"
fi

printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "Running all layout Inputs..."
echo "##########################################################################################"

if [ "$TEST_TYPE" -eq 0 ]; then
    for case in ${CASE_LIST[@]};
    do
        if [ "$case" -lt "0" ] || [ "$case" -gt " 88" ]; then
            echo "Invalid case number $case. case number must be in the 0:88 range!"
            continue
        fi
        for ((layout=2;layout<3;layout++))
        do
            if [ $layout -eq 0 ]; then
                directory_name_generator "hip" "pkd3" "$case"
                log_file_layout="pkd3"
            fi
            if [ $layout -eq 1 ]; then
                directory_name_generator "hip" "pln3" "$case"
                log_file_layout="pln3"
            fi
            if [ $layout -eq 2 ]; then
                directory_name_generator "hip" "pln1" "$case"
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
            for ((bitDepth=0;bitDepth<1;bitDepth++))
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

                    if [ "$case" -eq 40 ] || [ "$case" -eq 41 ] || [ "$case" -eq 49 ]
                    then
                        for ((kernelSize=3;kernelSize<=9;kernelSize+=2))
                        do
                            printf "\n./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case $kernelSize"
                            ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$kernelSize" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" "$QA_MODE" "$DECODER_TYPE"
                        done
                    elif [ "$case" -eq 8 ]
                    then
                        for ((noiseType=0;noiseType<3;noiseType++))
                        do
                            printf "\n./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case $kernelSize"
                            ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$kernelSize" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" "$QA_MODE" "$DECODER_TYPE"
                        done
                    elif [ "$case" -eq 21 ] || [ "$case" -eq 23 ] || [ "$case" -eq 24 ]
                    then
                        for ((interpolationType=0;interpolationType<6;interpolationType++))
                        do
                            printf "\n./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case $interpolationType"
                            ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$interpolationType" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" "$QA_MODE" "$DECODER_TYPE"
                        done
                    else
                        printf "\n./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case ${NUM_ITERATIONS} ${TEST_TYPE} ${layout}"
                        ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" "$QA_MODE" "$DECODER_TYPE"
                    fi

                    echo "------------------------------------------------------------------------------------------"
                done
            done
        done
    done
else
    for case in ${CASE_LIST[@]};
    do
        if [ "$case" -lt "0" ] || [ "$case" -gt " 88" ]; then
            echo "Invalid case number $case. case number must be in the 0:88 range!"
            continue
        fi
        for ((layout=2;layout<3;layout++))
        do
            if [ $layout -eq 0 ]; then
                directory_name_generator "hip" "pkd3" "$case"
                log_file_layout="pkd3"
            fi
            if [ $layout -eq 1 ]; then
                directory_name_generator "hip" "pln3" "$case"
                log_file_layout="pln3"
            fi
            if [ $layout -eq 2 ]; then
                directory_name_generator "hip" "pln1" "$case"
                log_file_layout="pln1"
            fi

            printf "\n\n\n\n"
            echo "--------------------------------"
            printf "Running a New Functionality...\n"
            echo "--------------------------------"
            for ((bitDepth=0;bitDepth<1;bitDepth++))
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

                    if [[ "$PROFILING_OPTION" -eq 0 ]]
                    then
                        if [ "$case" -eq 40 ] || [ "$case" -eq 41 ] || [ "$case" -eq 49 ]
                        then
                            for ((kernelSize=3;kernelSize<=9;kernelSize+=2))
                            do
                                printf "\n./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case $kernelSize"
                                ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$kernelSize" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" "$QA_MODE" "$DECODER_TYPE"| tee -a "$LOGGING_FOLDER/Tensor_hip_${log_file_layout}_raw_performance_log.txt"
                            done
                        elif [ "$case" -eq 8 ]
                        then
                            for ((noiseType=0;noiseType<3;noiseType++))
                            do
                                printf "\n./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case $kernelSize"
                                ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$kernelSize" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" "$QA_MODE" "$DECODER_TYPE"| tee -a "$LOGGING_FOLDER/Tensor_hip_${log_file_layout}_raw_performance_log.txt"
                            done
                        elif [ "$case" -eq 21 ] || [ "$case" -eq 23 ] || [ "$case" -eq 24 ]
                        then
                            for ((interpolationType=0;interpolationType<6;interpolationType++))
                            do
                                printf "\n./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case $interpolationType"
                                ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$interpolationType" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" "$QA_MODE" "$DECODER_TYPE"| tee -a "$LOGGING_FOLDER/Tensor_hip_${log_file_layout}_raw_performance_log.txt"
                            done
                        else
                            printf "\n./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case ${NUM_ITERATIONS} ${TEST_TYPE} ${layout}"
                            ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" "$QA_MODE" "$DECODER_TYPE"| tee -a "$LOGGING_FOLDER/Tensor_hip_${log_file_layout}_raw_performance_log.txt"
                        fi

                        echo "------------------------------------------------------------------------------------------"
                    elif [[ "$PROFILING_OPTION" -eq 1 ]]
                    then
                        if [ "$case" -eq 40 ] || [ "$case" -eq 41 ] || [ "$case" -eq 49 ]
                        then
                            for ((kernelSize=3;kernelSize<=9;kernelSize+=2))
                            do
                                if [ $layout -eq 0 ]
                                then
                                    if [ ! -d "$DST_FOLDER/Tensor_PKD3/case_$case" ]; then
                                        mkdir "$DST_FOLDER/Tensor_PKD3/case_$case"
                                    fi
                                    printf "\nrocprof --basenames on --timestamp on --stats -o $DST_FOLDER/Tensor_PKD3/case_$case/output_case$case" "_bitDepth$bitDepth" "_oft$outputFormatToggle" "_kernelSize$kernelSize.csv" "./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case $kernelSize 0"
                                    rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/Tensor_PKD3/case_$case""/output_case""$case""_bitDepth""$bitDepth""_oft""$outputFormatToggle""_kernelSize""$kernelSize"".csv" ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$kernelSize" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" "$QA_MODE" "$DECODER_TYPE"| tee -a "$DST_FOLDER/Tensor_hip_pkd3_hip_raw_performance_log.txt"
                                elif [ $layout -eq 1 ]
                                then
                                    if [ ! -d "$DST_FOLDER/Tensor_PLN3/case_$case" ]; then
                                        mkdir "$DST_FOLDER/Tensor_PLN3/case_$case"
                                    fi
                                    printf "\nrocprof --basenames on --timestamp on --stats -o $DST_FOLDER/Tensor_PLN3/case_$case/output_case$case" "_bitDepth$bitDepth" "_oft$outputFormatToggle" "_kernelSize$kernelSize.csv" "./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case $kernelSize 0"
                                    rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/Tensor_PLN3/case_$case""/output_case""$case""_bitDepth""$bitDepth""_oft""$outputFormatToggle""_kernelSize""$kernelSize"".csv" ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$kernelSize" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" "$QA_MODE" "$DECODER_TYPE"| tee -a "$DST_FOLDER/Tensor_hip_pln3_hip_raw_performance_log.txt"
                                elif [ $layout -eq 2 ]
                                then
                                    if [ ! -d "$DST_FOLDER/Tensor_PLN1/case_$case" ]; then
                                        mkdir "$DST_FOLDER/Tensor_PLN1/case_$case"
                                    fi
                                    printf "\nrocprof --basenames on --timestamp on --stats -o $DST_FOLDER/Tensor_PLN1/case_$case/output_case$case" "_bitDepth$bitDepth" "_oft$outputFormatToggle" "_kernelSize$kernelSize.csv" "./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case $kernelSize 0"
                                    rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/Tensor_PLN1/case_$case""/output_case""$case""_bitDepth""$bitDepth""_oft""$outputFormatToggle""_kernelSize""$kernelSize"".csv" ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$kernelSize" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" "$QA_MODE" "$DECODER_TYPE"| tee -a "$DST_FOLDER/Tensor_hip_pln1_hip_raw_performance_log.txt"
                                fi
                            done
                        elif [ "$case" -eq 8 ]
                        then
                            for ((noiseType=0;noiseType<3;noiseType++))
                            do
                                if [ $layout -eq 0 ]
                                then
                                    if [ ! -d "$DST_FOLDER/Tensor_PKD3/case_$case" ]; then
                                        mkdir "$DST_FOLDER/Tensor_PKD3/case_$case"
                                    fi
                                    printf "\nrocprof --basenames on --timestamp on --stats -o $DST_FOLDER/Tensor_PKD3/case_$case/output_case$case" "_bitDepth$bitDepth" "_oft$outputFormatToggle" "_noiseType$noiseType.csv" "./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case $noiseType 0"
                                    rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/Tensor_PKD3/case_$case""/output_case""$case""_bitDepth""$bitDepth""_oft""$outputFormatToggle""_noiseType""$noiseType"".csv" ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$noiseType" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" "$QA_MODE" "$DECODER_TYPE"| tee -a "$DST_FOLDER/Tensor_hip_pkd3_hip_raw_performance_log.txt"
                                elif [ $layout -eq 1 ]
                                then
                                    if [ ! -d "$DST_FOLDER/Tensor_PLN3/case_$case" ]; then
                                        mkdir "$DST_FOLDER/Tensor_PLN3/case_$case"
                                    fi
                                    printf "\nrocprof --basenames on --timestamp on --stats -o $DST_FOLDER/Tensor_PLN3/case_$case/output_case$case" "_bitDepth$bitDepth" "_oft$outputFormatToggle" "_noiseType$noiseType.csv" "./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case $noiseType 0"
                                    rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/Tensor_PLN3/case_$case""/output_case""$case""_bitDepth""$bitDepth""_oft""$outputFormatToggle""_noiseType""$noiseType"".csv" ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$noiseType" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" "$QA_MODE" "$DECODER_TYPE"| tee -a "$DST_FOLDER/Tensor_hip_pln3_hip_raw_performance_log.txt"
                                elif [ $layout -eq 2 ]
                                then
                                    if [ ! -d "$DST_FOLDER/Tensor_PLN1/case_$case" ]; then
                                        mkdir "$DST_FOLDER/Tensor_PLN1/case_$case"
                                    fi
                                    printf "\nrocprof --basenames on --timestamp on --stats -o $DST_FOLDER/Tensor_PLN1/case_$case/output_case$case" "_bitDepth$bitDepth" "_oft$outputFormatToggle" "_noiseType$noiseType.csv" "./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case $noiseType 0"
                                    rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/Tensor_PLN1/case_$case""/output_case""$case""_bitDepth""$bitDepth""_oft""$outputFormatToggle""_noiseType""$noiseType"".csv" ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$noiseType" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" "$QA_MODE" "$DECODER_TYPE"| tee -a "$DST_FOLDER/Tensor_hip_pln1_hip_raw_performance_log.txt"
                                fi
                            done
                        elif [ "$case" -eq 21 ] || [ "$case" -eq 23 ] || [ "$case" -eq 24 ]
                        then
                            for ((interpolationType=0;interpolationType<6;interpolationType++))
                            do
                                if [ $layout -eq 0 ]
                                then
                                    if [ ! -d "$DST_FOLDER/Tensor_PKD3/case_$case" ]; then
                                        mkdir "$DST_FOLDER/Tensor_PKD3/case_$case"
                                    fi
                                    printf "\nrocprof --basenames on --timestamp on --stats -o $DST_FOLDER/Tensor_PKD3/case_$case/output_case$case" "_bitDepth$bitDepth" "_oft$outputFormatToggle" "_interpolationType$interpolationType.csv" "./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case $interpolationType 0"
                                    rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/Tensor_PKD3/case_$case""/output_case""$case""_bitDepth""$bitDepth""_oft""$outputFormatToggle""_interpolationType""$interpolationType"".csv" ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$interpolationType" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" "$QA_MODE" "$DECODER_TYPE"| tee -a "$DST_FOLDER/Tensor_hip_pkd3_hip_raw_performance_log.txt"
                                elif [ $layout -eq 1 ]
                                then
                                    if [ ! -d "$DST_FOLDER/Tensor_PLN3/case_$case" ]; then
                                        mkdir "$DST_FOLDER/Tensor_PLN3/case_$case"
                                    fi
                                    printf "\nrocprof --basenames on --timestamp on --stats -o $DST_FOLDER/Tensor_PLN3/case_$case/output_case$case" "_bitDepth$bitDepth" "_oft$outputFormatToggle" "_interpolationType$interpolationType.csv" "./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case $interpolationType 0"
                                    rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/Tensor_PLN3/case_$case""/output_case""$case""_bitDepth""$bitDepth""_oft""$outputFormatToggle""_interpolationType""$interpolationType"".csv" ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$interpolationType" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" "$QA_MODE" "$DECODER_TYPE"| tee -a "$DST_FOLDER/Tensor_hip_pln3_hip_raw_performance_log.txt"
                                elif [ $layout -eq 2 ]
                                then
                                    if [ ! -d "$DST_FOLDER/Tensor_PLN1/case_$case" ]; then
                                        mkdir "$DST_FOLDER/Tensor_PLN1/case_$case"
                                    fi
                                    printf "\nrocprof --basenames on --timestamp on --stats -o $DST_FOLDER/Tensor_PLN1/case_$case/output_case$case" "_bitDepth$bitDepth" "_oft$outputFormatToggle" "_interpolationType$interpolationType.csv" "./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case $interpolationType 0"
                                    rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/Tensor_PLN1/case_$case""/output_case""$case""_bitDepth""$bitDepth""_oft""$outputFormatToggle""_interpolationType""$interpolationType"".csv" ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$interpolationType" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" "$QA_MODE" "$DECODER_TYPE"| tee -a "$DST_FOLDER/Tensor_hip_pln1_hip_raw_performance_log.txt"
                                fi
                            done
                        else
                            if [ $layout -eq 0 ]
                            then
                                if [ ! -d "$DST_FOLDER/Tensor_PKD3/case_$case" ]; then
                                    mkdir "$DST_FOLDER/Tensor_PKD3/case_$case"
                                fi
                                printf "\nrocprof --basenames on --timestamp on --stats -o $DST_FOLDER/Tensor_PKD3/case_$case/output_case$case" "_bitDepth$bitDepth" "_oft$outputFormatToggle"".csv" "./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case 0 0"
                                rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/Tensor_PKD3/case_$case""/output_case""$case""_bitDepth""$bitDepth""_oft""$outputFormatToggle"".csv" ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" "$QA_MODE" "$DECODER_TYPE"| tee -a "$DST_FOLDER/Tensor_hip_pkd3_hip_raw_performance_log.txt"
                            elif [ $layout -eq 1 ]
                            then
                                if [ ! -d "$DST_FOLDER/Tensor_PLN3/case_$case" ]; then
                                    mkdir "$DST_FOLDER/Tensor_PLN3/case_$case"
                                fi
                                printf "\nrocprof --basenames on --timestamp on --stats -o $DST_FOLDER/Tensor_PLN3/case_$case/output_case$case" "_bitDepth$bitDepth" "_oft$outputFormatToggle"".csv" "./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case 0 0"
                                rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/Tensor_PLN3/case_$case""/output_case""$case""_bitDepth""$bitDepth""_oft""$outputFormatToggle"".csv" ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" "$QA_MODE" "$DECODER_TYPE"| tee -a "$DST_FOLDER/Tensor_hip_pln3_hip_raw_performance_log.txt"
                            elif [ $layout -eq 2 ]
                            then
                                if [ ! -d "$DST_FOLDER/Tensor_PLN1/case_$case" ]; then
                                    mkdir "$DST_FOLDER/Tensor_PLN1/case_$case"
                                fi
                                printf "\nrocprof --basenames on --timestamp on --stats -o $DST_FOLDER/Tensor_PLN1/case_$case/output_case$case" "_bitDepth$bitDepth" "_oft$outputFormatToggle"".csv" "./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case 0 0"
                                rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/Tensor_PLN1/case_$case""/output_case""$case""_bitDepth""$bitDepth""_oft""$outputFormatToggle"".csv" ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" "$QA_MODE" "$DECODER_TYPE"| tee -a "$DST_FOLDER/Tensor_hip_pln1_hip_raw_performance_log.txt"
                            fi
                        fi

                        echo "------------------------------------------------------------------------------------------"
                    fi
                done
            done
        done
    done
fi