#!/bin/bash

# Fill with default values if all arguments are not given by user
CASE_MIN=0
CASE_MAX=86
if (( "$#" < 4 )); then
    PROFILING_OPTION="0"
    TEST_TYPE="0"
    NUM_ITERATIONS="1"
    CASE_LIST=()
    for ((case=$CASE_MIN;case<=$CASE_MAX;case++))
    do
        CASE_LIST+=("$case")
    done
else
    PROFILING_OPTION="$1"
    TEST_TYPE="$2"
    NUM_ITERATIONS="$3"
    CASE_LIST="${@:4}"
fi

# <<<<<<<<<<<<<< DEFAULT SOURCE AND DESTINATION FOLDERS (NEED NOT CHANGE) >>>>>>>>>>>>>>

cwd=$(pwd)

# Input Images - Single image (224 x 224)
# DEFAULT_SRC_FOLDER_1="$cwd/../../rpp-unittests/TEST_IMAGES/single_image_224x224_src1"
# DEFAULT_SRC_FOLDER_2="$cwd/../../rpp-unittests/TEST_IMAGES/single_image_224x224_src2"

# Input Images - Two images (224 x 224)
# DEFAULT_SRC_FOLDER_1="$cwd/../../rpp-unittests/TEST_IMAGES/two_images_224x224_src1"
# DEFAULT_SRC_FOLDER_2="$cwd/../../rpp-unittests/TEST_IMAGES/two_images_224x224_src2"

# Input Images - Three images (224 x 224)
# DEFAULT_SRC_FOLDER_1="$cwd/../../rpp-unittests/TEST_IMAGES/three_images_224x224_src1"
# DEFAULT_SRC_FOLDER_2="$cwd/../../rpp-unittests/TEST_IMAGES/three_images_224x224_src2"

#Input Images - Three images (mixed size)
DEFAULT_SRC_FOLDER_1="$cwd/../TEST_IMAGES/three_images_mixed_src1"
DEFAULT_SRC_FOLDER_2="$cwd/../TEST_IMAGES/three_images_mixed_src2"

# Input Images - Two images (mixed size)
# DEFAULT_SRC_FOLDER_1="$cwd/../../rpp-unittests/TEST_IMAGES/two_images_mixed_src1"
# DEFAULT_SRC_FOLDER_2="$cwd/../../rpp-unittests/TEST_IMAGES/two_images_mixed_src2"

# <<<<<<<<<<<<<< PRINTING THE TEST TYPE THAT USER SPECIFIED >>>>>>>>>>>>>>>>>>>>>>>>>>>>
if [ $TEST_TYPE -eq 0 ]; then
    printf "\nRunning Unittests...\n"
    mkdir "$cwd/../OUTPUT_IMAGES_HIP_NEW"
    DEFAULT_DST_FOLDER="$cwd/../OUTPUT_IMAGES_HIP_NEW"
elif [ $TEST_TYPE -eq 1 ]; then
    printf "\nRunning Performance tests...\n"
    rm -rvf "$cwd/../OUTPUT_PERFORMANCE_LOGS_HIP_NEW"
    mkdir "$cwd/../OUTPUT_PERFORMANCE_LOGS_HIP_NEW"
    DEFAULT_DST_FOLDER="$cwd/../OUTPUT_PERFORMANCE_LOGS_HIP_NEW"
    LOGGING_FOLDER="$cwd/../OUTPUT_PERFORMANCE_LOGS_HIP_NEW"
fi

# <<<<<<<<<<<<<< DEFAULT SOURCE AND DESTINATION FOLDERS (NEED NOT CHANGE) >>>>>>>>>>>>>>

# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
SRC_FOLDER_1="$DEFAULT_SRC_FOLDER_1"
SRC_FOLDER_2="$DEFAULT_SRC_FOLDER_2"
DST_FOLDER="$DEFAULT_DST_FOLDER"
# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>

# <<<<<<<<<<<<<< EXECUTION OF ALL FUNCTIONALITIES (NEED NOT CHANGE) >>>>>>>>>>>>>>

directory_name_generator() {

    AFFINITY=$1
    TYPE=$2
    CASE=$3

    if [[ "$case" -lt 15 ]]
    then
        FUNCTIONALITY_GROUP="image_augmentations"
    elif [[ "$case" -lt 20 ]]
    then
        FUNCTIONALITY_GROUP="statistical_functions"
    elif [[ "$case" -lt 29 ]]
    then
        FUNCTIONALITY_GROUP="geometry_transforms"
    elif [[ "$case" -lt 36 ]]
    then
        FUNCTIONALITY_GROUP="advanced_augmentations"
    elif [[ "$case" -lt 40 ]]
    then
        FUNCTIONALITY_GROUP="fused_functions"
    elif [[ "$case" -lt 42 ]]
    then
        FUNCTIONALITY_GROUP="morphological_transforms"
    elif [[ "$case" -lt 49 ]]
    then
        FUNCTIONALITY_GROUP="color_model_conversions"
    elif [[ "$case" -lt 56 ]]
    then
        FUNCTIONALITY_GROUP="filter_operations"
    elif [[ "$case" -lt 65 ]]
    then
        FUNCTIONALITY_GROUP="arithmetic_operations"
    elif [[ "$case" -lt 69 ]]
    then
        FUNCTIONALITY_GROUP="logical_operations"
    elif [[ "$case" -lt 79 ]]
    then
        FUNCTIONALITY_GROUP="computer_vision"
    else
        FUNCTIONALITY_GROUP="miscellaneous"
    fi

    DST_FOLDER_TEMP="$DST_FOLDER""/rpp_""$AFFINITY""_""$TYPE""_""$FUNCTIONALITY_GROUP"
}

rm -rvf "$DST_FOLDER"/*
shopt -s extglob
mkdir build
cd build
rm -rvf ./*
cmake ..
make -j16

if [ $TEST_TYPE -eq 1 ] && [ "$PROFILING_OPTION" -eq 1 ]
then
    mkdir "$DST_FOLDER/Tensor_PKD3"
    mkdir "$DST_FOLDER/Tensor_PLN1"
    mkdir "$DST_FOLDER/Tensor_PLN3"
fi

# <<<<<<<<<<<<<< EXECUTION OF ALL FUNCTIONALITIES (NEED NOT CHANGE) >>>>>>>>>>>>>>
printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "Running all layout Inputs..."
echo "##########################################################################################"

for case in ${CASE_LIST[@]};
do
    if [ "$case" -lt "0" ] || [ "$case" -gt " 86" ]; then
        echo "Invalid case number $case. casenumber must be in the 0:86 range!"
        continue
    fi
    for ((layout=0;layout<3;layout++))
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

        if [ $TEST_TYPE -eq 0 ]; then
            mkdir $DST_FOLDER_TEMP
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

                if [[ "$case" -eq 74 ]]
                then
                    SRC_FOLDER_1_TEMP="$DEFAULT_HARRIS_CORNER_DETECTOR_IMAGES"
                    SRC_FOLDER_2_TEMP="$DEFAULT_HARRIS_CORNER_DETECTOR_IMAGES"
                elif [[ "$case" -eq 75 ]]
                then
                    SRC_FOLDER_1_TEMP="$DEFAULT_FAST_CORNER_DETECTOR_IMAGES"
                    SRC_FOLDER_2_TEMP="$DEFAULT_FAST_CORNER_DETECTOR_IMAGES"
                elif [[ "$case" -eq 77 ]]
                then
                    SRC_FOLDER_1_TEMP="$DEFAULT_HOUGH_LINES_IMAGES"
                    SRC_FOLDER_2_TEMP="$DEFAULT_HOUGH_LINES_IMAGES"
                elif [[ "$case" -eq 78 ]]
                then
                    SRC_FOLDER_1_TEMP="$DEFAULT_HOG_IMAGES"
                    SRC_FOLDER_2_TEMP="$DEFAULT_HOG_IMAGES"
                else
                    SRC_FOLDER_1_TEMP="$SRC_FOLDER_1"
                    SRC_FOLDER_2_TEMP="$SRC_FOLDER_2"
                fi

                if [[ "$PROFILING_OPTION" -eq 0 ]]
                then
                    if [ "$case" -eq 40 ] || [ "$case" -eq 41 ] || [ "$case" -eq 49 ]
                    then
                        for ((kernelSize=3;kernelSize<=9;kernelSize+=2))
                        do
                            printf "\n./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case $kernelSize"
                            ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$kernelSize" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0"| tee -a "$LOGGING_FOLDER/Tensor_hip_${log_file_layout}_raw_performance_log.txt"
                        done
                    elif [ "$case" -eq 8 ]
                    then
                        for ((noiseType=0;noiseType<3;noiseType++))
                        do
                            printf "\n./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case $kernelSize"
                            ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$kernelSize" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0"| tee -a "$LOGGING_FOLDER/Tensor_hip_${log_file_layout}_raw_performance_log.txt"
                        done
                    elif [ "$case" -eq 21 ] || [ "$case" -eq 23 ] || [ "$case" -eq 24 ]
                    then
                        for ((interpolationType=0;interpolationType<6;interpolationType++))
                        do
                            printf "\n./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case $interpolationType"
                            ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$interpolationType" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0"| tee -a "$LOGGING_FOLDER/Tensor_hip_${log_file_layout}_raw_performance_log.txt"
                        done
                    else
                        printf "\n./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case ${NUM_ITERATIONS} ${TEST_TYPE} ${layout}"
                        ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0"| tee -a "$LOGGING_FOLDER/Tensor_hip_${log_file_layout}_raw_performance_log.txt"
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
                                mkdir "$DST_FOLDER/Tensor_PKD3/case_$case"
                                printf "\nrocprof --basenames on --timestamp on --stats -o $DST_FOLDER/Tensor_PKD3/case_$case/output_case$case" "_bitDepth$bitDepth" "_oft$outputFormatToggle" "_kernelSize$kernelSize.csv" "./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case $kernelSize 0"
                                rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/Tensor_PKD3/case_$case""/output_case""$case""_bitDepth""$bitDepth""_oft""$outputFormatToggle""_kernelSize""$kernelSize"".csv" ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$kernelSize" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" | tee -a "$DST_FOLDER/Tensor_hip_pkd3_hip_raw_performance_log.txt"
                            elif [ $layout -eq 1 ]
                            then
                                mkdir "$DST_FOLDER/Tensor_PLN3/case_$case"
                                printf "\nrocprof --basenames on --timestamp on --stats -o $DST_FOLDER/Tensor_PLN3/case_$case/output_case$case" "_bitDepth$bitDepth" "_oft$outputFormatToggle" "_kernelSize$kernelSize.csv" "./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case $kernelSize 0"
                                rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/Tensor_PLN3/case_$case""/output_case""$case""_bitDepth""$bitDepth""_oft""$outputFormatToggle""_kernelSize""$kernelSize"".csv" ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$kernelSize" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" | tee -a "$DST_FOLDER/Tensor_hip_pln3_hip_raw_performance_log.txt"
                            elif [ $layout -eq 2 ]
                            then
                                mkdir "$DST_FOLDER/Tensor_PLN1/case_$case"
                                printf "\nrocprof --basenames on --timestamp on --stats -o $DST_FOLDER/Tensor_PLN1/case_$case/output_case$case" "_bitDepth$bitDepth" "_oft$outputFormatToggle" "_kernelSize$kernelSize.csv" "./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case $kernelSize 0"
                                rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/Tensor_PLN1/case_$case""/output_case""$case""_bitDepth""$bitDepth""_oft""$outputFormatToggle""_kernelSize""$kernelSize"".csv" ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$kernelSize" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" | tee -a "$DST_FOLDER/Tensor_hip_pln1_hip_raw_performance_log.txt"
                            fi
                        done
                    elif [ "$case" -eq 8 ]
                    then
                        for ((noiseType=0;noiseType<3;noiseType++))
                        do
                            if [ $layout -eq 0 ]
                            then
                                mkdir "$DST_FOLDER/Tensor_PKD3/case_$case"
                                printf "\nrocprof --basenames on --timestamp on --stats -o $DST_FOLDER/Tensor_PKD3/case_$case/output_case$case" "_bitDepth$bitDepth" "_oft$outputFormatToggle" "_noiseType$noiseType.csv" "./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case $noiseType 0"
                                rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/Tensor_PKD3/case_$case""/output_case""$case""_bitDepth""$bitDepth""_oft""$outputFormatToggle""_noiseType""$noiseType"".csv" ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$noiseType" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" | tee -a "$DST_FOLDER/Tensor_hip_pkd3_hip_raw_performance_log.txt"
                            elif [ $layout -eq 1 ]
                            then
                                mkdir "$DST_FOLDER/Tensor_PLN3/case_$case"
                                printf "\nrocprof --basenames on --timestamp on --stats -o $DST_FOLDER/Tensor_PLN3/case_$case/output_case$case" "_bitDepth$bitDepth" "_oft$outputFormatToggle" "_noiseType$noiseType.csv" "./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case $noiseType 0"
                                rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/Tensor_PLN3/case_$case""/output_case""$case""_bitDepth""$bitDepth""_oft""$outputFormatToggle""_noiseType""$noiseType"".csv" ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$noiseType" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" | tee -a "$DST_FOLDER/Tensor_hip_pln3_hip_raw_performance_log.txt"
                            elif [ $layout -eq 2 ]
                            then
                                mkdir "$DST_FOLDER/Tensor_PLN1/case_$case"
                                printf "\nrocprof --basenames on --timestamp on --stats -o $DST_FOLDER/Tensor_PLN1/case_$case/output_case$case" "_bitDepth$bitDepth" "_oft$outputFormatToggle" "_noiseType$noiseType.csv" "./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case $noiseType 0"
                                rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/Tensor_PLN1/case_$case""/output_case""$case""_bitDepth""$bitDepth""_oft""$outputFormatToggle""_noiseType""$noiseType"".csv" ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$noiseType" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" | tee -a "$DST_FOLDER/Tensor_hip_pln1_hip_raw_performance_log.txt"
                            fi
                        done
                    elif [ "$case" -eq 21 ] || [ "$case" -eq 23 ] || [ "$case" -eq 24 ]
                    then
                        for ((interpolationType=0;interpolationType<6;interpolationType++))
                        do
                            if [ $layout -eq 0 ]
                            then
                                mkdir "$DST_FOLDER/Tensor_PKD3/case_$case"
                                printf "\nrocprof --basenames on --timestamp on --stats -o $DST_FOLDER/Tensor_PKD3/case_$case/output_case$case" "_bitDepth$bitDepth" "_oft$outputFormatToggle" "_interpolationType$interpolationType.csv" "./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case $interpolationType 0"
                                rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/Tensor_PKD3/case_$case""/output_case""$case""_bitDepth""$bitDepth""_oft""$outputFormatToggle""_interpolationType""$interpolationType"".csv" ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$interpolationType" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" | tee -a "$DST_FOLDER/Tensor_hip_pkd3_hip_raw_performance_log.txt"
                            elif [ $layout -eq 1 ]
                            then
                                mkdir "$DST_FOLDER/Tensor_PLN3/case_$case"
                                printf "\nrocprof --basenames on --timestamp on --stats -o $DST_FOLDER/Tensor_PLN3/case_$case/output_case$case" "_bitDepth$bitDepth" "_oft$outputFormatToggle" "_interpolationType$interpolationType.csv" "./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case $interpolationType 0"
                                rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/Tensor_PLN3/case_$case""/output_case""$case""_bitDepth""$bitDepth""_oft""$outputFormatToggle""_interpolationType""$interpolationType"".csv" ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$interpolationType" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" | tee -a "$DST_FOLDER/Tensor_hip_pln3_hip_raw_performance_log.txt"
                            elif [ $layout -eq 2 ]
                            then
                                mkdir "$DST_FOLDER/Tensor_PLN1/case_$case"
                                printf "\nrocprof --basenames on --timestamp on --stats -o $DST_FOLDER/Tensor_PLN1/case_$case/output_case$case" "_bitDepth$bitDepth" "_oft$outputFormatToggle" "_interpolationType$interpolationType.csv" "./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case $interpolationType 0"
                                rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/Tensor_PLN1/case_$case""/output_case""$case""_bitDepth""$bitDepth""_oft""$outputFormatToggle""_interpolationType""$interpolationType"".csv" ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$interpolationType" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" | tee -a "$DST_FOLDER/Tensor_hip_pln1_hip_raw_performance_log.txt"
                            fi
                        done
                    else
                        if [ $layout -eq 0 ]
                        then
                            mkdir "$DST_FOLDER/Tensor_PKD3/case_$case"
                            printf "\nrocprof --basenames on --timestamp on --stats -o $DST_FOLDER/Tensor_PKD3/case_$case/output_case$case" "_bitDepth$bitDepth" "_oft$outputFormatToggle"".csv" "./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case 0 0"
                            rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/Tensor_PKD3/case_$case""/output_case""$case""_bitDepth""$bitDepth""_oft""$outputFormatToggle"".csv" ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" | tee -a "$DST_FOLDER/Tensor_hip_pkd3_hip_raw_performance_log.txt"
                        elif [ $layout -eq 1 ]
                        then
                            mkdir "$DST_FOLDER/Tensor_PLN3/case_$case"
                            printf "\nrocprof --basenames on --timestamp on --stats -o $DST_FOLDER/Tensor_PLN3/case_$case/output_case$case" "_bitDepth$bitDepth" "_oft$outputFormatToggle"".csv" "./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case 0 0"
                            rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/Tensor_PLN3/case_$case""/output_case""$case""_bitDepth""$bitDepth""_oft""$outputFormatToggle"".csv" ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" | tee -a "$DST_FOLDER/Tensor_hip_pln3_hip_raw_performance_log.txt"
                        elif [ $layout -eq 2 ]
                        then
                            mkdir "$DST_FOLDER/Tensor_PLN1/case_$case"
                            printf "\nrocprof --basenames on --timestamp on --stats -o $DST_FOLDER/Tensor_PLN1/case_$case/output_case$case" "_bitDepth$bitDepth" "_oft$outputFormatToggle"".csv" "./Tensor_hip $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case 0 0"
                            rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/Tensor_PLN1/case_$case""/output_case""$case""_bitDepth""$bitDepth""_oft""$outputFormatToggle"".csv" ./Tensor_hip "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" | tee -a "$DST_FOLDER/Tensor_hip_pln1_hip_raw_performance_log.txt"
                        fi
                    fi

                    echo "------------------------------------------------------------------------------------------"
                fi
            done
        done
    done
done

if [ $TEST_TYPE -eq 0 ]; then
    for ((layout=0;layout<3;layout++))
    do
        if [[ "$layout" -eq 0 ]]
        then
            mkdir "$DST_FOLDER/PKD3"
            PKD3_FOLDERS=$(find $DST_FOLDER -maxdepth 1 -name "*pkd3*")
            for TEMP_FOLDER in $PKD3_FOLDERS
            do
                mv "$TEMP_FOLDER" "$DST_FOLDER/PKD3"
            done
        elif [[ "$layout" -eq 1 ]]
        then
            mkdir "$DST_FOLDER/PLN3"
            PLN3_FOLDERS=$(find $DST_FOLDER -maxdepth 1 -name "*pln3*")
            for TEMP_FOLDER in $PLN3_FOLDERS
            do
                mv "$TEMP_FOLDER" "$DST_FOLDER/PLN3"
            done
        else
            mkdir "$DST_FOLDER/PLN1"
            PLN1_FOLDERS=$(find $DST_FOLDER -maxdepth 1 -name "*pln1*")
            for TEMP_FOLDER in $PLN1_FOLDERS
            do
                mv "$TEMP_FOLDER" "$DST_FOLDER/PLN1"
            done
        fi
    done
fi
# <<<<<<<<<<<<<< EXECUTION OF ALL FUNCTIONALITIES (NEED NOT CHANGE) >>>>>>>>>>>>>>