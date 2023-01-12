#!/bin/bash

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

# Input Images - Two images (mixed size)
# DEFAULT_SRC_FOLDER_1="$cwd/../../rpp-unittests/TEST_IMAGES/two_images_mixed_src1"
# DEFAULT_SRC_FOLDER_2="$cwd/../../rpp-unittests/TEST_IMAGES/two_images_mixed_src2"

#Input Images - Two images (mixed size)
# DEFAULT_SRC_FOLDER_1="$cwd/TEST_IMAGES/"
# DEFAULT_SRC_FOLDER_2="$cwd/TEST_IMAGES/"

#Input Images - Three images (mixed size)
DEFAULT_SRC_FOLDER_1="$cwd/../TEST_IMAGES/three_images_mixed_src1"
DEFAULT_SRC_FOLDER_2="$cwd/../TEST_IMAGES/three_images_mixed_src2"

TEST_TYPE=$3

# Output Images
mkdir "$cwd/../OUTPUT_IMAGES_HOST_NEW"
DEFAULT_DST_FOLDER="$cwd/../OUTPUT_IMAGES_HOST_NEW"

# logging folders for performance tests
if [ $TEST_TYPE -eq 1 ]; then
    rm -rvf "$cwd/../OUTPUT_PERFORMANCE_LOGS_HOST_NEW"
    mkdir "$cwd/../OUTPUT_PERFORMANCE_LOGS_HOST_NEW"
    LOGGING_FOLDER="$cwd/../OUTPUT_PERFORMANCE_LOGS_HOST_NEW"
fi

# <<<<<<<<<<<<<< PRINTING THE TEST TYPE THAT USER SPECIFIED >>>>>>>>>>>>>>>>>>>>>>>>>>>>
if [ $TEST_TYPE -eq 0 ]; then
    printf "\nRunning Unittests...\n"
elif [ $TEST_TYPE -eq 1 ]; then
    printf "\nRunning Performance tests...\n"
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

if [[ "$1" -lt 0 ]] | [[ "$1" -gt 86 ]]; then
    echo "The starting case# must be in the 0:86 range!"
    echo
    echo "The testAllScript.sh bash script runs the RPP performance testsuite for AMDRPP functionalities in HOST/OCL/HIP backends."
    echo
    echo "Syntax: ./testAllScript.sh <S> <E> <T> <N>"
    echo "S     CASE_START (Starting case# (0:86))"
    echo "E     CASE_END (Ending case# (0:86))"
    echo "T     TEST_TYPE - (0 = Unittests / 1 = Performancetests)"
    echo "N     NUM_ITERATIONS - (0 = Unittests / 1 = Performancetests)"
    exit 1
fi

if [[ "$2" -lt 0 ]] | [[ "$2" -gt 86 ]]; then
    echo "The ending case# must be in the 0:86 range!"
    echo
    echo "The testAllScript.sh bash script runs the RPP performance testsuite for AMDRPP functionalities in HOST/OCL/HIP backends."
    echo
    echo "Syntax: ./testAllScript.sh <S> <E> <T> <N>"
    echo "S     CASE_START (Starting case# (0:86))"
    echo "E     CASE_END (Ending case# (0:86))"
    echo "T     TEST_TYPE - (0 = Unittests / 1 = Performancetests)"
    echo "N     NUM_ITERATIONS - (0 = Unittests / 1 = Performancetests)"
    exit 1
fi

if (( "$#" < 2 )); then
    CASE_START="0"
    CASE_END="86"
    TEST_TYPE="0"
    NUM_ITERATIONS="1"
else
    CASE_START="$1"
    CASE_END="$2"
    TEST_TYPE="$3"
    NUM_ITERATIONS="$4"
fi

rm -rvf "$DST_FOLDER"/*
shopt -s extglob
mkdir build
cd build
rm -rvf ./*
cmake ..
make -j16

printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "Running all layout Inputs..."
echo "##########################################################################################"

for ((layout=0;layout<=2;layout++))
do
    for ((case=$CASE_START;case<=$CASE_END;case++))
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

        mkdir $DST_FOLDER_TEMP
        printf "\n\n\n\n"
        echo "--------------------------------"
        printf "Running a New Functionality...\n"
        echo "--------------------------------"
        for ((bitDepth=0;bitDepth<1;bitDepth++))
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

                if [ "$case" -eq 8 ]
                then
                    for ((noiseType=0;noiseType<3;noiseType++))
                    do
                        printf "\n./Tensor_host $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case $noiseType 0"
                        ./Tensor_host "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$noiseType" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" | tee -a "$LOGGING_FOLDER/Tensor_host_${log_file_layout}_raw_performance_log.txt"
                    done
                elif [ "$case" -eq 21 ] || [ "$case" -eq 23 ] || [ "$case" -eq 24 ]
                then
                    for ((interpolationType=0;interpolationType<6;interpolationType++))
                    do
                        printf "\n./Tensor_host $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case $interpolationType 0"
                        ./Tensor_host "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "$interpolationType" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" | tee -a "$LOGGING_FOLDER/Tensor_host_${log_file_layout}_raw_performance_log.txt"
                    done
                else
                    printf "\n./Tensor_host $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER_TEMP $bitDepth $outputFormatToggle $case ${NUM_ITERATIONS} ${TEST_TYPE} ${layout} 0"
                    ./Tensor_host "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$DST_FOLDER_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0" "$NUM_ITERATIONS" "$TEST_TYPE" "$layout" "0" | tee -a "$LOGGING_FOLDER/Tensor_host_${log_file_layout}_raw_performance_log.txt"
                fi

                echo "------------------------------------------------------------------------------------------"
            done
        done
    done

    if [[ "$layout" -eq 0 ]]
    then
        mkdir "$DST_FOLDER/PKD3"
        mv "$DST_FOLDER/"!(PKD3) "$DST_FOLDER/PKD3"
    elif [[ "$layout" -eq 1 ]]
    then
        mkdir "$DST_FOLDER/PLN3"
        mv "$DST_FOLDER/"!(PKD3|PLN3) "$DST_FOLDER/PLN3"
    else
        mkdir "$DST_FOLDER/PLN1"
        mv "$DST_FOLDER/"!(PKD3|PLN1|PLN3) "$DST_FOLDER/PLN1"
    fi

done
# <<<<<<<<<<<<<< EXECUTION OF ALL FUNCTIONALITIES (NEED NOT CHANGE) >>>>>>>>>>>>>>