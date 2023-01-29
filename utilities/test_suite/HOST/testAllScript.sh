#!/bin/bash

#Input Images - Three images (mixed size)
DEFAULT_SRC_FOLDER_1="$cwd/../TEST_IMAGES/three_images_mixed_src1"
DEFAULT_SRC_FOLDER_2="$cwd/../TEST_IMAGES/three_images_mixed_src2"

# Fill with default values if all arguments are not given by user
CASE_MIN=0
CASE_MAX=86
if (( "$#" < 3 )); then
    SRC_FOLDER_1="$DEFAULT_SRC_FOLDER_1"
    SRC_FOLDER_2="$DEFAULT_SRC_FOLDER_2"
    TEST_TYPE="0"
    NUM_ITERATIONS="1"
    CASE_LIST=()
    for ((case=$CASE_MIN;case<=$CASE_MAX;case++))
    do
        CASE_LIST+=("$case")
    done
else
    SRC_FOLDER_1="$1"
    SRC_FOLDER_2="$2"
    TEST_TYPE="$3"
    NUM_ITERATIONS="$4"
    CASE_LIST="${@:5}"
fi

if [[ "$TEST_TYPE" -ne 0 ]] && [[ "$TEST_TYPE" -ne 1 ]]; then
    echo "Inavlid TEST_TYPE specified. TEST_TYPE should be 0/1 (0 = Unittests / 1 = Performancetests)"
    exit
fi

for case in $CASE_LIST; do
    if [[ $case -lt 0 || $case -gt 86 ]]; then
        echo "The case# must be in the 0:86 range!"
    fi
done

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

# <<<<<<<<<<<<<< CREATE OUTPUT FOLDERS BASED ON TEST TYPE>>>>>>>>>>>>>>>>>>>>>>>>>>>>
if [ $TEST_TYPE -eq 0 ]; then
    printf "\nRunning Unittests...\n"
    mkdir "$cwd/../OUTPUT_IMAGES_HOST_NEW"
    DEFAULT_DST_FOLDER="$cwd/../OUTPUT_IMAGES_HOST_NEW"
elif [ $TEST_TYPE -eq 1 ]; then
    printf "\nRunning Performance tests...\n"
    mkdir "$cwd/../OUTPUT_PERFORMANCE_LOGS_HOST_NEW"
    DEFAULT_DST_FOLDER="$cwd/../OUTPUT_PERFORMANCE_LOGS_HOST_NEW"
    LOGGING_FOLDER="$cwd/../OUTPUT_PERFORMANCE_LOGS_HOST_NEW"
fi

# <<<<<<<<<<<<<< DEFAULT SOURCE AND DESTINATION FOLDERS (NEED NOT CHANGE) >>>>>>>>>>>>>>

# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
DST_FOLDER="$DEFAULT_DST_FOLDER"
# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>

# <<<<<<<<<<<<<< EXECUTION OF ALL FUNCTIONALITIES (NEED NOT CHANGE) >>>>>>>>>>>>>>

directory_name_generator() {

    AFFINITY=$1
    TYPE=$2
    CASE=$3

    if [[ "$case" -lt 5 ]] || [ "$case" -eq 13 ] || [ "$case" -eq 36 ]
    then
        FUNCTIONALITY_GROUP="color_augmentations"
    elif [[ "$case" -eq 8 ]] || [ "$case" -eq 30 ] || [ "$case" -eq 83 ] || [ "$case" -eq 84 ]
    then
        FUNCTIONALITY_GROUP="effects_augmentations"
    elif [[ "$case" -lt 40 ]]
    then
        FUNCTIONALITY_GROUP="geometric_augmentations"
    elif [[ "$case" -lt 42 ]]
    then
        FUNCTIONALITY_GROUP="morphological_operations"
    elif [[ "$case" -eq 49 ]]
    then
        FUNCTIONALITY_GROUP="filter_augmentations"
    elif [[ "$case" -lt 86 ]]
    then
        FUNCTIONALITY_GROUP="data_exchange_operations"
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

        if [ $TEST_TYPE -eq 0 ]; then
            mkdir $DST_FOLDER_TEMP
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
done
# <<<<<<<<<<<<<< EXECUTION OF ALL FUNCTIONALITIES (NEED NOT CHANGE) >>>>>>>>>>>>>>