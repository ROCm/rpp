#!/bin/bash

# <<<<<<<<<<<<<< DEFAULT SOURCE AND DESTINATION FOLDERS (NEED NOT CHANGE) >>>>>>>>>>>>>>

cwd=$(pwd)

# Input audio files - Eight audio files
DEFAULT_SRC_FOLDER="$cwd/../../../TEST_AUDIO_FILES/eight_samples_single_channel_src1/"

# Inputs for Testing Downmixing
# Input audio file - single audio file - multi channel
# DEFAULT_SRC_FOLDER="$cwd/../../../TEST_AUDIO_FILES/single_sample_multi_channel_src1/"

# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
SRC_FOLDER="$DEFAULT_SRC_FOLDER"

# <<<<<<<<<<<<<< EXECUTION OF ALL FUNCTIONALITIES (NEED NOT CHANGE) >>>>>>>>>>>>>>
if [[ "$1" -lt 0 ]] | [[ "$1" -gt 9 ]]; then
    echo "The starting case# must be in the 0-9 range!"
    echo
    echo "The testAllScript.sh bash script runs the RPP audio unittest testsuite for AMDRPP functionalities in HOST/OCL/HIP backends."
    echo
    echo "Syntax: ./testAllScriptAudio.sh <S> <E>"
    echo "S     CASE_START (Starting case# (0-9))"
    echo "E     CASE_END (Ending case# (0-9))"
    exit 1
fi

if [[ "$2" -lt 0 ]] | [[ "$2" -gt 9 ]]; then
    echo "The ending case# must be in the 0-9 range!"
    echo
    echo "The testAllScript.sh bash script runs the RPP audio unittest testsuite for AMDRPP functionalities in HOST/OCL/HIP backends."
    echo
    echo "Syntax: ./testAllScriptAudio.sh <S> <E>"
    echo "S     CASE_START (Starting case# (0-9))"
    echo "E     CASE_END (Ending case# (0-9))"
    exit 1
fi

if (( "$#" < 2 )); then
    CASE_START="0"
    CASE_END="9"
else
    CASE_START="$1"
    CASE_END="$2"
fi

shopt -s extglob
mkdir build
cd build
rm -rvf ./*
cmake ..
make -j16

printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "Running all Audio Inputs..."
echo "##########################################################################################"

printf "\n\nUsage: ./Tensor_host_audio <src folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <case number = 0:1>"

for ((case=$CASE_START;case<=$CASE_END;case++))
do
    if [ "$case" -eq 3 ]
    then
        SRC_FOLDER="$cwd/../../TEST_AUDIO_FILES/single_sample_multi_channel_src1/"
    else
        SRC_FOLDER="$cwd/../../TEST_AUDIO_FILES/eight_samples_single_channel_src1/"
    fi

    printf "\n\n\n\n"
    echo "--------------------------------"
    printf "Running a New Functionality...\n"
    echo "--------------------------------"
    for ((bitDepth=2;bitDepth<3;bitDepth++))
    do
        printf "\n\n\nRunning New Bit Depth...\n-------------------------\n\n"
        printf "\n./Tensor_host_audio $SRC_FOLDER $bitDepth $case "
        ./Tensor_host_audio "$SRC_FOLDER" "$bitDepth" "$case"

        echo "------------------------------------------------------------------------------------------"
    done
done

# <<<<<<<<<<<<<< EXECUTION OF ALL FUNCTIONALITIES (NEED NOT CHANGE) >>>>>>>>>>>>>>
