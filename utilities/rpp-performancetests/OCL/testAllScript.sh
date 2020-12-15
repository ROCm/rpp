#!/bin/bash

# <<<<<<<<<<<<<< DEFAULT SOURCE AND DESTINATION FOLDERS (NEED NOT CHANGE) >>>>>>>>>>>>>>

cwd=$(pwd)

# Input Images - Two images (224 x 224)
DEFAULT_SRC_FOLDER_1="$cwd/../TEST_IMAGES/two_images_224x224_src1"
DEFAULT_SRC_FOLDER_2="$cwd/../TEST_IMAGES/two_images_224x224_src2"

# Output Images
# Additional Note - There are no output images for rpp-performancetests - Only max, min, avg performance times for 1000 batches - A destination folder path however needs to be passed
mkdir $cwd/../OUTPUT_IMAGES_OCL
DEFAULT_DST_FOLDER="$cwd/../OUTPUT_IMAGES_OCL"

# <<<<<<<<<<<<<< DEFAULT SOURCE AND DESTINATION FOLDERS (NEED NOT CHANGE) >>>>>>>>>>>>>>





# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
SRC_FOLDER_1="$DEFAULT_SRC_FOLDER_1"
SRC_FOLDER_2="$DEFAULT_SRC_FOLDER_2"
DST_FOLDER="$DEFAULT_DST_FOLDER"
# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>





# <<<<<<<<<<<<<< EXECUTION OF ALL FUNCTIONALITIES (NEED NOT CHANGE) >>>>>>>>>>>>>>

rm -rvf $DST_FOLDER/*
shopt -s extglob
mkdir build
cd build
rm -rvf *
cmake ..
make

for ((case=0;case<65;case++))
do
printf "\n\n\n\n"
echo "--------------------------------"
printf "Running a New Functionality...\n"
echo "--------------------------------"
printf "\n./BatchPD_ocl $SRC_FOLDER_1 $SRC_FOLDER_2 $DST_FOLDER $case 0\n"
./BatchPD_ocl $SRC_FOLDER_1 $SRC_FOLDER_2 $DST_FOLDER $case 0
echo "------------------------------------------------------------------------------------------"
done