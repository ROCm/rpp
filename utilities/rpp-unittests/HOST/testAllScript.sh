#!/bin/bash





# <<<<<<<<<<<<<< DEFAULT SOURCE AND DESTINATION FOLDERS (NEED NOT CHANGE) >>>>>>>>>>>>>>

# Input Images
cwd=$(pwd)
DEFAULT_SRC_FOLDER_1="$cwd/../TEST_IMAGES/single_image_224x224_src1"
DEFAULT_SRC_FOLDER_2="$cwd/../TEST_IMAGES/single_image_224x224_src2"

# Output Images
mkdir $cwd/../OUTPUT_IMAGES
DEFAULT_DST_FOLDER="$cwd/../OUTPUT_IMAGES"

# Images for unique functionalities
DEFAULT_FAST_CORNER_DETECTOR_IMAGES="$cwd/../TEST_IMAGES/fast_corner_detector"
DEFAULT_HARRIS_CORNER_DETECTOR_IMAGES="$cwd/../TEST_IMAGES/harris_corner_detector"
DEFAULT_HOUGH_LINES_IMAGES="$cwd/../TEST_IMAGES/hough_lines"
DEFAULT_HOG_IMAGES="$cwd/../TEST_IMAGES/hog"

# <<<<<<<<<<<<<< DEFAULT SOURCE AND DESTINATION FOLDERS (NEED NOT CHANGE) >>>>>>>>>>>>>>





# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
SRC_FOLDER_1="$DEFAULT_SRC_FOLDER_1"
SRC_FOLDER_2="$DEFAULT_SRC_FOLDER_2"
DST_FOLDER="$DEFAULT_DST_FOLDER"
# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>





# <<<<<<<<<<<<<< EXECUTION OF ALL FUNCTIONALITIES (NEED NOT CHANGE) >>>>>>>>>>>>>>

rm -rvf $DST_FOLDER/*
shopt -s extglob
cd build
rm -rvf *
cmake ..
make

printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "Running all PKD3 Inputs..."
echo "##########################################################################################"

printf "\n\nUsage: ./BatchPD_host_pkd3 <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:81> <verbosity = 0/1>"

for ((case=74;case<75;case++))
do
printf "\n\n\n\n"
echo "--------------------------------"
printf "Running a New Functionality...\n"
echo "--------------------------------"
for ((bitDepth=0;bitDepth<7;bitDepth++))
do
printf "\n\n\nRunning New Bit Depth...\n-------------------------\n\n"
for ((outputFormatToggle=0;outputFormatToggle<2;outputFormatToggle++))
do

if [[ "$case" -eq 66 ]]
then
    SRC_FOLDER_1_TEMP="$DEFAULT_FAST_CORNER_DETECTOR_IMAGES"
    SRC_FOLDER_2_TEMP="$DEFAULT_FAST_CORNER_DETECTOR_IMAGES"
elif [[ "$case" -eq 67 ]]
then
    SRC_FOLDER_1_TEMP="$DEFAULT_HARRIS_CORNER_DETECTOR_IMAGES"
    SRC_FOLDER_2_TEMP="$DEFAULT_HARRIS_CORNER_DETECTOR_IMAGES"
elif [[ "$case" -eq 68 ]]
then
    SRC_FOLDER_1_TEMP="$DEFAULT_HOUGH_LINES_IMAGES"
    SRC_FOLDER_2_TEMP="$DEFAULT_HOUGH_LINES_IMAGES"
elif [[ "$case" -eq 73 ]]
then
    SRC_FOLDER_1_TEMP="$DEFAULT_HOG_IMAGES"
    SRC_FOLDER_2_TEMP="$DEFAULT_HOG_IMAGES"
else
    SRC_FOLDER_1_TEMP="$SRC_FOLDER_1"
    SRC_FOLDER_2_TEMP="$SRC_FOLDER_2"
fi

printf "\n./BatchPD_host_pkd3 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER $bitDepth $outputFormatToggle $case 0"
./BatchPD_host_pkd3 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER $bitDepth $outputFormatToggle $case 0
echo "------------------------------------------------------------------------------------------"
done
done
done

mkdir $DST_FOLDER/PKD3
mv $DST_FOLDER/!(PKD3) $DST_FOLDER/PKD3




printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "Running all PLN1 Inputs..."
echo "##########################################################################################"

printf "\n\nUsage: ./BatchPD_host_pln1 <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:81> <verbosity = 0/1>"

for ((case=74;case<75;case++))
do
printf "\n\n\n\n"
echo "--------------------------------"
printf "Running a New Functionality...\n"
echo "--------------------------------"
for ((bitDepth=0;bitDepth<7;bitDepth++))
do
printf "\n\n\nRunning New Bit Depth...\n-------------------------\n\n"
for ((outputFormatToggle=0;outputFormatToggle<1;outputFormatToggle++))
do

if [[ "$case" -eq 66 ]]
then
    SRC_FOLDER_1_TEMP="$DEFAULT_FAST_CORNER_DETECTOR_IMAGES"
    SRC_FOLDER_2_TEMP="$DEFAULT_FAST_CORNER_DETECTOR_IMAGES"
elif [[ "$case" -eq 67 ]]
then
    SRC_FOLDER_1_TEMP="$DEFAULT_HARRIS_CORNER_DETECTOR_IMAGES"
    SRC_FOLDER_2_TEMP="$DEFAULT_HARRIS_CORNER_DETECTOR_IMAGES"
elif [[ "$case" -eq 68 ]]
then
    SRC_FOLDER_1_TEMP="$DEFAULT_HOUGH_LINES_IMAGES"
    SRC_FOLDER_2_TEMP="$DEFAULT_HOUGH_LINES_IMAGES"
elif [[ "$case" -eq 73 ]]
then
    SRC_FOLDER_1_TEMP="$DEFAULT_HOG_IMAGES"
    SRC_FOLDER_2_TEMP="$DEFAULT_HOG_IMAGES"
else
    SRC_FOLDER_1_TEMP="$SRC_FOLDER_1"
    SRC_FOLDER_2_TEMP="$SRC_FOLDER_2"
fi

printf "\n./BatchPD_host_pln1 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER $bitDepth $outputFormatToggle $case 0"
./BatchPD_host_pln1 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER $bitDepth $outputFormatToggle $case 0
echo "------------------------------------------------------------------------------------------"
done
done
done

mkdir $DST_FOLDER/PLN1
mv $DST_FOLDER/!(PKD3|PLN1) $DST_FOLDER/PLN1




printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "Running all PLN3 Inputs..."
echo "##########################################################################################"

printf "\n\nUsage: ./BatchPD_host_pln3 <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:81> <verbosity = 0/1>"

for ((case=74;case<75;case++))
do
printf "\n\n\n\n"
echo "--------------------------------"
printf "Running a New Functionality...\n"
echo "--------------------------------"
for ((bitDepth=0;bitDepth<7;bitDepth++))
do
printf "\n\n\nRunning New Bit Depth...\n-------------------------\n\n"
for ((outputFormatToggle=0;outputFormatToggle<2;outputFormatToggle++))
do

if [[ "$case" -eq 66 ]]
then
    SRC_FOLDER_1_TEMP="$DEFAULT_FAST_CORNER_DETECTOR_IMAGES"
    SRC_FOLDER_2_TEMP="$DEFAULT_FAST_CORNER_DETECTOR_IMAGES"
elif [[ "$case" -eq 67 ]]
then
    SRC_FOLDER_1_TEMP="$DEFAULT_HARRIS_CORNER_DETECTOR_IMAGES"
    SRC_FOLDER_2_TEMP="$DEFAULT_HARRIS_CORNER_DETECTOR_IMAGES"
elif [[ "$case" -eq 68 ]]
then
    SRC_FOLDER_1_TEMP="$DEFAULT_HOUGH_LINES_IMAGES"
    SRC_FOLDER_2_TEMP="$DEFAULT_HOUGH_LINES_IMAGES"
elif [[ "$case" -eq 73 ]]
then
    SRC_FOLDER_1_TEMP="$DEFAULT_HOG_IMAGES"
    SRC_FOLDER_2_TEMP="$DEFAULT_HOG_IMAGES"
else
    SRC_FOLDER_1_TEMP="$SRC_FOLDER_1"
    SRC_FOLDER_2_TEMP="$SRC_FOLDER_2"
fi

printf "\n./BatchPD_host_pln3 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER $bitDepth $outputFormatToggle $case 0"
./BatchPD_host_pln3 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $DST_FOLDER $bitDepth $outputFormatToggle $case 0
echo "------------------------------------------------------------------------------------------"
done
done
done

mkdir $DST_FOLDER/PLN3
mv $DST_FOLDER/!(PKD3|PLN1|PLN3) $DST_FOLDER/PLN3




# printf "\n\n\n\n\n"
# echo "##########################################################################################"
# echo "Running all Unique functionalities..."
# echo "##########################################################################################"

# printf "\n\nUsage: ./uniqueFunctionalities_host <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <case number = 0:9>"

# for ((case=0;case<10;case++))
# do
# printf "\n\n\n\n" | tee -a $DST_FOLDER/uniqueFunctionalities_host_log.txt
# echo "--------------------------------" | tee -a $DST_FOLDER/uniqueFunctionalities_host_log.txt
# printf "Running a New Functionality...\n" | tee -a $DST_FOLDER/uniqueFunctionalities_host_log.txt
# echo "--------------------------------" | tee -a $DST_FOLDER/uniqueFunctionalities_host_log.txt
# for ((bitDepth=0;bitDepth<7;bitDepth++))
# do
# printf "\n\n\nRunning New Bit Depth...\n-------------------------\n\n" | tee -a $DST_FOLDER/uniqueFunctionalities_host_log.txt
# echo "./uniqueFunctionalities_host 0 $case" | tee -a $DST_FOLDER/uniqueFunctionalities_host_log.txt
# ./uniqueFunctionalities_host 0 $case | tee -a $DST_FOLDER/uniqueFunctionalities_host_log.txt
# echo "------------------------------------------------------------------------------------------"
# done
# done

# <<<<<<<<<<<<<< EXECUTION OF ALL FUNCTIONALITIES (NEED NOT CHANGE) >>>>>>>>>>>>>>