#!/bin/bash

# <<<<<<<<<<<<<< JUST CHANGE SOURCE AND DESTINATION FOLDERS >>>>>>>>>>>>>>
SRC_FOLDER_1="/home/abishek/abishek/tests/Input_Images/single1920x1080"
SRC_FOLDER_2="/home/abishek/abishek/tests/Input_Images/single1920x1080_1"
DST_FOLDER="/home/abishek/abishek/tests/Output_Images"
# <<<<<<<<<<<<<< JUST CHANGE SOURCE AND DESTINATION FOLDERS >>>>>>>>>>>>>>








rm -rvf $DST_FOLDER/*
shopt -s extglob
cd build

printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "Running all PKD3 Inputs..."
echo "##########################################################################################"

printf "\n\nUsage: ./BatchPD_host_pkd3 <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 1:7> <verbosity = 0/1>"

for ((case=1;case<9;case++))
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
printf "\n./BatchPD_host_pkd3 $SRC_FOLDER_1 $SRC_FOLDER_2 $DST_FOLDER $bitDepth $outputFormatToggle $case 0"
./BatchPD_host_pkd3 $SRC_FOLDER_1 $SRC_FOLDER_2 $DST_FOLDER $bitDepth $outputFormatToggle $case 0
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

printf "\n\nUsage: ./BatchPD_host_pln1 <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 1:7> <verbosity = 0/1>"

for ((case=1;case<9;case++))
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
printf "\n./BatchPD_host_pln1 $SRC_FOLDER_1 $SRC_FOLDER_2 $DST_FOLDER $bitDepth $outputFormatToggle $case 0"
./BatchPD_host_pln1 $SRC_FOLDER_1 $SRC_FOLDER_2 $DST_FOLDER $bitDepth $outputFormatToggle $case 0
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

printf "\n\nUsage: ./BatchPD_host_pln3 <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 1:7> <verbosity = 0/1>"

for ((case=1;case<9;case++))
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
printf "\n./BatchPD_host_pln3 $SRC_FOLDER_1 $SRC_FOLDER_2 $DST_FOLDER $bitDepth $outputFormatToggle $case 0"
./BatchPD_host_pln3 $SRC_FOLDER_1 $SRC_FOLDER_2 $DST_FOLDER $bitDepth $outputFormatToggle $case 0
echo "------------------------------------------------------------------------------------------"
done
done
done

mkdir $DST_FOLDER/PLN3
mv $DST_FOLDER/!(PKD3|PLN1|PLN3) $DST_FOLDER/PLN3




printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "Running all Unique functionalities..."
echo "##########################################################################################"

printf "\n\nUsage: ./uniqueFunctionalities_host <case number = 1:1>"

for ((case=1;case<3;case++))
do
printf "\n\n\n\n"
echo "--------------------------------"
printf "Running a New Functionality...\n"
echo "--------------------------------"
echo "./uniqueFunctionalities_host 0 $case"
./uniqueFunctionalities_host 0 $case
echo "------------------------------------------------------------------------------------------"
done