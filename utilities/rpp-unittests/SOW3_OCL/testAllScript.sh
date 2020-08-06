#!/bin/bash
# <<<<<<<<<<<<<< JUST CHANGE SOURCE AND DESTINATION FOLDERS >>>>>>>>>>>>>>
SRC_FOLDER_1="~/Lokesh/fol1/"
SRC_FOLDER_2="~/Lokesh/fol1/"
DST_FOLDER="./"

SRC_FOLDER_1_pln="~/Lokesh/fol1/"
SRC_FOLDER_2_pln="~/Lokesh/fol1/"

# <<<<<<<<<<<<<< JUST CHANGE SOURCE AND DESTINATION FOLDERS >>>>>>>>>>>>>>
shopt -s extglob
cd build
printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "Running all PKD3 Inputs..."
echo "##########################################################################################"
for ((case=1;case<8;case++))
do
printf "\n\n\n\n"
echo "--------------------------------"
printf "Running a New Functionality...\n"
echo "--------------------------------"
for ((bitDepth=0;bitDepth<7;bitDepth++))
do
printf "\n\n\nRunning New Bit Depth..."
for ((outputFormatToggle=0;outputFormatToggle<2;outputFormatToggle++))
do
./BatchPD_ocl_pkd3 $SRC_FOLDER_1 $SRC_FOLDER_2 $DST_FOLDER $bitDepth $outputFormatToggle $case
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
for ((case=1;case<8;case++))
do
printf "\n\n\n\n"
echo "--------------------------------"
printf "Running a New Functionality...\n"
echo "--------------------------------"
for ((bitDepth=0;bitDepth<7;bitDepth++))
do
printf "\n\n\nRunning New Bit Depth..."
for ((outputFormatToggle=0;outputFormatToggle<1;outputFormatToggle++))
do
./BatchPD_ocl_pln1 $SRC_FOLDER_1_pln $SRC_FOLDER_2_pln $DST_FOLDER $bitDepth $outputFormatToggle $case
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
for ((case=1;case<8;case++))
do
printf "\n\n\n\n"
echo "--------------------------------"
printf "Running a New Functionality...\n"
echo "--------------------------------"
for ((bitDepth=0;bitDepth<7;bitDepth++))
do
printf "\n\n\nRunning New Bit Depth..."
for ((outputFormatToggle=0;outputFormatToggle<2;outputFormatToggle++))
do
./BatchPD_ocl_pln3 $SRC_FOLDER_1 $SRC_FOLDER_2 $DST_FOLDER $bitDepth $outputFormatToggle $case
echo "------------------------------------------------------------------------------------------"
done
done
done
mkdir $DST_FOLDER/PLN3
mv $DST_FOLDER/!(PKD3|PLN1|PLN3) $DST_FOLDER/PLN3




