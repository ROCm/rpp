# RPP UNITTESTS
## For HOST - Test all functionalities with default images

```
cd utilities/rpp-unittests/HOST
mkdir build
cd build
cmake ..
make
cd ..
./testAllScript.sh
```

## For HOST - Test all functionalities with 

Edit "testAllScript.sh" to set the correct SRC1, SRC2 and DST paths.

```
# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
SRC_FOLDER_1="<your complete folder path that containes images for  srcPtr1>"
SRC_FOLDER_2="<your complete folder path that containes images for  srcPtr2 (can be same as SRC_FOLDER_1 for trials)>"
DST_FOLDER="<your complete folder path where the dstPtr images can be dumped>"
# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
```

Run the script

```
./testAllScript.sh
```


## For HOST - 7 specific tensor-support functionalities with output format toggling options
* Running all cases using testAllScript.sh:

  This script runs host tests for all the sub-cases for the 7 specific tensor-support functionalities - resize, crop, resize_crop, resize_crop_mirror, rotate, crop_mirror_normalize and color_twist.

  * Build the test code:
    ```
    cd SOW3_HOST
    mkdir build
    cd build
    cmake ..
    make
    cd ..
    ```
  * Edit first three lines of testAllScript.sh to point to the right source and destination folders and save it:
    ```
    # <<<<<<<<<<<<<< JUST CHANGE SOURCE AND DESTINATION FOLDERS >>>>>>>>>>>>>>
    SRC_FOLDER_1="/home/abishek/abishek/tests/Input_Images/temp224x224"
    SRC_FOLDER_2="/home/abishek/abishek/tests/Input_Images/temp224x224"
    DST_FOLDER="/home/abishek/abishek/tests/Output_Images"
    # <<<<<<<<<<<<<< JUST CHANGE SOURCE AND DESTINATION FOLDERS >>>>>>>>>>>>>>

```
  * Run the script:
    ```
    ./testAllScript.sh
    ```
  The script should create viewable outputs for a total of (37 * 2) + (37 * 2) + 33 = 181 cases that include:
  *  74 cases for pkd3 input:
      * 21 cases (7 cases each of resize, crop, crop_mirror_normalize) + 16 cases (4 cases each of rotate, resize_crop, resize_crop_mirror, color_twist)
      * Each of the above 37 cases * 2 = 74 (for both pkd3->pkd3 and pkd3->pln3 tests)
  *  74 cases for pln3 input:
      * 21 cases (7 cases each of resize, crop, crop_mirror_normalize) + 16 cases (4 cases each of rotate, resize_crop, resize_crop_mirror, color_twist)
      * Each of the above 37 cases * 2 = 74 (for both pln3->pln3 and pln3->pkd3 tests)
  *  33 cases for pln1 input:
      * 21 cases (7 cases each of resize, crop, crop_mirror_normalize) + 12 cases (4 cases each of rotate, resize_crop, resize_crop_mirror) - not applicable for color_twist

* Running individual cases:

  Individual cases can be run from command line with the following guideline:
  ```
    ./BatchPD_host_pkd3 <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 1:7>
  ```
  ```
    ./BatchPD_host_pln3 <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 1:7>
  ```
  ```
    ./BatchPD_host_pln1 <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 1:7>
  ```

## For OCL - Test all functionalities

## For HIP - Test all functionalities


## Run
Have this in run.sh file
```
for i in $(seq 0 2); do 
for j in $(seq 1 7); do
./BatchPD_ocl /path/to/first/input/folder/ /path/to/second/input/folder/ /path/to/output/folder $i $j
    done
done

```
