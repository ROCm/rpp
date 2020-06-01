# RPP UNITTESTS
## For Host
* Usage - PKD3 tests:
```
cd HOST
mkdir build
cd build
cmake ..
make
./BatchPD_host_pkd3 <src1 folder> <src2 folder (same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2> <case number = 0:64>
```
* Usage - PLN1 tests:
```
cd HOST
mkdir build
cd build
cmake ..
make
./BatchPD_host_pln1 <src1 folder> <src2 folder (same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2> <case number = 0:64>
```

## For OCL

## For HIP


## Run
Have this in run.sh file
```
for i in $(seq 0 2); do 
for j in $(seq 1 7); do
./BatchPD_ocl /path/to/first/input/folder/ /path/to/second/input/folder/ /path/to/output/folder $i $j
    done
done

```