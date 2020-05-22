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


