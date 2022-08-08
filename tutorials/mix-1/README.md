# Readme for linear acoustics

# Introduction

## Data (default)

## Running code

Unzip data file with images stack
```
unzip  out_arrays.txt.zip
```

```
./phase -file_name mesh.cub -k 1 -base_order 2 \
	-order_savitzky_golay 2 -window_savitzky_golay 5 \
	-window_shift -1  && mbconvert out_0.h5m out_0.vtk
```

## Extra input

Extra input commands can be found in ./param_file.petsc